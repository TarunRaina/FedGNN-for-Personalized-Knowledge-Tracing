"""
src/training/centralised_batched.py

GPU-batched training loop for FedGKT -- the Colab speed experiment. This is
NOT used by the local sequential training run (src/training/centralised.py,
which remains the system of record for Phase 1's actual reported results).

DESIGN: fully mirrors centralised.py's resume-support design, adapted for
batched training:
- Best model (by val macro_auc) checkpointed every epoch it improves.
- Patience-based early stopping (cfg.EARLY_STOPPING_PATIENCE).
- FULL resumable state saved every epoch: model weights, optimizer state,
  PyTorch's internal RNG state (governs dropout -- this was the exact thing
  missed the first time resume was built for centralised.py, and caught
  only by testing an interrupted-vs-uninterrupted comparison; the same
  discipline is applied here from the start rather than risking the same
  bug twice), best-so-far tracking, patience counter, and history.
- resume_from restores all of this and continues from the next epoch.

NOTE ON RNG: unlike centralised.py, this file does NOT need to save/restore
a Python `random.Random` shuffle state, because batching here uses
DETERMINISTIC length-sorted bucketing (see _make_length_sorted_batches),
not random shuffling -- shuffling would defeat the point of bucketing
similar-length students together. So batch composition is already fully
reproducible from train_ids alone. PyTorch's RNG (dropout) is the only
stochastic element, and it IS saved/restored, same lesson as before.

OTHER DESIGN NOTES (unchanged from the original version):
- fedgkt.py needed one small, already-verified fix (range-check assertion
  now derives its bound from the actual embeddings tensor size rather than
  a hardcoded cfg.NUM_NODES) -- otherwise unchanged, works for both single-
  graph and batched/flattened multi-graph use without a mode flag.
- edge_index for a batch of B students is built once per distinct B
  encountered (via build_batched_edge_index) and reused for every batch of
  that size.
- Students are sorted by sequence length and grouped into batches of up to
  B consecutive students in that order ("bucketing"), to minimize wasted
  padding compute vs. a randomly-mixed batch.
- Per-student equal weighting: each student's loss at each step is divided
  by (their own total real sequence length x B) -- same normalization
  principle as centralised.py's (student_length x group_size) divisor.
- evaluator.py is used UNCHANGED for validation (sequential, CPU) -- model
  is moved to CPU for that call and back to the training device afterward.
"""

import os
import sys
import time
import json

import torch
import torch.nn.functional as F

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))       # src/training
_SRC_DIR = os.path.dirname(_THIS_DIR)                          # src
_PROJECT_ROOT = os.path.dirname(_SRC_DIR)                      # fedgkt_colab_staging/
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.utils import config as cfg
from src.data.pkg_batched import BatchedPersonalKnowledgeGraph
from src.training.evaluator import evaluate


def build_batched_edge_index(edge_index, B, num_nodes, device):
    """
    Replicates edge_index B times, offsetting node indices in copy i by
    i * num_nodes, so the result correctly represents B disjoint copies of
    the same graph living inside one flattened [B*num_nodes, F] node tensor.
    """
    edge_index = edge_index.to(device)
    offsets = (torch.arange(B, device=device) * num_nodes).view(B, 1, 1)
    replicated = edge_index.unsqueeze(0).expand(B, 2, -1) + offsets
    batched = replicated.permute(1, 0, 2).reshape(2, -1)
    return batched


def _load_student_sequence(user_id):
    pkg_path = os.path.join(cfg.PKG_DIR, f'pkg_{user_id}.pt')
    assert os.path.exists(pkg_path), f"BUG: PKG file not found for user_id={user_id} at {pkg_path}"
    data = torch.load(pkg_path, weights_only=False)
    return data['exercise_idx'], data['correct'], data['time_done']


def _make_length_sorted_batches(student_ids, batch_size):
    """
    Sorts students by their real interaction count (DETERMINISTIC, no
    randomness -- see module docstring on why this file doesn't need a
    shuffle-RNG resume path), then chunks the sorted list into groups of
    up to batch_size.
    """
    lengths = {}
    for uid in student_ids:
        ex_seq, _, _ = _load_student_sequence(uid)
        lengths[uid] = ex_seq.shape[0]

    sorted_ids = sorted(student_ids, key=lambda uid: lengths[uid])
    batches = [sorted_ids[i:i + batch_size] for i in range(0, len(sorted_ids), batch_size)]
    return batches, lengths


def train_one_epoch_batched(model, optimizer, train_ids, edge_index, batch_size, device,
                             backward_chunk_size=50):
    """
    Runs one full pass over train_ids, grouped into length-sorted batches.
    Returns the average RAW (unweighted) per-interaction BCE loss for the
    epoch, for logging/interpretability -- same convention as centralised.py.

    backward_chunk_size: how many consecutive timesteps' losses get summed
    together before calling .backward() once. backward_chunk_size=1
    reproduces the ORIGINAL per-step-backward behavior exactly (kept as a
    verification baseline / fallback -- see the self-test below, which
    compares gradients between chunk_size=1 and the real default to prove
    they are mathematically equivalent, not just asserted to be).

    TWO GPU-UTILIZATION FIXES applied here (identified after observing low
    GPU utilization on Colab despite plenty of free VRAM headroom):

    FIX 1 (data transfer): padded_ex/padded_correct/padded_time/
    active_matrix are moved to `device` ONCE per batch, not once per step.
    The original version called .to(device) inside the per-step loop --
    a fresh CPU->GPU transfer at every single timestep, hundreds of
    thousands of times per epoch. Each transfer has fixed per-call latency
    regardless of how little data moves; doing this once per batch instead
    eliminates that overhead while producing byte-identical values at each
    step (same data, just transferred once instead of repeatedly).

    FIX 2 (backward frequency): losses from `backward_chunk_size`
    consecutive steps are summed into one scalar before calling
    .backward(), instead of calling .backward() after every single step.
    optimizer.step() timing is UNCHANGED -- still exactly once per batch,
    same as before. By linearity of differentiation, sum-then-backward on
    a chunk of N steps produces the IDENTICAL accumulated gradient as N
    separate backward() calls, one per step -- what changes is purely how
    many times the (comparatively expensive, GPU-kernel-launch-heavy)
    backward() call itself gets invoked, not which values get summed into
    the gradient. Forward passes still happen one step at a time, in
    strict order, since each step's forward pass depends on the PKG state
    left by the previous step's update() call -- only the backward()
    TRIGGER frequency changes, never the sequential nature of the forward
    computation or the PKG update logic itself.

    Chunk size is kept modest (default 50) specifically so the accumulated
    autograd graph never grows unboundedly -- the same memory-safety
    principle that motivated per-step backward() in the original design,
    just recalibrated for a GPU with much more headroom (Colab's T4, 16GB,
    vs. the 8GB RAM / ~4GB VRAM laptop that motivated the original,
    stricter per-step design). 50 steps' worth of this model's tiny
    per-student graph is a trivial memory footprint even accumulated
    together, nowhere near the memory pressure that motivated the original.
    """
    model.train()

    batches, lengths = _make_length_sorted_batches(train_ids, batch_size)

    raw_loss_sum = 0.0
    raw_loss_count = 0

    edge_index_cache = {}

    for batch_ids in batches:
        B = len(batch_ids)
        if B not in edge_index_cache:
            edge_index_cache[B] = build_batched_edge_index(edge_index, B, cfg.NUM_NODES, device)
        batched_edge_index = edge_index_cache[B]

        sequences = [_load_student_sequence(uid) for uid in batch_ids]
        real_lengths = torch.tensor([lengths[uid] for uid in batch_ids], dtype=torch.float32, device=device)
        max_len = int(real_lengths.max().item())

        padded_ex = torch.zeros((B, max_len), dtype=torch.int64)
        padded_correct = torch.zeros((B, max_len), dtype=torch.float32)
        padded_time = torch.zeros((B, max_len), dtype=torch.int64)
        active_matrix = torch.zeros((B, max_len), dtype=torch.bool)

        for i, (ex_seq, c_seq, t_seq) in enumerate(sequences):
            L = ex_seq.shape[0]
            padded_ex[i, :L] = ex_seq
            padded_correct[i, :L] = c_seq
            padded_time[i, :L] = t_seq
            active_matrix[i, :L] = True
            if L < max_len:
                padded_ex[i, L:] = ex_seq[-1]
                padded_correct[i, L:] = 0.0
                padded_time[i, L:] = t_seq[-1]

        # FIX 1: transfer once per batch, not once per step (see docstring)
        padded_ex = padded_ex.to(device)
        padded_correct = padded_correct.to(device)
        padded_time = padded_time.to(device)
        active_matrix = active_matrix.to(device)

        pkg_batch = BatchedPersonalKnowledgeGraph(batch_size=B, device=device)
        batch_offsets = (torch.arange(B, device=device) * cfg.NUM_NODES)

        normalizer = real_lengths.to(device) * B

        optimizer.zero_grad()

        # FIX 2: accumulate losses across chunks before backward() (see docstring)
        chunk_loss_accum = None
        chunk_step_count = 0

        for step in range(max_len):
            active_mask = active_matrix[:, step]
            step_time = padded_time[:, step]
            step_ex = padded_ex[:, step]
            step_correct = padded_correct[:, step]

            pkg_batch.refresh_time_decay(step_time, active_mask)

            x_flat = pkg_batch.get_x().reshape(B * cfg.NUM_NODES, cfg.NUM_FEATURES).clone()
            # .clone() is required here (not just a nicety) once backward() is
            # deferred across multiple steps: pkg_batch.x is mutated IN-PLACE
            # by update() every step. Without cloning, autograd would try to
            # differentiate through memory that's been overwritten by later
            # steps' update() calls by the time .backward() actually runs --
            # PyTorch correctly detects and refuses this (RuntimeError: in-
            # place modification), rather than silently computing a wrong
            # gradient. Cloning gives each step's forward pass an independent
            # snapshot that later mutations can never touch, restoring exact
            # equivalence with the original immediate-backward design. Cost
            # is negligible (a few KB per step).
            flat_idx = step_ex + batch_offsets

            preds = model(x_flat, batched_edge_index, flat_idx)

            eps = 1e-7
            preds_clamped = preds.clamp(eps, 1 - eps)
            per_student_loss = F.binary_cross_entropy(preds_clamped, step_correct, reduction='none')

            weighted = per_student_loss / normalizer
            masked_weighted = weighted * active_mask.float()
            step_loss = masked_weighted.sum()

            chunk_loss_accum = step_loss if chunk_loss_accum is None else (chunk_loss_accum + step_loss)
            chunk_step_count += 1

            n_active = int(active_mask.sum().item())
            if n_active > 0:
                raw_loss_sum += (per_student_loss.detach() * active_mask.float()).sum().item()
                raw_loss_count += n_active

            pkg_batch.update(step_ex, step_correct, step_time, active_mask)

            is_last_step = (step == max_len - 1)
            if chunk_step_count >= backward_chunk_size or is_last_step:
                if chunk_loss_accum is not None and chunk_loss_accum.requires_grad:
                    chunk_loss_accum.backward()
                chunk_loss_accum = None
                chunk_step_count = 0

        optimizer.step()

    assert raw_loss_count > 0, "BUG: no interactions were processed this epoch"
    avg_train_loss = raw_loss_sum / raw_loss_count
    return avg_train_loss


def train_batched(model, train_ids, val_ids, edge_index=None, num_epochs=None,
                   batch_size=32, backward_chunk_size=50, patience=None, device=None,
                   checkpoint_dir=None,
                   checkpoint_name='best_model_batched.pt',
                   resume_checkpoint_name='latest_state_batched.pt',
                   resume_from=None, verbose=True):
    """
    Full batched training run, mirroring centralised.py's train() design.

    resume_from: path to a resume-state file (as saved by this function
                 every epoch), or None to start fresh.

    Returns a dict: {
        'history': [...], 'best_epoch': int, 'best_val_auc': float,
        'checkpoint_path': str, 'resume_checkpoint_path': str,
        'stopped_early': bool, 'total_seconds': float,
    }
    """
    device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_epochs = cfg.NUM_EPOCHS if num_epochs is None else num_epochs
    patience = cfg.EARLY_STOPPING_PATIENCE if patience is None else patience
    checkpoint_dir = checkpoint_dir if checkpoint_dir is not None else cfg.CHECKPOINT_DIR

    assert len(train_ids) > 0, "BUG: train_ids is empty"
    assert len(val_ids) > 0, "BUG: val_ids is empty"
    assert num_epochs > 0, "BUG: num_epochs must be positive"
    assert patience > 0, "BUG: patience must be positive"

    if edge_index is None:
        edge_index = torch.load(cfg.EDGE_INDEX_PATH, weights_only=False)

    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, checkpoint_name)
    resume_checkpoint_path = os.path.join(checkpoint_dir, resume_checkpoint_name)

    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.LEARNING_RATE)

    if resume_from is not None:
        assert os.path.exists(resume_from), f"BUG: resume_from file not found: {resume_from}"
        saved = torch.load(resume_from, weights_only=False, map_location=device)

        model.load_state_dict(saved['model_state_dict'])
        optimizer.load_state_dict(saved['optimizer_state_dict'])
        torch.set_rng_state(saved['torch_rng_state'].cpu())

        best_val_auc = saved['best_val_auc']
        best_epoch = saved['best_epoch']
        patience_counter = saved['patience_counter']
        history = saved['history']
        stopped_early = saved.get('stopped_early', False)
        start_epoch = saved['epoch'] + 1

        if verbose:
            print(f"Resumed from {resume_from}: last completed epoch={saved['epoch']}, "
                  f"best_epoch={best_epoch} (val_auc={best_val_auc:.4f}), "
                  f"patience_counter={patience_counter}/{patience}, "
                  f"continuing from epoch {start_epoch}")
    else:
        best_val_auc = None
        best_epoch = None
        patience_counter = 0
        history = []
        stopped_early = False
        start_epoch = 1

    if verbose:
        print(f"Batched training config: epochs={num_epochs}  batch_size={batch_size}  "
              f"backward_chunk_size={backward_chunk_size}  patience={patience}  device={device}  "
              f"train_students={len(train_ids)}  val_students={len(val_ids)}")

    run_start = time.time()

    if stopped_early:
        if verbose:
            print(f"\nResumed state already triggered early stopping -- nothing more to train.")
    elif start_epoch > num_epochs:
        if verbose:
            print(f"\nResumed state already completed all {num_epochs} epochs -- nothing more to train.")
    else:
        for epoch in range(start_epoch, num_epochs + 1):
            t0 = time.time()
            avg_train_loss = train_one_epoch_batched(
                model, optimizer, train_ids, edge_index, batch_size, device,
                backward_chunk_size=backward_chunk_size
            )
            train_seconds = time.time() - t0

            t1 = time.time()
            model_cpu = model.to('cpu')
            cpu_edge_index = torch.load(cfg.EDGE_INDEX_PATH, weights_only=False)
            val_result = evaluate(model_cpu, val_ids, edge_index=cpu_edge_index, verbose=False)
            model = model_cpu.to(device)
            val_seconds = time.time() - t1

            val_auc = val_result['macro_auc']
            is_new_best = (best_val_auc is None) or (val_auc > best_val_auc)

            if verbose:
                marker = "  <-- new best" if is_new_best else ""
                print(f"Epoch {epoch:>3}/{num_epochs}  train_loss={avg_train_loss:.4f}  "
                      f"val_auc={val_auc:.4f} ({val_result['macro_n_valid']}/{val_result['macro_n_total']} valid)  "
                      f"train_time={train_seconds:.1f}s  val_time={val_seconds:.1f}s{marker}")

            history.append({
                'epoch': epoch,
                'train_loss': avg_train_loss,
                'val_macro_auc': val_auc,
                'train_seconds': train_seconds,
                'val_seconds': val_seconds,
            })

            if is_new_best:
                best_val_auc = val_auc
                best_epoch = epoch
                patience_counter = 0
                torch.save(model.state_dict(), checkpoint_path)
            else:
                patience_counter += 1

            resume_state = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'torch_rng_state': torch.get_rng_state(),
                'best_val_auc': best_val_auc,
                'best_epoch': best_epoch,
                'patience_counter': patience_counter,
                'history': history,
                'stopped_early': False,
            }

            if patience_counter >= patience:
                if verbose:
                    print(f"\nEarly stopping: val_auc has not improved for {patience} epochs "
                          f"(best was epoch {best_epoch}, val_auc={best_val_auc:.4f})")
                stopped_early = True
                resume_state['stopped_early'] = True
                torch.save(resume_state, resume_checkpoint_path)
                break

            torch.save(resume_state, resume_checkpoint_path)

    total_seconds = time.time() - run_start

    assert os.path.exists(checkpoint_path), (
        f"BUG: no checkpoint was ever saved at {checkpoint_path}"
    )
    model.load_state_dict(torch.load(checkpoint_path, weights_only=True, map_location=device))
    if verbose:
        print(f"\nReloaded best checkpoint (epoch {best_epoch}, val_auc={best_val_auc:.4f}) "
              f"from {checkpoint_path}")

    history_path = os.path.join(checkpoint_dir, 'batched_training_history.json')
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    if verbose:
        print(f"Total wall-clock time this call: {total_seconds:.1f}s ({total_seconds/60:.1f} min)")
        print(f"Saved: {history_path}")

    return {
        'history': history,
        'best_epoch': best_epoch,
        'best_val_auc': best_val_auc,
        'checkpoint_path': checkpoint_path,
        'resume_checkpoint_path': resume_checkpoint_path,
        'stopped_early': stopped_early,
        'total_seconds': total_seconds,
    }


if __name__ == '__main__':
    print("=" * 70)
    print("centralised_batched.py -- self-tests")
    print("=" * 70)

    from src.models.fedgkt import FedGKT

    real_ids = [233536, 45224]
    missing = [
        uid for uid in real_ids
        if not os.path.exists(os.path.join(cfg.PKG_DIR, f'pkg_{uid}.pt'))
    ]
    assert not missing, f"Missing PKG files for self-test: {missing}"

    device = torch.device('cpu')

    # ── Test 0: gradient equivalence between chunk_size=1 and chunk_size=50 ──
    print("\n--- Test 0: gradient equivalence, backward_chunk_size=1 vs 50 ---")
    print("Verifies the two GPU-utilization fixes (batched data transfer +")
    print("chunked backward calls) produce the mathematically IDENTICAL")
    print("gradient as the original per-step-backward version, not just")
    print("similar-looking training numbers.\n")

    edge_index = torch.load(cfg.EDGE_INDEX_PATH, weights_only=False)

    torch.manual_seed(cfg.RANDOM_SEED)
    model_chunk1 = FedGKT()
    optimizer_chunk1 = torch.optim.Adam(model_chunk1.parameters(), lr=cfg.LEARNING_RATE)
    train_one_epoch_batched(model_chunk1, optimizer_chunk1, real_ids, edge_index,
                             batch_size=2, device=device, backward_chunk_size=1)
    grads_chunk1 = {name: p.grad.clone() for name, p in model_chunk1.named_parameters()}

    torch.manual_seed(cfg.RANDOM_SEED)
    model_chunk50 = FedGKT()
    optimizer_chunk50 = torch.optim.Adam(model_chunk50.parameters(), lr=cfg.LEARNING_RATE)
    train_one_epoch_batched(model_chunk50, optimizer_chunk50, real_ids, edge_index,
                             batch_size=2, device=device, backward_chunk_size=50)
    grads_chunk50 = {name: p.grad.clone() for name, p in model_chunk50.named_parameters()}

    assert set(grads_chunk1.keys()) == set(grads_chunk50.keys()), "BUG: parameter sets differ"

    max_grad_diff = 0.0
    for name in grads_chunk1:
        diff = (grads_chunk1[name] - grads_chunk50[name]).abs().max().item()
        max_grad_diff = max(max_grad_diff, diff)
        print(f"  {name}: max grad diff = {diff:.2e}")
        assert diff < 1e-5, (
            f"BUG: gradient diverges for parameter '{name}' between chunk_size=1 "
            f"and chunk_size=50 -- the chunking fix is NOT gradient-equivalent, "
            f"do not trust it."
        )

    print(f"\nMax gradient difference across all parameters: {max_grad_diff:.2e}")
    print("(Small nonzero values expected -- float32 summation order differs")
    print("slightly between many-small-backward-calls vs sum-then-backward,")
    print("same class of platform/order rounding drift seen elsewhere in this")
    print("project -- not a correctness bug at this magnitude.)")
    print("Gradient equivalence verified -- the chunking optimization is safe.")

    # ── Test 1: resume correctness, now using the new default chunk size ────
    print("\n" + "=" * 70)
    print("--- Test 1: resume correctness (using new default backward_chunk_size=50) ---")
    print("=" * 70)

    selftest_dir = os.path.join(_PROJECT_ROOT, 'checkpoints_batched_resume_selftest_v2')

    print("\n--- Uninterrupted 3-epoch run (baseline) ---")
    torch.manual_seed(cfg.RANDOM_SEED)
    model_a = FedGKT()
    result_a = train_batched(
        model_a, train_ids=real_ids, val_ids=real_ids,
        num_epochs=3, batch_size=2, patience=5, device=device,
        checkpoint_dir=os.path.join(selftest_dir, 'uninterrupted'), verbose=True,
    )

    print("\n--- Interrupted run -- 1 epoch, simulate restart, resume for 2 more ---")
    torch.manual_seed(cfg.RANDOM_SEED)
    model_b = FedGKT()
    interrupted_dir = os.path.join(selftest_dir, 'interrupted')
    train_batched(
        model_b, train_ids=real_ids, val_ids=real_ids,
        num_epochs=1, batch_size=2, patience=5, device=device,
        checkpoint_dir=interrupted_dir, verbose=True,
    )
    print("\n  [SIMULATING INTERRUPTION -- fresh model object, different seed]\n")

    torch.manual_seed(9999)
    model_b_resumed = FedGKT()
    result_b2 = train_batched(
        model_b_resumed, train_ids=real_ids, val_ids=real_ids,
        num_epochs=3, batch_size=2, patience=5, device=device,
        checkpoint_dir=interrupted_dir,
        resume_from=os.path.join(interrupted_dir, 'latest_state_batched.pt'),
        verbose=True,
    )

    print("\n--- Verifying resumed run matches uninterrupted run exactly ---")
    for h_a, h_b in zip(result_a['history'], result_b2['history']):
        diff_loss = abs(h_a['train_loss'] - h_b['train_loss'])
        diff_auc = abs(h_a['val_macro_auc'] - h_b['val_macro_auc'])
        print(f"  epoch {h_a['epoch']}: train_loss diff={diff_loss:.2e}  val_auc diff={diff_auc:.2e}")
        assert diff_loss < 1e-6, f"BUG: train_loss diverges at epoch {h_a['epoch']}"
        assert diff_auc < 1e-6, f"BUG: val_auc diverges at epoch {h_a['epoch']}"

    assert result_a['best_epoch'] == result_b2['best_epoch']
    assert abs(result_a['best_val_auc'] - result_b2['best_val_auc']) < 1e-6

    print("\nAll epochs match exactly. Resume correctness verified with the new chunking fix.")
    print("\nAll self-tests complete -- no assertion failures.")