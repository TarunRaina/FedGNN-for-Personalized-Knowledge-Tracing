"""
src/training/centralised.py

The centralised (single-machine) training loop for FedGKT Phase 1.

Design decisions locked before this file was written (flagged and confirmed
before any code was written, per this project's working style):

1. MEMORY: naively accumulating BCE loss across an entire student's
   interaction sequence before calling .backward() once would require
   holding every forward pass's computational graph in memory
   simultaneously -- infeasible for students with thousands of interactions
   (the working subset's max is 9,548) on 8GB RAM / ~4GB VRAM hardware.
   FIX: call loss.backward() after EVERY interaction, without calling
   optimizer.zero_grad() in between. PyTorch accumulates gradients
   additively across backward() calls by default, and by linearity of
   differentiation this produces the mathematically IDENTICAL final
   gradient as summing all losses first and backpropping once -- just
   without ever holding more than one interaction's graph in memory.

2. PER-STUDENT EQUAL WEIGHTING: each interaction's loss is divided by
   (that student's total interaction count x the number of students in the
   current gradient-accumulation group) before backward(). This gives every
   student equal weight in the resulting gradient step regardless of
   sequence length -- a student with 9,548 interactions does not get ~300x
   more influence than a student with 30.

3. DEVICE: CPU by default (cfg.DEVICE) -- graphs this small, processed
   sequentially, would not clearly benefit from GPU transfer overhead per
   call.

4. Only the raw (unweighted) per-interaction loss is used for the reported
   "avg_train_loss" metric -- the tiny weighted value used internally for
   backward() would not be interpretable in logs.

5. Best model (by val macro_auc) is checkpointed to disk during training,
   and the best weights are reloaded into the model before this module
   returns -- so callers get the best-performing version, not whatever the
   last epoch happened to produce.

6. RESUME SUPPORT (added after the above was already verified working):
   a real run over 800 students for up to ~20 epochs could take hours on a
   personal laptop, with a real risk of interruption (sleep, accidental
   close, crash). To make an interruption non-catastrophic, a full resumable
   state (model weights, optimizer state -- including Adam's per-parameter
   momentum buffers, RNG state for shuffling, epoch number, best-so-far
   tracking, and history) is saved to disk after EVERY epoch, not just on
   improvement. Passing that file's path as `resume_from` restores all of
   this and continues training from the next epoch onward.

   Correctness requirement for resume: resuming from an interruption must
   produce IDENTICAL results to an uninterrupted run of the same total
   epoch count, given the same seed. This is only true if model weights,
   optimizer state, the shuffling RNG state, AND PyTorch's own internal RNG
   state (which governs dropout -- a separate random stream from Python's
   `random` module) are all restored exactly. This was caught empirically,
   not just reasoned about in advance: an early version of this file
   restored the shuffle RNG correctly but not PyTorch's RNG, and a
   resumed run measurably diverged from an uninterrupted run starting at
   the very next epoch after the simulated interruption (dropout masks
   differed silently, with no error raised). Both RNG streams are now
   saved and restored, and this exact equivalence is verified in the
   self-test below.
"""

import os
import sys
import time
import json
import random

import torch

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))       # src/training
_SRC_DIR = os.path.dirname(_THIS_DIR)                          # src
_PROJECT_ROOT = os.path.dirname(_SRC_DIR)                      # fedgkt/
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.utils import config as cfg
from src.utils.metrics import bce_loss
from src.data.pkg import PersonalKnowledgeGraph
from src.training.evaluator import evaluate


def _load_student_sequence(user_id):
    pkg_path = os.path.join(cfg.PKG_DIR, f'pkg_{user_id}.pt')
    assert os.path.exists(pkg_path), f"BUG: PKG file not found for user_id={user_id} at {pkg_path}"
    data = torch.load(pkg_path, weights_only=False)
    return data['exercise_idx'], data['correct'], data['time_done']


def train_one_epoch(model, optimizer, train_ids, edge_index, group_size, rng):
    """
    Runs one full pass over train_ids (shuffled), using the gradient
    accumulation design described in the module docstring. Returns the
    average RAW (unweighted) per-interaction BCE loss for the epoch.
    """
    model.train()

    shuffled_ids = list(train_ids)
    rng.shuffle(shuffled_ids)

    raw_loss_sum = 0.0
    raw_loss_count = 0

    for group_start in range(0, len(shuffled_ids), group_size):
        group = shuffled_ids[group_start:group_start + group_size]
        group_actual_size = len(group)

        optimizer.zero_grad()

        for user_id in group:
            exercise_idx_seq, correct_seq, time_done_seq = _load_student_sequence(user_id)
            n = exercise_idx_seq.shape[0]
            assert n > 0, f"BUG: student {user_id} has zero interactions"

            pkg = PersonalKnowledgeGraph()

            for step in range(n):
                ex = int(exercise_idx_seq[step].item())
                c = float(correct_seq[step].item())
                t = int(time_done_seq[step].item())

                pkg.refresh_time_decay(t)
                pred = model(pkg.get_x(), edge_index, ex)
                target = torch.tensor(c, dtype=torch.float32)

                raw_loss = bce_loss(pred, target)
                weighted_loss = raw_loss / (n * group_actual_size)
                weighted_loss.backward()

                raw_loss_sum += raw_loss.item()
                raw_loss_count += 1

                pkg.update(ex, c, t)

        optimizer.step()

    assert raw_loss_count > 0, "BUG: no interactions were processed this epoch"
    avg_train_loss = raw_loss_sum / raw_loss_count
    return avg_train_loss


def train(model, train_ids, val_ids, edge_index=None, num_epochs=None,
          group_size=None, patience=None, checkpoint_dir=None,
          checkpoint_name='best_model.pt',
          resume_checkpoint_name='latest_state.pt',
          resume_from=None, verbose=True):
    """
    Full training run. See module docstring for the resume-support design.

    resume_from: path to a resume-state file (as saved by this function
                 every epoch), or None to start fresh. If provided, model
                 weights, optimizer state, RNG state, epoch counter, and
                 all tracking variables are restored, and training
                 continues from the next epoch onward.

    Returns a dict: {
        'history': [...], 'best_epoch': int, 'best_val_auc': float,
        'checkpoint_path': str, 'resume_checkpoint_path': str,
        'stopped_early': bool,
    }
    """
    num_epochs = cfg.NUM_EPOCHS if num_epochs is None else num_epochs
    group_size = cfg.STUDENTS_PER_GRADIENT_STEP if group_size is None else group_size
    patience = cfg.EARLY_STOPPING_PATIENCE if patience is None else patience
    checkpoint_dir = cfg.CHECKPOINT_DIR if checkpoint_dir is None else checkpoint_dir

    assert len(train_ids) > 0, "BUG: train_ids is empty"
    assert len(val_ids) > 0, "BUG: val_ids is empty"
    assert group_size > 0, "BUG: group_size must be positive"
    assert num_epochs > 0, "BUG: num_epochs must be positive"
    assert patience > 0, "BUG: patience must be positive"

    if edge_index is None:
        assert os.path.exists(cfg.EDGE_INDEX_PATH), f"BUG: edge_index.pt not found at {cfg.EDGE_INDEX_PATH}"
        edge_index = torch.load(cfg.EDGE_INDEX_PATH, weights_only=False)

    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, checkpoint_name)
    resume_checkpoint_path = os.path.join(checkpoint_dir, resume_checkpoint_name)

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.LEARNING_RATE)

    # ── fresh start vs. resume ────────────────────────────────────────────
    if resume_from is not None:
        assert os.path.exists(resume_from), f"BUG: resume_from file not found: {resume_from}"
        saved = torch.load(resume_from, weights_only=False)

        model.load_state_dict(saved['model_state_dict'])
        optimizer.load_state_dict(saved['optimizer_state_dict'])

        rng = random.Random()
        rng.setstate(saved['rng_state'])
        torch.set_rng_state(saved['torch_rng_state'])

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
        rng = random.Random(cfg.RANDOM_SEED)
        best_val_auc = None
        best_epoch = None
        patience_counter = 0
        history = []
        stopped_early = False
        start_epoch = 1

    if verbose:
        print(f"Training config: epochs={num_epochs}  group_size={group_size}  "
              f"patience={patience}  lr={cfg.LEARNING_RATE}  "
              f"train_students={len(train_ids)}  val_students={len(val_ids)}")

    # ── epoch loop (may start partway through if resuming) ──────────────────
    if stopped_early:
        if verbose:
            print(f"\nResumed state already triggered early stopping at epoch "
                  f"{best_epoch + patience} -- nothing more to train.")
    elif start_epoch > num_epochs:
        if verbose:
            print(f"\nResumed state already completed all {num_epochs} epochs -- "
                  f"nothing more to train.")
    else:
        for epoch in range(start_epoch, num_epochs + 1):
            t0 = time.time()
            avg_train_loss = train_one_epoch(model, optimizer, train_ids, edge_index, group_size, rng)
            train_seconds = time.time() - t0

            t1 = time.time()
            val_result = evaluate(model, val_ids, edge_index=edge_index, verbose=False)
            val_seconds = time.time() - t1

            val_auc = val_result['macro_auc']
            val_bce = val_result['overall_bce']

            is_new_best = (best_val_auc is None) or (val_auc > best_val_auc)

            if verbose:
                marker = "  <-- new best" if is_new_best else ""
                print(f"Epoch {epoch:>3}/{num_epochs}  "
                      f"train_loss={avg_train_loss:.4f}  "
                      f"val_auc={val_auc:.4f} ({val_result['macro_n_valid']}/{val_result['macro_n_total']} valid)  "
                      f"val_bce={val_bce:.4f}  "
                      f"train_time={train_seconds:.1f}s  val_time={val_seconds:.1f}s{marker}")

            history.append({
                'epoch': epoch,
                'train_loss': avg_train_loss,
                'val_macro_auc': val_auc,
                'val_overall_bce': val_bce,
                'val_n_valid': val_result['macro_n_valid'],
                'val_n_total': val_result['macro_n_total'],
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

            # save FULL resumable state every epoch, regardless of whether
            # this epoch was a new best -- this is what makes resume possible
            # after ANY completed epoch, not just improving ones
            resume_state = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'rng_state': rng.getstate(),
                'torch_rng_state': torch.get_rng_state(),
                'best_val_auc': best_val_auc,
                'best_epoch': best_epoch,
                'patience_counter': patience_counter,
                'history': history,
                'stopped_early': False,  # updated below if this epoch triggers it
            }

            if patience_counter >= patience:
                if verbose:
                    print(f"\nEarly stopping: val_auc has not improved for "
                          f"{patience} epochs (best was epoch {best_epoch}, "
                          f"val_auc={best_val_auc:.4f})")
                stopped_early = True
                resume_state['stopped_early'] = True
                torch.save(resume_state, resume_checkpoint_path)
                break

            torch.save(resume_state, resume_checkpoint_path)

    # ── reload BEST checkpoint before returning ──────────────────────────────
    assert os.path.exists(checkpoint_path), (
        f"BUG: no checkpoint was ever saved at {checkpoint_path} -- "
        f"this should be impossible since at least one epoch always sets a new best"
    )
    model.load_state_dict(torch.load(checkpoint_path, weights_only=True))
    if verbose:
        print(f"\nReloaded best checkpoint (epoch {best_epoch}, val_auc={best_val_auc:.4f}) "
              f"from {checkpoint_path}")

    history_path = os.path.join(checkpoint_dir, 'training_history.json')
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    if verbose:
        print(f"Saved training history: {history_path}")

    return {
        'history': history,
        'best_epoch': best_epoch,
        'best_val_auc': best_val_auc,
        'checkpoint_path': checkpoint_path,
        'resume_checkpoint_path': resume_checkpoint_path,
        'stopped_early': stopped_early,
    }


if __name__ == '__main__':
    print("=" * 70)
    print("centralised.py -- standalone self-test (REAL data, tiny scale)")
    print("=" * 70)

    from src.models.fedgkt import FedGKT

    # ── part 1: early-stopping logic (isolated, synthetic sequence) ─────────
    print("\n--- Part 1: early-stopping logic (isolated, synthetic sequence) ---")
    synthetic_val_aucs = [0.55, 0.60, 0.58, 0.57, 0.56, 0.55, 0.54]
    test_patience = 3

    best = None
    best_ep = None
    counter = 0
    stopped_at = None
    for ep, auc in enumerate(synthetic_val_aucs, start=1):
        is_best = (best is None) or (auc > best)
        if is_best:
            best = auc
            best_ep = ep
            counter = 0
        else:
            counter += 1
            if counter >= test_patience:
                stopped_at = ep
                break

    print(f"  patience={test_patience}  ->  best_epoch={best_ep} (auc={best})  stopped_at_epoch={stopped_at}")
    assert best_ep == 2 and best == 0.60
    assert stopped_at == 5
    print("  Early-stopping counter logic verified correct -- OK")

    tiny_train_ids = [45224]
    tiny_val_ids = [233536]
    # NOTE: using only the two SMALL real students here (106 and 30
    # interactions) rather than including the 9,548-interaction student.
    # This part of the self-test verifies the RESUME MECHANISM is correct
    # (bit-for-bit reproducibility across an interruption), not realistic
    # val performance -- that was already verified separately with the full
    # 2 train / 1 val real-student setup earlier in this project. Including
    # the large student here would just make this particular test slow
    # (multiple validation passes needed) without adding any correctness
    # signal specific to what THIS test checks.

    missing = [
        uid for uid in tiny_train_ids + tiny_val_ids
        if not os.path.exists(os.path.join(cfg.PKG_DIR, f'pkg_{uid}.pt'))
    ]
    assert not missing, f"Missing PKG files for self-test: {missing}"

    selftest_dir = os.path.join(_PROJECT_ROOT, 'checkpoints_selftest')

    # ── part 2: uninterrupted 3-epoch run (baseline for comparison) ─────────
    print("\n--- Part 2: uninterrupted 3-epoch run (baseline) ---")
    torch.manual_seed(cfg.RANDOM_SEED)
    model_a = FedGKT()
    result_a = train(
        model_a, train_ids=tiny_train_ids, val_ids=tiny_val_ids,
        num_epochs=3, group_size=2, patience=5,
        checkpoint_dir=os.path.join(selftest_dir, 'uninterrupted'),
        checkpoint_name='model.pt', verbose=True,
    )

    # ── part 3: interrupted run -- train 1 epoch, "restart", resume for 2 more ──
    print("\n--- Part 3: interrupted run -- 1 epoch, simulate restart, resume for 2 more ---")
    torch.manual_seed(cfg.RANDOM_SEED)
    model_b = FedGKT()
    interrupted_dir = os.path.join(selftest_dir, 'interrupted')
    result_b1 = train(
        model_b, train_ids=tiny_train_ids, val_ids=tiny_val_ids,
        num_epochs=1, group_size=2, patience=5,
        checkpoint_dir=interrupted_dir, checkpoint_name='model.pt', verbose=True,
    )
    print("\n  [SIMULATING INTERRUPTION HERE -- building a fresh model object,")
    print("   as if the process had been killed and restarted]\n")

    # a brand new model object, with DIFFERENT random init weights than
    # model_b's original init -- deliberately, to prove resume restores
    # everything needed from the checkpoint file, not relying on the Python
    # object still holding leftover state from before the "interruption"
    torch.manual_seed(9999)
    model_b_resumed = FedGKT()

    result_b2 = train(
        model_b_resumed, train_ids=tiny_train_ids, val_ids=tiny_val_ids,
        num_epochs=3, group_size=2, patience=5,
        checkpoint_dir=interrupted_dir, checkpoint_name='model.pt',
        resume_from=os.path.join(interrupted_dir, 'latest_state.pt'),
        verbose=True,
    )

    # ── part 4: verify resumed run produced IDENTICAL results to uninterrupted ──
    print("\n--- Part 4: verifying resumed run matches uninterrupted run exactly ---")
    for h_a, h_b in zip(result_a['history'], result_b2['history']):
        print(f"  epoch {h_a['epoch']}: uninterrupted train_loss={h_a['train_loss']:.8f}  "
              f"resumed train_loss={h_b['train_loss']:.8f}  "
              f"diff={abs(h_a['train_loss'] - h_b['train_loss']):.2e}")
        assert abs(h_a['train_loss'] - h_b['train_loss']) < 1e-6, (
            f"BUG: resumed run's train_loss diverges from uninterrupted run at "
            f"epoch {h_a['epoch']} -- resume is not restoring state correctly"
        )
        assert abs(h_a['val_macro_auc'] - h_b['val_macro_auc']) < 1e-6, (
            f"BUG: resumed run's val_auc diverges from uninterrupted run at "
            f"epoch {h_a['epoch']}"
        )

    assert result_a['best_epoch'] == result_b2['best_epoch']
    assert abs(result_a['best_val_auc'] - result_b2['best_val_auc']) < 1e-6

    print("\n  All epochs match exactly between uninterrupted and resumed runs.")
    print("  Resume correctness verified -- OK")

    print("\nSelf-test complete -- no assertion failures.")