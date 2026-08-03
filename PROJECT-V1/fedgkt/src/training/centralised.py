"""
src/training/centralised.py

The centralised (single-machine) training loop for FedGKT Phase 1.

Design decisions locked before this file was written (see conversation
record -- flagged and confirmed before any code was written, per this
project's working style):

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
   more influence than a student with 30. Students are the unit of analysis
   for this project (FL clients, stratified sampling), so this is the
   correct inductive bias.

3. DEVICE: CPU by default (see config.py, cfg.DEVICE) -- graphs this small,
   processed sequentially, would not clearly benefit from GPU transfer
   overhead per call.

4. Only the raw (unweighted) per-interaction loss is used for the reported
   "avg_train_loss" metric -- the tiny weighted value used internally for
   backward() would not be an interpretable number to look at in logs.

5. Best model (by val macro_auc) is checkpointed to disk during training,
   and the best weights are reloaded into the model before this module
   returns -- so callers get the best-performing version, not just
   whatever the last epoch happened to produce (which may have overfit
   past the best point).
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
    average RAW (unweighted) per-interaction BCE loss for the epoch, for
    logging/interpretability.
    """
    model.train()

    shuffled_ids = list(train_ids)
    rng.shuffle(shuffled_ids)

    raw_loss_sum = 0.0
    raw_loss_count = 0

    for group_start in range(0, len(shuffled_ids), group_size):
        group = shuffled_ids[group_start:group_start + group_size]
        group_actual_size = len(group)  # last group may be smaller than group_size

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
                pred = model(pkg.get_x(), edge_index, ex)  # scalar tensor, WITH grad
                target = torch.tensor(c, dtype=torch.float32)

                raw_loss = bce_loss(pred, target)

                # per-student equal weighting: divide by (this student's
                # sequence length x number of students in this group), so
                # every student contributes an equal-magnitude gradient
                # regardless of how many interactions they have.
                weighted_loss = raw_loss / (n * group_actual_size)
                weighted_loss.backward()  # accumulates into .grad, frees this
                                            # step's graph immediately -- does
                                            # NOT zero_grad(), by design (see
                                            # module docstring point 1)

                raw_loss_sum += raw_loss.item()
                raw_loss_count += 1

                # feature update happens AFTER the forward pass that used the
                # pre-update state -- no label leakage (same contract as
                # evaluator.py and the original pkg.py manual verification)
                pkg.update(ex, c, t)

        # one optimizer step per group, after every student in the group has
        # contributed their (already-weighted) gradient via backward() above
        optimizer.step()

    assert raw_loss_count > 0, "BUG: no interactions were processed this epoch"
    avg_train_loss = raw_loss_sum / raw_loss_count
    return avg_train_loss


def train(model, train_ids, val_ids, edge_index=None, num_epochs=None,
          group_size=None, patience=None, checkpoint_dir=None,
          checkpoint_name='best_model.pt', verbose=True):
    """
    Full training run: repeatedly calls train_one_epoch(), evaluates on
    val_ids after every epoch, checkpoints the best model (by val macro_auc),
    and stops early if val macro_auc hasn't improved for `patience` epochs.

    Before returning, reloads the BEST checkpoint into `model` (not
    necessarily the last epoch's weights) -- so the caller always gets the
    best-performing version seen during training.

    Returns a dict: {
        'history': [ {epoch, train_loss, val_macro_auc, val_bce,
                       train_seconds, val_seconds}, ... ],
        'best_epoch': int,
        'best_val_auc': float,
        'checkpoint_path': str,
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

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.LEARNING_RATE)
    rng = random.Random(cfg.RANDOM_SEED)

    history = []
    best_val_auc = None
    best_epoch = None
    patience_counter = 0
    stopped_early = False

    if verbose:
        print(f"Training config: epochs={num_epochs}  group_size={group_size}  "
              f"patience={patience}  lr={cfg.LEARNING_RATE}  "
              f"train_students={len(train_ids)}  val_students={len(val_ids)}")

    for epoch in range(1, num_epochs + 1):
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
            if patience_counter >= patience:
                if verbose:
                    print(f"\nEarly stopping: val_auc has not improved for "
                          f"{patience} epochs (best was epoch {best_epoch}, "
                          f"val_auc={best_val_auc:.4f})")
                stopped_early = True
                break

    # reload the BEST checkpoint before returning -- not necessarily the
    # last epoch's weights, since training may have continued past the best
    # point before triggering early stopping (or simply finished all epochs
    # with the best result not being the final one)
    assert os.path.exists(checkpoint_path), (
        f"BUG: no checkpoint was ever saved at {checkpoint_path} -- "
        f"this should be impossible since at least epoch 1 always sets a new best"
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
        'stopped_early': stopped_early,
    }


if __name__ == '__main__':
    print("=" * 70)
    print("centralised.py -- standalone self-test (REAL data, tiny scale)")
    print("=" * 70)

    from src.models.fedgkt import FedGKT

    # ── part 1: verify early-stopping counter logic in isolation ────────────
    print("\n--- Part 1: early-stopping logic (isolated, synthetic sequence) ---")
    synthetic_val_aucs = [0.55, 0.60, 0.58, 0.57, 0.56, 0.55, 0.54]  # best is epoch 2
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

    print(f"  Synthetic val_auc sequence: {synthetic_val_aucs}")
    print(f"  patience={test_patience}  ->  best_epoch={best_ep} (auc={best})  "
          f"stopped_at_epoch={stopped_at}")
    assert best_ep == 2 and best == 0.60, f"BUG: expected best_epoch=2 (0.60), got {best_ep} ({best})"
    assert stopped_at == 5, f"BUG: expected early stop at epoch 5, got {stopped_at}"
    print("  Early-stopping counter logic verified correct -- OK")

    # ── part 2: real end-to-end run, tiny scale ──────────────────────────────
    print("\n--- Part 2: real end-to-end training run (2 train students, 1 val student) ---")

    tiny_train_ids = [233536, 45224]   # confirmed 'train' in student_splits.json
    tiny_val_ids = [21419]              # confirmed 'val' in student_splits.json

    missing = [
        uid for uid in tiny_train_ids + tiny_val_ids
        if not os.path.exists(os.path.join(cfg.PKG_DIR, f'pkg_{uid}.pt'))
    ]
    assert not missing, f"Missing PKG files for self-test: {missing}"

    torch.manual_seed(cfg.RANDOM_SEED)
    model = FedGKT()

    result = train(
        model,
        train_ids=tiny_train_ids,
        val_ids=tiny_val_ids,
        num_epochs=3,
        group_size=2,
        patience=5,
        checkpoint_dir=os.path.join(_PROJECT_ROOT, 'checkpoints_selftest'),
        checkpoint_name='selftest_model.pt',
        verbose=True,
    )

    print(f"\n--- Result summary ---")
    print(f"Best epoch: {result['best_epoch']}")
    print(f"Best val AUC: {result['best_val_auc']:.4f}")
    print(f"Stopped early: {result['stopped_early']}")
    print(f"Checkpoint saved at: {result['checkpoint_path']}")
    print(f"Checkpoint file exists on disk: {os.path.exists(result['checkpoint_path'])}")

    assert os.path.exists(result['checkpoint_path']), "BUG: checkpoint file was not actually saved to disk"
    assert len(result['history']) >= 1, "BUG: training history is empty"
    assert 0.0 <= result['best_val_auc'] <= 1.0, "BUG: best_val_auc outside [0,1]"

    for h in result['history']:
        print(f"  epoch {h['epoch']}: train_loss={h['train_loss']:.4f}  "
              f"val_auc={h['val_macro_auc']:.4f}  "
              f"train_time={h['train_seconds']:.1f}s  val_time={h['val_seconds']:.1f}s")

    print("\nSelf-test complete -- no assertion failures.")
    print("\nNOTE: this used only 2 train / 1 val student for 3 epochs as a fast")
    print("plumbing check. The real run (train_baseline.py, next file) will use")
    print("the full 800 train / 100 val students.")