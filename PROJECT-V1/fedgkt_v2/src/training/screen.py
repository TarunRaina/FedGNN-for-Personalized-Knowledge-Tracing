"""
src/training/screen.py

Fixed-budget screening loop for the v2 improvement campaign.

Every v2 variant -- the noise floor, each hyperparameter setting, each
architecture change -- runs through this, so all of them are compared on
identical terms.

WHY A SEPARATE LOOP AND NOT centralised_batched.train_batched()
----------------------------------------------------------------
train_batched() is the production loop: early stopping on patience,
validation every epoch through the locked CPU evaluator. Both are right
for a real run and wrong for screening:

  - Early stopping makes runs incomparable. A variant that stops at
    epoch 14 and one that stops at epoch 40 have had different amounts
    of training, so their final numbers cannot be set side by side.
    Screening therefore uses a FIXED epoch budget with no early stopping.

  - CPU validation costs ~195s of every epoch. Over a 25-epoch screen
    that is ~80 minutes of the ~3.5 hours. The batched GPU evaluator
    does the same work in ~69s (2.8x), verified to agree with the locked
    evaluator: predictions identical (BCE difference 0.0) and macro AUC
    within 2.8e-05, which is ~350x smaller than the ~0.01 effects being
    measured. Validating every other epoch cuts it further.

The TRAINING step itself is NOT reimplemented here. This loop calls
centralised_batched.train_one_epoch_batched() unchanged -- the same
function, with the same per-student 1/length loss weighting, the same
length-sorted batching and the same chunked-backward behaviour that
produced the canonical result. Only the surrounding loop differs.

WHAT IT RECORDS
---------------
    <run_dir>/history.json    every epoch: loss, and val macro AUC where measured
    <run_dir>/best.pt         weights at the best validated epoch
    <run_dir>/latest.pt       full resume state, rewritten every epoch
    <run_dir>/screen.json     the summary line for comparing variants

RESUME
------
Colab disconnects during a 3-hour run are routine, so full state --
weights, optimiser, RNG, history -- is written atomically after every
epoch. Re-running the same command continues from the next epoch.
Resuming is refused if the settings differ from the saved run, so two
different variants can never be spliced into one.

COMPARING VARIANTS
------------------
Compare at the SAME epoch, not at each run's own best. `final_val` (the
last validated epoch) is the primary comparison number; `best_val` is
recorded too but is biased upward by however many times validation
happened to be measured.

And compare against the NOISE FLOOR, not against zero: the same variant
run with two different seeds will differ by some amount purely by
chance. Until that amount is known, no difference means anything.
"""

import copy
import json
import os
import random
import sys
import time

import numpy as np
import torch

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))       # src/training
_SRC_DIR = os.path.dirname(_THIS_DIR)                          # src
_PROJECT_ROOT = os.path.dirname(_SRC_DIR)                      # fedgkt_v2/
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.utils import config as cfg
from src.training.centralised_batched import train_one_epoch_batched
from src.training.evaluator import evaluate
from src.training.evaluator_batched import evaluate_batched


# ── small helpers ────────────────────────────────────────────────────────
def _atomic_torch_save(obj, path):
    tmp = path + '.tmp'
    torch.save(obj, tmp)
    os.replace(tmp, path)


def _atomic_json_save(obj, path):
    tmp = path + '.tmp'
    with open(tmp, 'w') as f:
        json.dump(obj, f, indent=2)
    os.replace(tmp, path)


def _cpu_copy(state_dict):
    return {k: v.detach().cpu().clone() for k, v in state_dict.items()}


def _get_rng_state():
    state = {
        'python': random.getstate(),
        'numpy': np.random.get_state(),
        'torch': torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state['cuda'] = torch.cuda.get_rng_state_all()
    return state


def _set_rng_state(state):
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'])
    if 'cuda' in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state['cuda'])


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _fingerprint(variant, lr, batch_size, backward_chunk_size, val_every,
                 seed, batched_eval):
    """
    Settings a resumed run must share with the run it continues. `epochs`
    is excluded so a screen can be extended; everything that changes what
    is computed is included.
    """
    return {
        'variant': variant,
        'lr': lr,
        'batch_size': batch_size,
        'backward_chunk_size': backward_chunk_size,
        'val_every': val_every,
        'seed': seed,
        'batched_eval': batched_eval,
    }


# ── the screen ───────────────────────────────────────────────────────────
def run_screen(model, train_ids, val_ids, run_dir, *, variant='baseline',
               epochs=25, val_every=2, batch_size=8, backward_chunk_size=50,
               lr=None, seed=42, device=None, edge_index=None,
               eval_batch_size=16, batched_eval=True, resume=True, verbose=True):
    """
    Trains for a FIXED number of epochs with NO early stopping, validating
    every `val_every` epochs (and always on the final epoch). Returns the
    summary dict, which is also written to <run_dir>/screen.json.
    """
    assert len(train_ids) > 0 and len(val_ids) > 0
    assert epochs > 0 and val_every > 0

    lr = cfg.LEARNING_RATE if lr is None else lr
    device = device if device is not None else torch.device(
        'cuda' if torch.cuda.is_available() else 'cpu')

    os.makedirs(run_dir, exist_ok=True)
    best_path = os.path.join(run_dir, 'best.pt')
    latest_path = os.path.join(run_dir, 'latest.pt')
    history_path = os.path.join(run_dir, 'history.json')
    summary_path = os.path.join(run_dir, 'screen.json')

    fingerprint = _fingerprint(variant, lr, batch_size, backward_chunk_size,
                               val_every, seed, batched_eval)

    if edge_index is None:
        edge_index = torch.load(cfg.EDGE_INDEX_PATH, weights_only=False)

    model = model.to(device)
    set_seed(seed)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    start_epoch = 1
    best_val = -np.inf
    best_epoch = 0
    best_state = None
    history = []

    # ── resume ────────────────────────────────────────────────────────
    if os.path.exists(latest_path):
        if not resume:
            raise RuntimeError(
                f"{latest_path} exists and resume=False. Refusing to overwrite "
                f"an existing screen. Delete {run_dir} to start fresh.")
        ckpt = torch.load(latest_path, map_location='cpu', weights_only=False)
        if ckpt['fingerprint'] != fingerprint:
            diffs = {k: (ckpt['fingerprint'].get(k), fingerprint.get(k))
                     for k in set(ckpt['fingerprint']) | set(fingerprint)
                     if ckpt['fingerprint'].get(k) != fingerprint.get(k)}
            raise RuntimeError(
                f"Cannot resume: settings differ from the saved screen in "
                f"{run_dir}.\n  (saved, current): {diffs}\n"
                f"Resuming would mix two different variants into one run.")
        model.load_state_dict(ckpt['model_state'])
        optimizer.load_state_dict(ckpt['optimizer_state'])
        best_val = ckpt['best_val']
        best_epoch = ckpt['best_epoch']
        best_state = ckpt['best_state']
        history = ckpt['history']
        start_epoch = ckpt['epoch'] + 1
        _set_rng_state(ckpt['rng_state'])
        assert len(history) == ckpt['epoch'], (
            f"BUG: checkpoint says epoch {ckpt['epoch']} but holds "
            f"{len(history)} history entries.")
        if verbose:
            print(f"RESUMING '{variant}' from epoch {ckpt['epoch']} "
                  f"(best so far: epoch {best_epoch}, val {best_val:.4f})")

    eval_name = 'batched GPU' if batched_eval else 'locked CPU'
    if verbose:
        print("=" * 70)
        print(f"SCREEN '{variant}'   {epochs} epochs, no early stopping")
        print(f"  device={device}  lr={lr}  batch_size={batch_size}  seed={seed}")
        print(f"  validating every {val_every} epoch(s) with the {eval_name} evaluator")
        print(f"  run_dir={run_dir}")
        print("=" * 70)

    run_start = time.time()

    for epoch in range(start_epoch, epochs + 1):
        t0 = time.time()
        train_loss = train_one_epoch_batched(
            model, optimizer, train_ids, edge_index, batch_size, device,
            backward_chunk_size=backward_chunk_size)
        train_seconds = time.time() - t0
        assert np.isfinite(train_loss), f"BUG: train loss is {train_loss}"

        # validate every val_every epochs, and always on the last one
        do_val = (epoch % val_every == 0) or (epoch == epochs)
        val_auc = None
        val_seconds = 0.0

        if do_val:
            t1 = time.time()
            if batched_eval:
                val = evaluate_batched(model, val_ids, edge_index=edge_index,
                                       batch_size=eval_batch_size, device=device)
                model = model.to(device)
            else:
                model_cpu = model.to('cpu')
                val = evaluate(model_cpu, val_ids,
                               edge_index=edge_index.to('cpu'), verbose=False)
                model = model_cpu.to(device)
            val_seconds = time.time() - t1
            val_auc = val['macro_auc']
            assert np.isfinite(val_auc), f"BUG: val macro AUC is {val_auc}"

            if val_auc > best_val:
                best_val = val_auc
                best_epoch = epoch
                best_state = _cpu_copy(model.state_dict())
                _atomic_torch_save(best_state, best_path)

        history.append({
            'epoch': epoch,
            'train_loss': train_loss,
            'val_macro_auc': val_auc,          # None on unvalidated epochs
            'train_seconds': round(train_seconds, 1),
            'val_seconds': round(val_seconds, 1),
        })

        _atomic_torch_save({
            'epoch': epoch,
            'model_state': _cpu_copy(model.state_dict()),
            'optimizer_state': optimizer.state_dict(),
            'best_state': best_state,
            'best_val': best_val,
            'best_epoch': best_epoch,
            'history': history,
            'rng_state': _get_rng_state(),
            'fingerprint': fingerprint,
        }, latest_path)
        _atomic_json_save(history, history_path)

        if verbose:
            val_str = f"val={val_auc:.4f}" if val_auc is not None else "val=   -  "
            mark = "  <-- best" if (val_auc is not None and epoch == best_epoch) else ""
            print(f"epoch {epoch:>3}/{epochs}  loss={train_loss:.4f}  {val_str}  "
                  f"train={train_seconds:.0f}s val={val_seconds:.0f}s{mark}")

    assert best_state is not None, (
        "BUG: no epoch was ever validated -- check epochs/val_every.")

    validated = [h for h in history if h['val_macro_auc'] is not None]
    final_val = validated[-1]['val_macro_auc']
    total_seconds = time.time() - run_start

    summary = {
        'variant': variant,
        'seed': seed,
        'epochs': epochs,
        'val_every': val_every,
        'lr': lr,
        'batch_size': batch_size,
        'backward_chunk_size': backward_chunk_size,
        'batched_eval': batched_eval,
        'final_val_macro_auc': final_val,       # primary comparison number
        'final_val_epoch': validated[-1]['epoch'],
        'best_val_macro_auc': best_val,
        'best_val_epoch': best_epoch,
        'n_validations': len(validated),
        'train_students': len(train_ids),
        'val_students': len(val_ids),
        'total_seconds_this_session': round(total_seconds, 1),
        'history': history,
    }
    _atomic_json_save(summary, summary_path)

    if verbose:
        print("-" * 70)
        print(f"'{variant}' done.  final val (epoch {summary['final_val_epoch']}): "
              f"{final_val:.4f}   best (epoch {best_epoch}): {best_val:.4f}")
        print(f"this session: {total_seconds / 60:.0f} min")
        print(f"written: {summary_path}")
        print("-" * 70)
        print("Compare variants on final_val at the SAME epoch, and only against")
        print("the noise floor -- two seeds of the same variant differ by chance.")

    return summary