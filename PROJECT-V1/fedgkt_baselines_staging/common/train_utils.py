"""
common/train_utils.py

Shared training + evaluation loop for all five pyKT baselines
(DKT, DKVMN, SAKT, AKT, GKT). Every rule that must be identical across
the five lives here, exactly once, so it cannot drift between drivers.

WHAT IT DOES
------------
  1. Builds the train / validation / test data loaders.
  2. Trains one epoch at a time using pyKT's OWN training step
     (model_forward), so the forward pass and loss are exactly pyKT's.
  3. After every epoch, scores the 100 validation students by MACRO AUC
     (one AUC per student, then averaged) -- FedGKT's locked metric.
  4. Keeps the epoch with the best validation macro AUC; stops after
     `patience` epochs without improvement, or at `max_epochs`.
  5. Reloads that best epoch and scores the 100 test students on the
     WINDOWED test file, reporting both macro and pooled AUC.

WHY NOT pyKT's OWN train_model()  (verified against 0.0.38)
-----------------------------------------------------------
pyKT's train_model() picks its best epoch like this:

    auc, acc = evaluate(model, valid_loader, model.model_name)
    if auc > max_auc+1e-3:

and evaluate() returns a single POOLED roc_auc_score over every
interaction. Using it would choose each baseline's "best" epoch by a
metric we do not report, while FedGKT's epoch 68 was chosen by
validation macro AUC. It also has no early stopping at all -- it runs
every epoch to num_epochs. So this file supplies its own epoch loop,
and reuses pyKT's model_forward() for the training step itself, which
keeps the actual learning computation byte-for-byte pyKT's.

HOW PREDICTIONS ARE MATCHED TO STUDENTS  (the delicate part)
-----------------------------------------------------------
Macro AUC needs to know which student each prediction belongs to.
pyKT's batches do NOT carry the student id: KTDataset.__getitem__
returns cseqs / rseqs / masks / smasks and their shifted versions, and
nothing else.

It can be recovered safely, because every link in the chain preserves
row order:
  - KTDataset filters the CSV with df[df["fold"].isin(folds)], which
    keeps the original row order, then walks it with iterrows().
  - The evaluation loaders here are built with shuffle=False
    explicitly, so batch k holds dataset rows k*B .. k*B+B-1, in order.
  - torch.masked_select(y, sm) flattens row by row.

So applying the SAME mask to a [B, L] tensor holding each row's uid
yields a uid for every prediction, in exactly the order of the
predictions.

That argument is not trusted on its own. Before any model runs, the
expected targets and uids at every scored position are computed
directly from the CSV (independent of pyKT), and every evaluation pass
asserts that pyKT's targets and our uids reproduce them exactly. If
row order ever broke, targets would stop matching, and the run would
stop instead of silently mis-assigning predictions to the wrong
students. This re-check runs on EVERY evaluation, not just the first.

CROSS-CHECK AGAINST pyKT's OWN EVALUATE
---------------------------------------
The per-model forward dispatch below is copied from pyKT's evaluate()
branch by branch. To prove it matches, the final test pass also calls
pyKT's own evaluate() on the same loader and asserts that its pooled
AUC equals ours. If our dispatch disagreed with pyKT's for any model,
this catches it. A small tolerance (1e-6) is used here rather than
exact equality: on GPU, some kernels are not bit-deterministic between
two passes, unlike the CPU-only FedGKT replay where exact 0.0 was
required.

PARITY SETTINGS  (the defaults below mirror FedGKT's config.py)
---------------------------------------------------------------
  optimiser Adam, lr 1e-3, early-stopping patience 5, selection on
  validation macro AUC, improvement = any strict increase
  (min_delta 0.0, as FedGKT), max 100 epochs, seed 42.
  No shuffling in any model (locked): FedGKT's canonical GPU run used
  deterministic length-sorted bucketing, and pyKT's own training loader
  does not shuffle either. Note the ORDER is not identical -- FedGKT's is
  length-sorted, the baselines' is CSV order -- so the accurate claim is
  "no model shuffles", not "same ordering as FedGKT".
Model-specific sizes (emb_size, dropout, etc.) live in each driver, set to
pyKT's own standard defaults rather than tuned.

RESUME
------
Full resume state is written to run_dir/latest_model.pt after every
epoch, atomically, so a Colab disconnect loses at most the epoch in
progress. See fit_and_evaluate().
"""

import copy
import json
import os
import random
import sys
import time

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from torch.nn.functional import one_hot
from torch.utils.data import DataLoader

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

from baseline_step2_macro_auc_eval import evaluate_macro_auc   # noqa: E402
import data_config as dc                                        # noqa: E402

from pykt.datasets.data_loader import KTDataset                 # noqa: E402
from pykt.models.train_model import model_forward               # noqa: E402
from pykt.models.evaluate_model import evaluate as pykt_evaluate  # noqa: E402


SUPPORTED_MODELS = ('dkt', 'dkvmn', 'sakt', 'akt', 'gkt')


# ── reproducibility ──────────────────────────────────────────────────────
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ── expected targets, computed straight from the CSV ─────────────────────
def expected_scored_positions(csv_path, folds):
    """
    Independent of pyKT: reads the CSV and lists, in row order, the
    target and uid of every position pyKT will score.

    pyKT scores shifted position k (0-based) of a row when
    smasks[k+1] == 1, and the target there is responses[k+1]. So we
    walk original indices 1..MAXLEN-1 and keep those whose selectmask
    is 1 -- mirroring dori["smasks"][:, 1:].

    Returns (row_uids, expected_targets, expected_uids).
    """
    df = pd.read_csv(csv_path)
    df = df[df['fold'].isin(folds)]

    row_uids = df['uid'].to_numpy(dtype=np.int64)
    exp_targets, exp_uids = [], []

    for uid, rstr, sstr in zip(df['uid'], df['responses'], df['selectmasks']):
        r = [int(v) for v in str(rstr).split(',')]
        s = [int(v) for v in str(sstr).split(',')]
        for k in range(1, len(s)):
            if s[k] == 1:
                exp_targets.append(r[k])
                exp_uids.append(int(uid))

    return (row_uids,
            np.asarray(exp_targets, dtype=np.float64),
            np.asarray(exp_uids, dtype=np.int64))


# ── loaders ──────────────────────────────────────────────────────────────
def build_loaders(data_config, batch_size, validation_fold=dc.VALIDATION_FOLD):
    """
    Builds the three loaders directly from KTDataset rather than via
    pyKT's init_dataset4train(), for two reasons verified in 0.0.38:

      - init_dataset4train() hard-returns None for the windowed test
        loader (the line is commented out). Our test numbers must come
        from the windowed file, which keeps every interaction.
      - Building them here lets shuffle=False be set explicitly on the
        evaluation loaders, which the student-matching depends on.

    The training loader also uses shuffle=False -- a locked decision.
    It matches pyKT's own init_dataset4train() (DataLoader(curtrain,
    batch_size=batch_size), no shuffle argument) and FedGKT's canonical
    GPU run, which did not shuffle either.
    """
    dpath = data_config['dpath']
    input_type = data_config['input_type']
    tv_path = os.path.join(dpath, data_config['train_valid_file'])
    test_path = os.path.join(dpath, data_config['test_window_file'])

    all_folds = set(data_config['folds'])
    assert validation_fold in all_folds, (
        f"validation_fold={validation_fold} not in data_config folds {sorted(all_folds)}"
    )
    train_folds = all_folds - {validation_fold}

    train_ds = KTDataset(tv_path, input_type, train_folds)
    valid_ds = KTDataset(tv_path, input_type, {validation_fold})
    test_ds = KTDataset(test_path, input_type, {-1})

    splits = {}
    for name, ds, path, folds in [
        ('valid', valid_ds, tv_path, {validation_fold}),
        ('test', test_ds, test_path, {-1}),
    ]:
        row_uids, exp_t, exp_u = expected_scored_positions(path, folds)
        assert len(ds) == len(row_uids), (
            f"BUG: {name} dataset has {len(ds)} rows but the CSV has "
            f"{len(row_uids)} for folds {sorted(folds)}. Row order cannot be trusted."
        )
        splits[name] = {
            'loader': DataLoader(ds, batch_size=batch_size, shuffle=False),
            'row_uids': row_uids,
            'expected_targets': exp_t,
            'expected_uids': exp_u,
            'n_students': int(len(np.unique(row_uids))),
        }

    splits['train'] = {
        'loader': DataLoader(train_ds, batch_size=batch_size, shuffle=False),
        'n_rows': len(train_ds),
    }
    return splits


# ── forward pass for evaluation, per model ───────────────────────────────
def forward_for_eval(model, model_name, dcur, device):
    """
    Copied branch-for-branch from pyKT 0.0.38's evaluate() for our five
    models. Returns (y, rshft, sm), with y already aligned to rshft/sm,
    exactly as pyKT aligns it before its own masked_select.
    """
    q, c, r = dcur['qseqs'], dcur['cseqs'], dcur['rseqs']
    qshft, cshft, rshft = dcur['shft_qseqs'], dcur['shft_cseqs'], dcur['shft_rseqs']
    sm = dcur['smasks']
    q, c, r = q.to(device), c.to(device), r.to(device)
    qshft, cshft, rshft = qshft.to(device), cshft.to(device), rshft.to(device)
    sm = sm.to(device)

    cq = torch.cat((q[:, 0:1], qshft), dim=1)
    cc = torch.cat((c[:, 0:1], cshft), dim=1)
    cr = torch.cat((r[:, 0:1], rshft), dim=1)

    if model_name == 'dkt':
        y = model(c.long(), r.long())
        y = (y * one_hot(cshft.long(), model.num_c)).sum(-1)
    elif model_name == 'dkvmn':
        y = model(cc.long(), cr.long())
        y = y[:, 1:]
    elif model_name == 'sakt':
        y = model(c.long(), r.long(), cshft.long())
    elif model_name == 'akt':
        y, _reg_loss = model(cc.long(), cr.long(), cq.long())
        y = y[:, 1:]
    elif model_name == 'gkt':
        y = model(cc.long(), cr.long())
    else:
        raise ValueError(f"Unsupported model '{model_name}'. Expected one of {SUPPORTED_MODELS}.")

    return y, rshft, sm


# ── collect every scored prediction, with its student id ─────────────────
def collect_predictions(model, model_name, split, device):
    """
    One evaluation pass. Returns flat, parallel arrays
    (predictions, targets, uids), one entry per scored position.

    Every call re-verifies the student matching: the targets pyKT
    produces, and the uids we attach, must equal the ones computed
    independently from the CSV, position for position.
    """
    model.eval()
    row_uids = split['row_uids']
    preds, targs, uids = [], [], []
    offset = 0

    with torch.no_grad():
        for dcur in split['loader']:
            y, rshft, sm = forward_for_eval(model, model_name, dcur, device)
            B = sm.shape[0]

            batch_uids = torch.as_tensor(
                row_uids[offset:offset + B], dtype=torch.long, device=sm.device
            ).unsqueeze(1).expand_as(sm)
            offset += B

            preds.append(torch.masked_select(y, sm).detach().cpu().numpy())
            targs.append(torch.masked_select(rshft, sm).detach().cpu().numpy())
            uids.append(torch.masked_select(batch_uids, sm).detach().cpu().numpy())

    assert offset == len(row_uids), (
        f"BUG: consumed {offset} rows but expected {len(row_uids)}."
    )

    preds = np.concatenate(preds).astype(np.float64)
    targs = np.concatenate(targs).astype(np.float64)
    uids = np.concatenate(uids).astype(np.int64)

    exp_t, exp_u = split['expected_targets'], split['expected_uids']
    assert len(targs) == len(exp_t), (
        f"BUG: pyKT scored {len(targs)} positions, the CSV says {len(exp_t)}."
    )
    assert np.array_equal(targs, exp_t), (
        "BUG: pyKT's targets do not match the CSV position-for-position. "
        "Row order between the loader and the CSV has broken, so predictions "
        "cannot be matched to students. Stopping rather than mis-assigning them."
    )
    assert np.array_equal(uids, exp_u), (
        "BUG: attached student ids do not match the CSV position-for-position."
    )
    assert np.all(np.isfinite(preds)), "BUG: non-finite predictions (NaN/Inf)."

    return preds, targs, uids


def score(preds, targs, uids):
    """Macro (FedGKT's locked metric) and pooled AUC on one set of predictions."""
    macro = evaluate_macro_auc(preds, targs, uids)
    pooled = float(roc_auc_score(targs, preds))
    return {
        'macro_auc': macro['macro_auc'],
        'macro_n_valid': macro['n_valid'],
        'macro_n_skipped': macro['n_skipped'],
        'pooled_auc': pooled,
        'n_scored': int(len(preds)),
        'per_student': {int(k): v for k, v in macro['per_student'].items()},
    }


# ── one training epoch, using pyKT's own step ────────────────────────────
def train_one_epoch(model, loader, opt):
    """Mirrors the inner loop of pyKT's train_model(), step for step."""
    model.train()
    losses = []
    for data in loader:
        loss = model_forward(model, data)
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(float(loss.detach().cpu().item()))
    mean_loss = float(np.mean(losses))
    assert np.isfinite(mean_loss), f"BUG: training loss is {mean_loss}."
    return mean_loss


# ── checkpointing helpers (resume support) ───────────────────────────────
def _atomic_torch_save(obj, path):
    """
    Write to a temporary file, then rename over the target. A Colab
    disconnect mid-write can then only ever leave the PREVIOUS complete
    file behind, never a half-written one that fails to load.
    """
    tmp = path + '.tmp'
    torch.save(obj, tmp)
    os.replace(tmp, path)


def _atomic_json_save(obj, path):
    tmp = path + '.tmp'
    with open(tmp, 'w') as f:
        json.dump(obj, f, indent=2)
    os.replace(tmp, path)


def _cpu_copy(state_dict):
    """Detached CPU copy -- small on disk, and loads on any device."""
    return {k: v.detach().cpu().clone() for k, v in state_dict.items()}


def _get_rng_state():
    """
    Every random-number generator that training touches. With no
    shuffling and eval-mode scoring, the only randomness left is dropout
    during training -- but restoring these makes a resumed run continue
    the same random sequence it would have used had it never stopped.
    """
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


def _fingerprint(model_name, model_config, lr, batch_size, patience, min_delta, seed):
    """
    The settings a resumed run MUST share with the run it continues.
    If any differ, resuming would silently splice two different
    experiments together -- so it is refused instead.

    max_epochs is deliberately NOT included, so a run that reached its
    epoch cap can be extended by raising max_epochs.
    """
    return {
        'model_name': model_name,
        'model_config': model_config or {},
        'lr': lr,
        'batch_size': batch_size,
        'patience': patience,
        'min_delta': min_delta,
        'seed': seed,
    }


# ── the full run ─────────────────────────────────────────────────────────
def fit_and_evaluate(model, model_name, data_config, *, batch_size, run_dir,
                     lr=1e-3, max_epochs=100, patience=5, min_delta=0.0,
                     seed=42, model_config=None, resume=True):
    """
    Trains with early stopping on validation macro AUC, then scores the
    best epoch on the windowed test set.

    Everything for one baseline goes in run_dir (on Google Drive in
    Colab, so it survives disconnects):

        best_model.pt     weights of the best epoch (validation macro AUC)
        latest_model.pt   full resume state, rewritten after EVERY epoch
        history.json      every epoch's loss and validation scores
        results.json      final test macro + pooled AUC

    RESUME
    ------
    If latest_model.pt already exists in run_dir, training continues
    from the epoch after the one it records -- model weights, optimiser
    state (Adam's running averages matter; restarting them is not a true
    resume), best-so-far, the patience counter, history and RNG state
    all restored. A run that already finished skips straight to the test
    evaluation.

    Resuming is refused if the saved settings (model, sizes, lr, batch
    size, patience, min_delta, seed) differ from the current ones, so
    two different experiments can never be silently joined. To start
    over deliberately, delete run_dir, or pass resume=False to get an
    error rather than an accidental overwrite.

    min_delta: how much validation macro AUC must rise to count as an
    improvement. 0.0 -- any strict increase -- matches FedGKT.
    """
    assert model_name in SUPPORTED_MODELS, f"Unsupported model '{model_name}'."
    assert getattr(model, 'model_name', model_name) == model_name, (
        f"BUG: model.model_name={getattr(model, 'model_name', None)!r} "
        f"does not match model_name={model_name!r}. pyKT's model_forward "
        f"dispatches on model.model_name."
    )

    device = get_device()
    model = model.to(device)
    set_seed(seed)

    os.makedirs(run_dir, exist_ok=True)
    best_path = os.path.join(run_dir, 'best_model.pt')
    latest_path = os.path.join(run_dir, 'latest_model.pt')
    history_path = os.path.join(run_dir, 'history.json')
    results_path = os.path.join(run_dir, 'results.json')

    fingerprint = _fingerprint(model_name, model_config, lr, batch_size,
                               patience, min_delta, seed)

    print("=" * 70)
    print(f"Training {model_name.upper()}   device={device}")
    print(f"Run folder: {run_dir}")
    print("=" * 70)

    splits = build_loaders(data_config, batch_size)
    print(f"train rows={splits['train']['n_rows']:,}  "
          f"valid students={splits['valid']['n_students']}  "
          f"test students={splits['test']['n_students']}")
    print(f"scored positions: valid={len(splits['valid']['expected_targets']):,}  "
          f"test={len(splits['test']['expected_targets']):,}")

    opt = torch.optim.Adam(model.parameters(), lr=lr)

    # ── fresh-run defaults ────────────────────────────────────────────
    start_epoch = 1
    best_auc = -np.inf
    best_epoch = 0
    best_state = None
    epochs_without_improvement = 0
    history = []
    finished = False
    stopped_reason = 'max_epochs'

    # ── resume, if a previous run left state behind ───────────────────
    if os.path.exists(latest_path):
        if not resume:
            raise RuntimeError(
                f"{latest_path} already exists and resume=False. Refusing to "
                f"overwrite an existing run. Delete {run_dir} to start fresh."
            )

        # map_location='cpu': the RNG states must stay CPU tensors for
        # set_rng_state. Model and optimiser states are moved to the
        # right device by load_state_dict itself.
        ckpt = torch.load(latest_path, map_location='cpu', weights_only=False)

        if ckpt['fingerprint'] != fingerprint:
            diffs = {k: (ckpt['fingerprint'].get(k), fingerprint.get(k))
                     for k in set(ckpt['fingerprint']) | set(fingerprint)
                     if ckpt['fingerprint'].get(k) != fingerprint.get(k)}
            raise RuntimeError(
                f"Cannot resume: settings differ from the saved run in "
                f"{run_dir}.\n  (saved, current): {diffs}\n"
                f"Resuming would mix two different experiments. Restore the "
                f"original settings, or delete {run_dir} to start over."
            )

        model.load_state_dict(ckpt['model_state'])
        opt.load_state_dict(ckpt['optimizer_state'])
        best_auc = ckpt['best_auc']
        best_epoch = ckpt['best_epoch']
        best_state = ckpt['best_state']
        epochs_without_improvement = ckpt['epochs_without_improvement']
        history = ckpt['history']
        finished = ckpt['finished']
        stopped_reason = ckpt['stopped_reason']
        start_epoch = ckpt['epoch'] + 1
        _set_rng_state(ckpt['rng_state'])

        assert len(history) == ckpt['epoch'], (
            f"BUG: checkpoint records epoch {ckpt['epoch']} but holds "
            f"{len(history)} history entries."
        )

        # a run that stopped only because it hit the old epoch cap can be
        # extended; one that early-stopped is genuinely done
        if finished and stopped_reason == 'max_epochs' and start_epoch <= max_epochs:
            finished = False

        print(f"\nRESUMING from epoch {ckpt['epoch']} "
              f"(best so far: epoch {best_epoch}, val macro {best_auc:.4f}, "
              f"patience {epochs_without_improvement}/{patience})")
        if finished:
            print(f"This run already finished ({stopped_reason}) -- "
                  f"skipping to test evaluation.")

    # ── training ──────────────────────────────────────────────────────
    run_start = time.time()
    if not finished:
        for epoch in range(start_epoch, max_epochs + 1):
            t0 = time.time()
            train_loss = train_one_epoch(model, splits['train']['loader'], opt)

            preds, targs, uids = collect_predictions(model, model_name, splits['valid'], device)
            val = score(preds, targs, uids)

            improved = val['macro_auc'] > best_auc + min_delta
            if improved:
                best_auc = val['macro_auc']
                best_epoch = epoch
                best_state = _cpu_copy(model.state_dict())
                epochs_without_improvement = 0
                _atomic_torch_save(best_state, best_path)
            else:
                epochs_without_improvement += 1

            elapsed = time.time() - t0
            history.append({
                'epoch': epoch,
                'train_loss': train_loss,
                'val_macro_auc': val['macro_auc'],
                'val_pooled_auc': val['pooled_auc'],
                'improved': bool(improved),
                'seconds': round(elapsed, 1),
            })

            if epochs_without_improvement >= patience:
                finished = True
                stopped_reason = f'early_stopping (patience {patience})'
            elif epoch == max_epochs:
                finished = True
                stopped_reason = 'max_epochs'

            # saved AFTER every epoch: a disconnect loses at most the
            # epoch in progress, never a completed one
            _atomic_torch_save({
                'epoch': epoch,
                'model_state': _cpu_copy(model.state_dict()),
                'optimizer_state': opt.state_dict(),
                'best_state': best_state,
                'best_auc': best_auc,
                'best_epoch': best_epoch,
                'epochs_without_improvement': epochs_without_improvement,
                'history': history,
                'finished': finished,
                'stopped_reason': stopped_reason,
                'rng_state': _get_rng_state(),
                'fingerprint': fingerprint,
            }, latest_path)
            _atomic_json_save(history, history_path)

            marker = '  *best*' if improved else f'  (no gain {epochs_without_improvement}/{patience})'
            print(f"epoch {epoch:3d}  loss={train_loss:.4f}  "
                  f"val macro={val['macro_auc']:.4f}  pooled={val['pooled_auc']:.4f}  "
                  f"{elapsed:.1f}s{marker}")

            if finished:
                break

    assert best_state is not None, "BUG: no epoch was ever recorded as best."
    model.load_state_dict(best_state)
    print(f"\nStopped: {stopped_reason}.  Best epoch {best_epoch}, "
          f"val macro AUC {best_auc:.4f}.  This session: {time.time() - run_start:.0f}s")

    # ── final test on the best epoch ─────────────────────────────────
    print("\n--- Test set (windowed file), best epoch ---")
    preds, targs, uids = collect_predictions(model, model_name, splits['test'], device)
    test = score(preds, targs, uids)

    pykt_pooled, _ = pykt_evaluate(model, splits['test']['loader'], model_name)
    diff = abs(pykt_pooled - test['pooled_auc'])
    assert diff < 1e-6, (
        f"BUG: pyKT's own evaluate() gives pooled AUC {pykt_pooled:.8f}, ours "
        f"gives {test['pooled_auc']:.8f} (difference {diff:.2e}). Our forward "
        f"dispatch for '{model_name}' disagrees with pyKT's."
    )
    print(f"pyKT's own evaluate() agrees on pooled AUC (difference {diff:.1e}).")

    print(f"\n  macro  AUC : {test['macro_auc']:.4f}   "
          f"({test['macro_n_valid']} students, {test['macro_n_skipped']} skipped as single-class)")
    print(f"  pooled AUC : {test['pooled_auc']:.4f}")
    print(f"  scored     : {test['n_scored']:,} predictions")

    results = {
        'model': model_name,
        'model_config': model_config or {},
        'training': {
            'optimizer': 'Adam', 'lr': lr, 'batch_size': batch_size,
            'max_epochs': max_epochs, 'patience': patience,
            'min_delta': min_delta, 'seed': seed, 'shuffle': False,
            'selection_metric': 'validation macro AUC',
        },
        'best_epoch': best_epoch,
        'best_val_macro_auc': best_auc,
        'stopped_reason': stopped_reason,
        'epochs_run': len(history),
        'test_macro_auc': test['macro_auc'],
        'test_pooled_auc': test['pooled_auc'],
        'test_macro_n_valid': test['macro_n_valid'],
        'test_macro_n_skipped': test['macro_n_skipped'],
        'test_n_scored': test['n_scored'],
        'pykt_evaluate_pooled_auc': float(pykt_pooled),
        'history': history,
        'test_per_student_auc': test['per_student'],
    }
    _atomic_json_save(results, results_path)
    print(f"\nWritten: {results_path}")

    return results