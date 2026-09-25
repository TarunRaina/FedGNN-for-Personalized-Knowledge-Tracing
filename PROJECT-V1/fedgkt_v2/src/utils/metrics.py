"""
src/utils/metrics.py

Loss and evaluation metric helpers.

Two families of function here, kept clearly separate:

1. bce_loss() -- stays in pure torch, differentiable, used INSIDE the
   training loop's backward pass.
2. per_student_auc() / macro_auc() / per_concept_auc() -- evaluation-only,
   operate on numpy after predictions are detached from the graph, used by
   evaluator.py, never called during the backward pass itself.

Design note on macro-averaged AUC (per the project's locked evaluation
metric, "AUC-ROC -- primary, macro-averaged across test students"): this
means compute ONE AUC per student (using that student's own full sequence of
predictions vs actual outcomes), then average across students -- NOT one
AUC pooled across every interaction from every student. A student who
answers every question correctly (or every question incorrectly) has an
undefined AUC (needs both classes present) -- these students are excluded
from the average, with the count of exclusions reported, not silently
dropped or defaulted to some value.
"""

import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))       # src/utils
_SRC_DIR = os.path.dirname(_THIS_DIR)                          # src
_PROJECT_ROOT = os.path.dirname(_SRC_DIR)                      # fedgkt/
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)


def _to_numpy(x):
    """Accepts a torch tensor (possibly requiring grad), list, or numpy
    array, returns a plain 1D numpy array. Only used in the evaluation-only
    functions below -- never in bce_loss, which must stay a pure
    differentiable torch operation."""
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


# ── training loss (differentiable, pure torch) ───────────────────────────────
def bce_loss(predictions, targets):
    """
    predictions: 1D torch tensor of probabilities in [0,1] (already
                 sigmoid-activated), WITH grad -- this is the tensor that
                 gets backpropagated through.
    targets:     1D torch tensor of 0.0/1.0 floats, no grad needed.
    Returns a scalar tensor loss.
    """
    assert predictions.shape == targets.shape, (
        f"BUG: shape mismatch, predictions {tuple(predictions.shape)} vs "
        f"targets {tuple(targets.shape)}"
    )
    eps = 1e-7
    # clamp to avoid log(0) -> -inf if a prediction is exactly 0.0 or 1.0
    predictions_clamped = predictions.clamp(eps, 1.0 - eps)
    return F.binary_cross_entropy(predictions_clamped, targets)


# ── evaluation metrics (numpy, not differentiable, used AFTER detach) ────────
def per_student_auc(predictions, targets):
    """
    predictions, targets: 1D arrays/tensors for ONE student's full sequence
                           of (prediction, actual outcome) pairs.
    Returns AUC as a float, or None if undefined (all outcomes were the same
    class -- e.g. student got every question right, or every question
    wrong -- AUC requires both classes to be present).
    """
    targets_np = _to_numpy(targets)
    predictions_np = _to_numpy(predictions)

    assert len(predictions_np) == len(targets_np), (
        f"BUG: length mismatch, predictions={len(predictions_np)} "
        f"targets={len(targets_np)}"
    )

    if len(np.unique(targets_np)) < 2:
        return None  # undefined -- not an error, just not computable

    return float(roc_auc_score(targets_np, predictions_np))


def macro_auc(per_student_aucs):
    """
    per_student_aucs: list of per-student AUC values, where some entries may
                       be None (students with undefined AUC, see above).
    Returns a dict: {
        'macro_auc': float,   -- mean over only the valid (non-None) entries
        'n_valid': int,       -- how many students contributed
        'n_skipped': int,     -- how many were excluded (undefined AUC)
        'n_total': int,
    }
    """
    valid = [a for a in per_student_aucs if a is not None]
    n_total = len(per_student_aucs)
    n_valid = len(valid)
    n_skipped = n_total - n_valid

    assert n_valid > 0, (
        "BUG: every student had an undefined AUC -- cannot compute a macro "
        "average. Check that predictions/targets are being passed correctly."
    )

    return {
        'macro_auc': float(np.mean(valid)),
        'n_valid': n_valid,
        'n_skipped': n_skipped,
        'n_total': n_total,
    }


def per_concept_auc(exercise_indices, predictions, targets, num_nodes):
    """
    Pooled across however many students' interactions are passed in --
    exercise_indices, predictions, targets are PARALLEL 1D arrays, one entry
    per interaction, concatenated across all students being evaluated (NOT
    per-student -- this groups by concept, not by student).

    Returns a dict: {
        'per_concept': {concept_idx: auc, ...},  -- only concepts with both
                                                      classes present among
                                                      their interactions
        'n_concepts_with_auc': int,
        'n_concepts_skipped': int,   -- concepts seen but single-class, or
                                        never seen at all in this data
    }
    """
    exercise_indices_np = _to_numpy(exercise_indices).astype(int)
    predictions_np = _to_numpy(predictions)
    targets_np = _to_numpy(targets)

    assert len(exercise_indices_np) == len(predictions_np) == len(targets_np), (
        f"BUG: length mismatch -- exercise_indices={len(exercise_indices_np)} "
        f"predictions={len(predictions_np)} targets={len(targets_np)}"
    )

    per_concept = {}
    n_skipped = 0

    unique_concepts = np.unique(exercise_indices_np)
    for concept_idx in unique_concepts:
        assert 0 <= concept_idx < num_nodes, (
            f"BUG: concept_idx={concept_idx} out of range [0, {num_nodes - 1}]"
        )
        mask = exercise_indices_np == concept_idx
        concept_targets = targets_np[mask]
        concept_predictions = predictions_np[mask]

        if len(np.unique(concept_targets)) < 2:
            n_skipped += 1
            continue

        per_concept[int(concept_idx)] = float(
            roc_auc_score(concept_targets, concept_predictions)
        )

    return {
        'per_concept': per_concept,
        'n_concepts_with_auc': len(per_concept),
        'n_concepts_skipped': n_skipped,
    }


if __name__ == '__main__':
    print("=" * 70)
    print("metrics.py -- standalone self-test")
    print("=" * 70)

    # ── bce_loss: known-value check + gradient flow check ────────────────────
    print("\n--- bce_loss ---")
    preds = torch.tensor([0.9, 0.1, 0.8, 0.2], requires_grad=True)
    targs = torch.tensor([1.0, 0.0, 1.0, 0.0])
    loss = bce_loss(preds, targs)
    expected = F.binary_cross_entropy(preds.clamp(1e-7, 1 - 1e-7), targs)
    print(f"  loss = {loss.item():.6f}  (manual F.binary_cross_entropy check: {expected.item():.6f})")
    assert abs(loss.item() - expected.item()) < 1e-9, "BUG: bce_loss doesn't match manual computation"
    loss.backward()
    assert preds.grad is not None and preds.grad.abs().sum().item() > 0, (
        "BUG: bce_loss is not differentiable / gradient did not flow"
    )
    print("  Gradient flowed correctly through bce_loss -- OK")

    # extreme clamp case -- must not produce NaN/Inf
    preds_extreme = torch.tensor([1.0, 0.0], requires_grad=True)
    targs_extreme = torch.tensor([1.0, 0.0])
    loss_extreme = bce_loss(preds_extreme, targs_extreme)
    print(f"  Extreme case (pred=1.0/0.0 exactly): loss = {loss_extreme.item():.6f} "
          f"(must not be NaN/Inf)")
    assert not torch.isnan(loss_extreme) and not torch.isinf(loss_extreme), (
        "BUG: bce_loss produced NaN/Inf on an exact 0.0/1.0 prediction -- clamp not working"
    )
    print("  No NaN/Inf on boundary values -- OK")

    # ── per_student_auc ────────────────────────────────────────────────────
    print("\n--- per_student_auc ---")

    # perfect separation -> AUC should be exactly 1.0
    perfect_preds = [0.1, 0.2, 0.8, 0.9]
    perfect_targs = [0, 0, 1, 1]
    auc_perfect = per_student_auc(perfect_preds, perfect_targs)
    print(f"  Perfect separation: AUC = {auc_perfect} (expected 1.0)")
    assert auc_perfect == 1.0, f"BUG: expected AUC=1.0, got {auc_perfect}"

    # all ties at 0.5 with balanced classes -> AUC should be exactly 0.5
    tied_preds = [0.5, 0.5, 0.5, 0.5]
    tied_targs = [0, 1, 0, 1]
    auc_tied = per_student_auc(tied_preds, tied_targs)
    print(f"  All predictions tied, balanced classes: AUC = {auc_tied} (expected 0.5)")
    assert auc_tied == 0.5, f"BUG: expected AUC=0.5, got {auc_tied}"

    # single-class student -> must return None, not crash or default to 0
    single_class_preds = [0.6, 0.7, 0.55]
    single_class_targs = [1, 1, 1]
    auc_none = per_student_auc(single_class_preds, single_class_targs)
    print(f"  Single-class student (all correct): AUC = {auc_none} (expected None)")
    assert auc_none is None, f"BUG: expected None for single-class student, got {auc_none}"

    # ── macro_auc ─────────────────────────────────────────────────────────
    print("\n--- macro_auc ---")
    per_student_results = [1.0, 0.5, None, 0.8, None]  # 3 valid, 2 skipped
    result = macro_auc(per_student_results)
    print(f"  Input: {per_student_results}")
    print(f"  Result: {result}")
    expected_macro = (1.0 + 0.5 + 0.8) / 3
    assert abs(result['macro_auc'] - expected_macro) < 1e-9, (
        f"BUG: expected macro_auc={expected_macro}, got {result['macro_auc']}"
    )
    assert result['n_valid'] == 3 and result['n_skipped'] == 2 and result['n_total'] == 5, (
        "BUG: valid/skipped/total counts don't match expected"
    )
    print("  Macro average and skip-counting both correct -- OK")

    # ── per_concept_auc ───────────────────────────────────────────────────
    print("\n--- per_concept_auc ---")
    # concept 5: mixed classes, computable
    # concept 7: single-class only, must be skipped
    ex_idx = [5, 5, 5, 5, 7, 7, 7]
    preds_c = [0.1, 0.9, 0.2, 0.8, 0.6, 0.7, 0.55]
    targs_c = [0, 1, 0, 1, 1, 1, 1]
    result_c = per_concept_auc(ex_idx, preds_c, targs_c, num_nodes=835)
    print(f"  per_concept dict: {result_c['per_concept']}")
    print(f"  n_concepts_with_auc: {result_c['n_concepts_with_auc']}  "
          f"n_concepts_skipped: {result_c['n_concepts_skipped']}")
    assert 5 in result_c['per_concept'], "BUG: concept 5 (mixed classes) should have an AUC"
    assert 7 not in result_c['per_concept'], "BUG: concept 7 (single-class) should be skipped"
    assert result_c['per_concept'][5] == 1.0, (
        f"BUG: concept 5 is perfectly separable, expected AUC=1.0, "
        f"got {result_c['per_concept'][5]}"
    )
    assert result_c['n_concepts_with_auc'] == 1 and result_c['n_concepts_skipped'] == 1, (
        "BUG: concept counts don't match expected"
    )
    print("  Per-concept grouping, computation, and skip-handling all correct -- OK")

    print("\nSelf-test complete -- no assertion failures.")