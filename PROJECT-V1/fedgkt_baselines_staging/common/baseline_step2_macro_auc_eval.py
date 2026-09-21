"""
baseline_step2_macro_auc_eval.py

Shared utility, used by every baseline's Colab training driver (DKT
first, then DKVMN/SAKT/AKT/GKT) to override pyKT's own default
evaluation, which reports POOLED interaction-level AUC -- confirmed
directly against pyKT's actual source (pykt/models/evaluate_model.py):
every eval path concatenates predictions and labels across ALL batches
into flat arrays and calls a single roc_auc_score() on the whole thing,
with no per-student grouping anywhere.

FedGKT's own locked evaluation convention (src/utils/metrics.py) is
different, and is what this project's baseline comparison requires:
compute ONE AUC per student (that student's own full sequence of
predictions vs. actual outcomes), THEN average across students.
Students whose outcomes are single-class (all correct, or all incorrect
-- AUC needs both classes present) are EXCLUDED from the average, with
the exclusion COUNTED, never defaulted to some value or silently
dropped without a record.

per_student_auc() and macro_auc() below are DELIBERATELY not just
"similar in spirit" to metrics.py -- the self-test proves them
numerically IDENTICAL to metrics.py's own functions on the same inputs,
by importing metrics.py directly and comparing outputs, not by
independently re-deriving the same description and hoping it matches.
This is the same standard the project already applied to pkg.py vs.
pkg_batched.py: two implementations meant to be equivalent get proven
equivalent via direct numerical comparison, not "looks similar."

SCOPE OF THIS FILE: just the aggregation utility itself
(per_student_auc / macro_auc / evaluate_macro_auc), self-tested against
synthetic-but-realistic prediction arrays. Extracting real per-interaction
(prediction, target, uid) triples out of a live pyKT training run is
tightly coupled to that model's actual forward pass and dataloader --
that hook gets built and verified against REAL model output once we're
building the first baseline's (DKT's) driver script in Step C, not
guessed at here. evaluate_macro_auc() below is written generically
against plain parallel arrays specifically so it can be reused unchanged
across all 5 baselines regardless of how each one's prediction-extraction
differs.

IMPORT NOTE (fixed): the self-test previously did a bare `import metrics`,
which only resolved if a COPY of metrics.py happened to be sitting next to
this file. That was fragile in the worst possible way -- a stale copy that
had drifted from the locked original would still import cleanly and be
used as "ground truth" without anyone noticing. It now imports the real
locked file directly, by anchoring to this script's own location
walking upward until it finds fedgkt/, the same way config.py anchors
to its own path. Nothing is copied and nothing in
fedgkt/ is modified -- it is opened read-only.
"""

import os
import sys

import numpy as np
from sklearn.metrics import roc_auc_score


def _to_numpy(x):
    return np.asarray(x)


def per_student_auc(predictions, targets):
    """
    predictions, targets: 1D arrays for ONE student's full sequence of
    (prediction, actual outcome) pairs -- pooled across ALL that
    student's sequence rows in the pyKT output (a long student may span
    several chunked/windowed rows; this function doesn't care, it just
    wants every real interaction for that one student, pooled together,
    in any order).
    Returns AUC as a float, or None if undefined (single-class outcomes
    -- not an error, just not computable).

    Same logic as FedGKT's own src/utils/metrics.py::per_student_auc --
    proven numerically identical in the self-test below.
    """
    targets_np = _to_numpy(targets)
    predictions_np = _to_numpy(predictions)
    assert len(predictions_np) == len(targets_np), (
        f"BUG: length mismatch, predictions={len(predictions_np)} "
        f"targets={len(targets_np)}"
    )
    if len(np.unique(targets_np)) < 2:
        return None
    return float(roc_auc_score(targets_np, predictions_np))


def macro_auc(per_student_aucs):
    """
    Same logic as FedGKT's own src/utils/metrics.py::macro_auc.
    per_student_aucs: list of per-student AUC values, where some entries
    may be None (excluded, single-class). Returns the same dict shape as
    metrics.py's own macro_auc: macro_auc, n_valid, n_skipped, n_total.
    """
    valid = [a for a in per_student_aucs if a is not None]
    n_total = len(per_student_aucs)
    n_valid = len(valid)
    n_skipped = n_total - n_valid
    assert n_valid > 0, (
        "BUG: every student had an undefined AUC -- cannot compute a "
        "macro average. Check that predictions/targets are being passed "
        "correctly."
    )
    return {
        "macro_auc": float(np.mean(valid)),
        "n_valid": n_valid,
        "n_skipped": n_skipped,
        "n_total": n_total,
    }


def evaluate_macro_auc(predictions, targets, uids):
    """
    THE actual override for pyKT's own pooled roc_auc_score(). Takes flat
    PARALLEL arrays covering an entire evaluation pass -- every real
    (non-padded) interaction from every student being evaluated, in any
    order, with a uid alongside each one identifying which student it
    belongs to. A student with multiple sequence rows (e.g. a long
    student split into several 200-length chunks, or several sliding
    windows) will have interactions from multiple rows mixed together
    here -- that's correct and intended, since the goal is that
    student's FULL macro AUC across their entire real sequence, not a
    separate AUC per row.

    This is the one piece of logic beyond a straight copy of metrics.py:
    metrics.py's own per_student_auc/macro_auc assume the caller has
    already grouped interactions by student. evaluate_macro_auc adds
    that grouping layer, because pyKT will hand back flat per-interaction
    arrays, not pre-grouped-by-student data.

    Returns the same dict shape as macro_auc(), plus 'per_student' (a
    dict of uid -> AUC or None) for inspection/debugging.
    """
    predictions = _to_numpy(predictions)
    targets = _to_numpy(targets)
    uids = _to_numpy(uids)
    assert len(predictions) == len(targets) == len(uids), (
        f"BUG: length mismatch -- predictions={len(predictions)} "
        f"targets={len(targets)} uids={len(uids)}"
    )

    per_student = {}
    for uid in np.unique(uids):
        mask = uids == uid
        per_student[uid] = per_student_auc(predictions[mask], targets[mask])

    result = macro_auc(list(per_student.values()))
    result["per_student"] = per_student
    return result


if __name__ == "__main__":
    print("=" * 70)
    print("baseline_step2_macro_auc_eval.py -- self-test")
    print("Proves per_student_auc()/macro_auc() are numerically IDENTICAL")
    print("to FedGKT's own src/utils/metrics.py (imported directly and")
    print("compared, not just independently re-derived from the same")
    print("description) -- then tests evaluate_macro_auc()'s own new")
    print("logic: grouping flat per-interaction arrays by uid, including")
    print("a student whose interactions are split across multiple rows.")
    print("=" * 70)

    # ---- locate FedGKT's locked metrics.py (read-only ground truth) -------
    # Anchored to THIS file's location, not the current working directory --
    # same principle as config.py. Walks upward until it finds a folder
    # containing fedgkt/src/utils/metrics.py, so it works whether this file
    # sits at the staging root or inside common/.
    #
    # NOTE: only this self-test block needs fedgkt/. When a driver imports
    # evaluate_macro_auc from this file, this block never runs -- which is
    # why it works in Colab, where fedgkt/ is not uploaded.
    _here = os.path.dirname(os.path.abspath(__file__))
    FEDGKT_ROOT = None
    for _ in range(4):
        _candidate = os.path.join(_here, 'fedgkt')
        if os.path.exists(os.path.join(_candidate, 'src', 'utils', 'metrics.py')):
            FEDGKT_ROOT = _candidate
            break
        _here = os.path.dirname(_here)

    assert FEDGKT_ROOT is not None, (
        "Could not find fedgkt/src/utils/metrics.py in any folder above this "
        "file. This self-test compares against FedGKT's locked metrics.py, so "
        "it only runs where fedgkt/ exists (your local machine, not Colab)."
    )
    if FEDGKT_ROOT not in sys.path:
        sys.path.insert(0, FEDGKT_ROOT)

    from src.utils import metrics as fedgkt_metrics

    print(f"\nGround truth loaded from: {fedgkt_metrics.__file__}")

    rng = np.random.default_rng(42)

    # ---- synthetic-but-realistic per-student data, 5 students -------------
    # student A: mixed classes, 20 interactions, normal case
    # student B: mixed classes, 8 interactions, normal case
    # student C: ALL CORRECT (single-class) -- must be excluded, counted
    # student D: ALL INCORRECT (single-class) -- must be excluded, counted
    # student E: mixed classes, but SPLIT ACROSS 3 SEPARATE ROWS (simulating
    #            a long student chunked into 3 pyKT sequence rows) -- tests
    #            that evaluate_macro_auc correctly pools all 3 rows for the
    #            SAME uid before computing that student's AUC, rather than
    #            (incorrectly) computing 3 separate per-row AUCs
    per_student_data = {
        "A": (rng.uniform(0, 1, 20), rng.integers(0, 2, 20)),
        "B": (rng.uniform(0, 1, 8), rng.integers(0, 2, 8)),
        "C": (rng.uniform(0.5, 1, 15), np.ones(15)),
        "D": (rng.uniform(0, 0.5, 10), np.zeros(10)),
    }
    # ensure A and B are genuinely mixed-class (rng could in principle,
    # if unlucky, produce single-class -- guard against that so the
    # self-test itself doesn't silently degrade)
    for uid in ("A", "B"):
        preds, targs = per_student_data[uid]
        if len(np.unique(targs)) < 2:
            targs = targs.copy()
            targs[0], targs[1] = 0, 1
            per_student_data[uid] = (preds, targs)

    # student E: 3 chunks, built so the FULL pooled sequence is mixed-class
    e_chunk1_preds = np.array([0.9, 0.2, 0.7])
    e_chunk1_targs = np.array([1, 0, 1])
    e_chunk2_preds = np.array([0.3, 0.8])
    e_chunk2_targs = np.array([0, 1])
    e_chunk3_preds = np.array([0.6, 0.1, 0.4, 0.95])
    e_chunk3_targs = np.array([1, 0, 0, 1])
    e_full_preds = np.concatenate([e_chunk1_preds, e_chunk2_preds, e_chunk3_preds])
    e_full_targs = np.concatenate([e_chunk1_targs, e_chunk2_targs, e_chunk3_targs])
    per_student_data["E"] = (e_full_preds, e_full_targs)

    # ---- 1. per_student_auc: direct numerical comparison vs metrics.py ----
    print("\n--- per_student_auc: comparing against FedGKT's own metrics.py ---")
    for uid, (preds, targs) in per_student_data.items():
        ours = per_student_auc(preds, targs)
        theirs = fedgkt_metrics.per_student_auc(preds, targs)
        match = (ours is None and theirs is None) or (
            ours is not None and theirs is not None and abs(ours - theirs) < 1e-12
        )
        print(f"  student {uid}: ours={ours}  metrics.py={theirs}  match={match}")
        assert match, f"BUG: per_student_auc diverges from metrics.py for student {uid}"

    assert per_student_data["C"][0] is not None  # sanity: C/D are real single-class cases
    assert per_student_auc(*per_student_data["C"]) is None, "BUG: student C (all-correct) should be None"
    assert per_student_auc(*per_student_data["D"]) is None, "BUG: student D (all-incorrect) should be None"
    print("  Single-class students C (all-correct) and D (all-incorrect) "
          "both correctly return None in both implementations -- OK")

    # ---- 2. macro_auc: direct numerical comparison vs metrics.py ----------
    print("\n--- macro_auc: comparing against FedGKT's own metrics.py ---")
    per_student_aucs = [per_student_auc(*per_student_data[uid]) for uid in ("A", "B", "C", "D", "E")]
    ours_macro = macro_auc(per_student_aucs)
    theirs_macro = fedgkt_metrics.macro_auc(per_student_aucs)
    print(f"  ours:      {ours_macro}")
    print(f"  metrics.py:{theirs_macro}")
    assert abs(ours_macro["macro_auc"] - theirs_macro["macro_auc"]) < 1e-12
    assert ours_macro["n_valid"] == theirs_macro["n_valid"] == 3   # A, B, E valid
    assert ours_macro["n_skipped"] == theirs_macro["n_skipped"] == 2  # C, D skipped
    assert ours_macro["n_total"] == theirs_macro["n_total"] == 5
    print("  macro_auc value, n_valid, n_skipped, n_total all match "
          "metrics.py exactly (3 valid: A,B,E; 2 skipped: C,D) -- OK")

    # ---- 3. evaluate_macro_auc: the actual new logic (flat-array grouping) -
    print("\n--- evaluate_macro_auc: flat per-interaction arrays -> grouped by uid ---")
    all_preds, all_targs, all_uids = [], [], []
    # students A, B, C, D contribute one "row" each (simple case)
    for uid in ("A", "B", "C", "D"):
        preds, targs = per_student_data[uid]
        all_preds.append(preds)
        all_targs.append(targs)
        all_uids.append(np.full(len(preds), uid))
    # student E contributes THREE separate rows -- this is the actual
    # thing being tested: does evaluate_macro_auc correctly pool them
    # into ONE per-student AUC for E, not three separate ones?
    for preds, targs in [
        (e_chunk1_preds, e_chunk1_targs),
        (e_chunk2_preds, e_chunk2_targs),
        (e_chunk3_preds, e_chunk3_targs),
    ]:
        all_preds.append(preds)
        all_targs.append(targs)
        all_uids.append(np.full(len(preds), "E"))

    flat_preds = np.concatenate(all_preds)
    flat_targs = np.concatenate(all_targs)
    flat_uids = np.concatenate(all_uids)
    print(f"  Flat input: {len(flat_preds)} total interactions across "
          f"{len(np.unique(flat_uids))} students (E split across 3 separate rows)")

    result = evaluate_macro_auc(flat_preds, flat_targs, flat_uids)
    print(f"  Result: macro_auc={result['macro_auc']:.6f}  "
          f"n_valid={result['n_valid']}  n_skipped={result['n_skipped']}  "
          f"n_total={result['n_total']}")

    assert abs(result["macro_auc"] - ours_macro["macro_auc"]) < 1e-12, (
        "BUG: evaluate_macro_auc's grouped-from-flat-arrays result doesn't "
        "match the direct per-student computation above"
    )
    assert result["n_valid"] == 3 and result["n_skipped"] == 2 and result["n_total"] == 5
    print("  Matches the direct per-student computation above exactly -- OK")

    # the critical check: E's AUC computed from 3 pooled rows must equal
    # E's AUC computed from its already-concatenated full sequence
    # (per_student_data["E"]) -- proving the 3-row split didn't change
    # the result versus treating it as one continuous sequence
    e_auc_from_grouping = result["per_student"]["E"]
    e_auc_from_full_sequence = per_student_auc(*per_student_data["E"])
    print(f"  Student E AUC (grouped from 3 separate rows): {e_auc_from_grouping:.6f}")
    print(f"  Student E AUC (from one concatenated sequence): {e_auc_from_full_sequence:.6f}")
    assert abs(e_auc_from_grouping - e_auc_from_full_sequence) < 1e-12, (
        "BUG: pooling a multi-row student's interactions before computing "
        "AUC gives a different result than treating them as one continuous "
        "sequence -- the grouping logic is broken"
    )
    print("  Identical -- confirms multi-row students (long sequences split "
          "across several pyKT chunks/windows) are pooled correctly before "
          "AUC is computed, not accidentally scored per-row -- OK")

    # ---- 4. contrast with what pyKT's own pooled AUC would report ---------
    print("\n--- Contrast: pyKT's own pooled (non-macro) AUC on the same data ---")
    pooled_auc = roc_auc_score(flat_targs.astype(float), flat_preds)
    print(f"  pyKT's own pooled interaction-level AUC: {pooled_auc:.6f}")
    print(f"  Our macro-averaged per-student AUC:      {result['macro_auc']:.6f}")
    print("  (These are expected to differ -- that's the entire point of "
          "this override. Pooled AUC also silently includes students C/D's "
          "interactions as if they contributed signal, when in fact a "
          "single-class student contributes no rankable information at "
          "all; macro AUC correctly excludes them instead.)")

    print("\n" + "=" * 70)
    print("Self-test complete -- no assertion failures.")
    print("per_student_auc/macro_auc proven numerically identical to")
    print("FedGKT's own metrics.py. evaluate_macro_auc's flat-array-to-")
    print("per-student grouping verified correct, including the multi-row")
    print("(long student split across several chunks) case.")
    print("=" * 70)