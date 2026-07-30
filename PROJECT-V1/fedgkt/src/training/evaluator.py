"""
src/training/evaluator.py

Runs a FedGKT model over a set of students WITHOUT updating weights --
replays each student's full interaction sequence (same PersonalKnowledgeGraph
update contract used in training: refresh_time_decay -> forward -> update),
collects predictions vs actual outcomes, and aggregates them into the
project's locked evaluation metrics:

  - macro AUC-ROC (averaged across students -- the primary metric)
  - per-concept AUC (pooled across all students passed in)
  - overall BCE (secondary, informational)

Used for both validation-during-training and final test-set evaluation.
Never calls .backward() or touches an optimizer -- purely inference.
"""

import os
import sys
import time

import torch

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))       # src/training
_SRC_DIR = os.path.dirname(_THIS_DIR)                          # src
_PROJECT_ROOT = os.path.dirname(_SRC_DIR)                      # fedgkt/
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.utils import config as cfg
from src.utils.metrics import bce_loss, per_student_auc, macro_auc, per_concept_auc
from src.data.pkg import PersonalKnowledgeGraph


def evaluate(model, student_ids, edge_index=None, verbose=False):
    """
    model:        a FedGKT instance (or anything with the same
                  forward(x, edge_index, exercise_idx) signature)
    student_ids:  list of user_ids to evaluate over (e.g. the val or test
                  split from student_splits.json)
    edge_index:   [2, NUM_EDGES] tensor. If None, loaded from
                  cfg.EDGE_INDEX_PATH (loading once per call, not once per
                  student, since it's identical for everyone).
    verbose:      if True, prints per-student progress.

    Returns a dict:
        macro_auc, macro_n_valid, macro_n_skipped, macro_n_total,
        per_concept_auc, n_concepts_with_auc, n_concepts_skipped,
        overall_bce, per_student_details (list of {user_id, auc, n_interactions}),
        n_students_evaluated, total_interactions, elapsed_seconds
    """
    assert len(student_ids) > 0, "BUG: student_ids is empty -- nothing to evaluate"

    if edge_index is None:
        assert os.path.exists(cfg.EDGE_INDEX_PATH), (
            f"BUG: edge_index.pt not found at {cfg.EDGE_INDEX_PATH}"
        )
        edge_index = torch.load(cfg.EDGE_INDEX_PATH, weights_only=False)

    model.eval()

    per_student_aucs = []
    per_student_details = []

    all_exercise_indices = []
    all_predictions = []
    all_targets = []

    start_time = time.time()

    with torch.no_grad():
        for i, user_id in enumerate(student_ids):
            pkg_path = os.path.join(cfg.PKG_DIR, f'pkg_{user_id}.pt')
            assert os.path.exists(pkg_path), (
                f"BUG: PKG file not found for user_id={user_id} at {pkg_path}. "
                f"Check that this student_id actually appears in the Step 5 "
                f"working subset (student_splits.json)."
            )
            data = torch.load(pkg_path, weights_only=False)

            exercise_idx_seq = data['exercise_idx']
            correct_seq = data['correct']
            time_done_seq = data['time_done']

            n = exercise_idx_seq.shape[0]
            assert n > 0, f"BUG: student {user_id} has zero interactions"

            pkg = PersonalKnowledgeGraph()
            student_preds = []
            student_targets = []

            for step in range(n):
                ex = int(exercise_idx_seq[step].item())
                c = float(correct_seq[step].item())
                t = int(time_done_seq[step].item())

                pkg.refresh_time_decay(t)
                pred = model(pkg.get_x(), edge_index, ex)
                pred_val = float(pred.item())

                pkg.update(ex, c, t)

                student_preds.append(pred_val)
                student_targets.append(c)

                all_exercise_indices.append(ex)
                all_predictions.append(pred_val)
                all_targets.append(c)

            student_auc = per_student_auc(student_preds, student_targets)
            per_student_aucs.append(student_auc)
            per_student_details.append({
                'user_id': int(user_id),
                'auc': student_auc,
                'n_interactions': n,
            })

            if verbose:
                auc_str = f"{student_auc:.4f}" if student_auc is not None else "N/A (single-class)"
                print(f"  [{i + 1}/{len(student_ids)}] user_id={user_id}  "
                      f"n_interactions={n}  auc={auc_str}")

    elapsed = time.time() - start_time

    macro_result = macro_auc(per_student_aucs)
    concept_result = per_concept_auc(
        all_exercise_indices, all_predictions, all_targets, cfg.NUM_NODES
    )

    preds_tensor = torch.tensor(all_predictions, dtype=torch.float32)
    targets_tensor = torch.tensor(all_targets, dtype=torch.float32)
    overall_bce = float(bce_loss(preds_tensor, targets_tensor).item())

    return {
        'macro_auc': macro_result['macro_auc'],
        'macro_n_valid': macro_result['n_valid'],
        'macro_n_skipped': macro_result['n_skipped'],
        'macro_n_total': macro_result['n_total'],
        'per_concept_auc': concept_result['per_concept'],
        'n_concepts_with_auc': concept_result['n_concepts_with_auc'],
        'n_concepts_skipped': concept_result['n_concepts_skipped'],
        'overall_bce': overall_bce,
        'per_student_details': per_student_details,
        'n_students_evaluated': len(student_ids),
        'total_interactions': len(all_predictions),
        'elapsed_seconds': elapsed,
    }


if __name__ == '__main__':
    print("=" * 70)
    print("evaluator.py -- standalone self-test (REAL student data, untrained model)")
    print("=" * 70)

    from src.models.fedgkt import FedGKT

    # use the same 3 real students already manually verified in preprocessing
    # Step 6 -- small (30), medium (106), large (9548) interactions
    test_student_ids = [233536, 45224, 21419]

    missing = [
        uid for uid in test_student_ids
        if not os.path.exists(os.path.join(cfg.PKG_DIR, f'pkg_{uid}.pt'))
    ]
    assert not missing, (
        f"Expected PKG files for {test_student_ids} not found -- missing: {missing}. "
        f"Check cfg.PKG_DIR = {cfg.PKG_DIR}"
    )

    torch.manual_seed(cfg.RANDOM_SEED)
    model = FedGKT()

    print(f"\nEvaluating UNTRAINED model on {len(test_student_ids)} real students: "
          f"{test_student_ids}")
    print("(Untrained/random-weight model -- AUC near 0.5 is EXPECTED here. This "
          "test verifies the evaluation PIPELINE works correctly, not that the "
          "model has learned anything yet -- that comes after real training.)\n")

    result = evaluate(model, test_student_ids, verbose=True)

    print(f"\n--- Aggregated results ---")
    print(f"Macro AUC:              {result['macro_auc']:.4f}")
    print(f"  Valid students:       {result['macro_n_valid']} / {result['macro_n_total']} "
          f"(skipped: {result['macro_n_skipped']})")
    print(f"Overall BCE:            {result['overall_bce']:.4f}")
    print(f"Concepts with AUC:      {result['n_concepts_with_auc']}")
    print(f"Concepts skipped:       {result['n_concepts_skipped']}")
    print(f"Total interactions:     {result['total_interactions']:,}")
    print(f"Elapsed time:           {result['elapsed_seconds']:.2f}s")
    print(f"Interactions/sec:       {result['total_interactions'] / result['elapsed_seconds']:.1f}")

    # ── sanity checks ────────────────────────────────────────────────────────
    expected_total_interactions = 30 + 106 + 9548
    assert result['total_interactions'] == expected_total_interactions, (
        f"BUG: expected {expected_total_interactions} total interactions "
        f"(30+106+9548), got {result['total_interactions']}"
    )
    print(f"\nTotal interaction count matches expected "
          f"(30+106+9548={expected_total_interactions}) -- OK")

    assert 0.0 <= result['macro_auc'] <= 1.0, "BUG: macro_auc outside [0,1]"
    assert result['overall_bce'] >= 0.0, "BUG: overall_bce is negative"
    assert result['macro_n_total'] == 3, "BUG: expected exactly 3 students in macro result"

    for detail in result['per_student_details']:
        print(f"  user_id={detail['user_id']}  n_interactions={detail['n_interactions']}  "
              f"auc={detail['auc']}")

    print("\nSelf-test complete -- no assertion failures.")
    print("\nNOTE: this used only 3 students for a fast smoke test. The full")
    print("training loop will call evaluate() on all 100 val / 100 test students")
    print("-- the interactions/sec figure above gives a rough sense of runtime")
    print("for that (though a trained model in training mode with dropout may")
    print("differ slightly in speed from this eval-mode, untrained run).")