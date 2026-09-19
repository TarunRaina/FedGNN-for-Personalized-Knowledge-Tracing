"""
baseline_step3_fedgkt_reeval.py

Re-runs an already-trained FedGKT checkpoint over a student split and
reports FOUR AUC numbers instead of the one the locked evaluator.py
returns. Nothing is retrained, no weights change, and nothing inside
fedgkt/ is modified -- that folder is opened strictly read-only.

WHY THIS EXISTS
---------------
evaluator.py computes exactly the numbers FedGKT was designed around
(macro AUC, per-concept AUC, BCE) and returns only those aggregates.
Internally it does build the flat per-interaction arrays
(all_predictions / all_targets / all_exercise_indices) -- it just never
returns them, and final_results_batched.json stores only
{user_id, auc, n_interactions} per student. So the raw per-interaction
predictions genuinely cannot be recovered from anything already on
disk; the only way to get them is to replay the sequences. Hence this
script.

The four numbers, and what each is for:

  1. macro_auc_full      -- one AUC per student, averaged. FedGKT's own
                            locked convention. MUST reproduce the
                            existing reported figure exactly; this is
                            the script's own correctness check.
  2. pooled_auc_full     -- every interaction from every student thrown
                            into one pile, one AUC over the pile. This
                            is how pyKT (and most published KT work)
                            reports. Included so a reader can situate
                            our result against the convention the field
                            uses -- NOT as a claim that our number is
                            directly comparable to published Junyi
                            results, because it isn't (different
                            preprocessing, different student subset,
                            different splits; the pyKT paper's own
                            motivating point is that cross-paper KT AUCs
                            diverge widely for exactly these reasons,
                            and Junyi isn't in pyKT's benchmark suite at
                            all).
  3. macro_auc_no_first  -- same as (1), but each student's FIRST
                            interaction is dropped.
  4. pooled_auc_no_first -- same as (2), with first interactions dropped.

WHY DROP THE FIRST INTERACTION (the "N-1" numbers)
--------------------------------------------------
FedGKT predicts on a student's very first interaction, from pure
cold-start PKG state. pyKT structurally cannot: its dataloader slices
smasks[:, 1:], so position 0 of every sequence row is never scored. A
sequence model has no prior context at t=1 and cannot produce a
prediction there at all.

So on the same test split, FedGKT is graded on N predictions per
student while every pyKT baseline is graded on N-1 -- and FedGKT's
extra one is specifically the hardest prediction in the sequence
(zero history). That is a real, if small, handicap against FedGKT, and
it makes a head-to-head table not strictly like-for-like.

DECISION (locked): the N-1 numbers are what goes in the head-to-head
comparison table, so both sides are scored on exactly the same
predictions. The full-N numbers are kept and reported as a clearly
labelled FedGKT-only secondary line, because "FedGKT can make a
cold-start prediction at all, and the sequence baselines structurally
cannot" is a genuine architectural point -- but it belongs in prose,
not smuggled into a number being compared against something measured
differently.

On the real test split this drops 100 of 27,971 predictions (one per
student, 0.36%). Small, but measured rather than assumed.

HOW THIS SCRIPT PROVES ITSELF
-----------------------------
The replay loop below is a deliberate duplicate of evaluator.py's loop
(it has to be -- evaluate() doesn't hand back what we need). Duplicated
logic is exactly the kind of thing that silently drifts, so the script
does not ask to be trusted:

it ALSO calls the locked evaluate() on the same model, same students,
same edge_index, and asserts that

  - its own macro AUC equals evaluate()'s with a difference of EXACTLY
    0.0, not merely within a tolerance, and
  - every single per-student AUC matches, and
  - the total interaction count matches.

Exact equality is the right bar here, not approximate: evaluate() calls
model.eval() and runs under torch.no_grad(), so dropout is off and
nothing samples. The computation is fully deterministic, so two runs of
identical logic must agree bit-for-bit. Anything else is a bug, and the
script stops rather than reporting a number that looks plausible.

This costs a second full pass over the data (a few minutes on CPU for
100 students). That is the price of the guarantee, and it is worth it.

USAGE
-----
    # the real thing -- test split, GPU checkpoint
    python baseline_step3_fedgkt_reeval.py

    # validation split instead
    python baseline_step3_fedgkt_reeval.py --split val

    # sandbox check with an untrained model and 3 known students
    python baseline_step3_fedgkt_reeval.py --self-test

fedgkt/ is expected to be a sibling folder of fedgkt_baselines_staging/.
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import torch
from sklearn.metrics import roc_auc_score


# ── locate the FedGKT project (read-only) ────────────────────────────────
# Anchored to THIS file's location, not the current working directory --
# same principle as config.py. fedgkt/ is a sibling of this folder.
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))   # fedgkt_baselines_staging/
_PARENT_DIR = os.path.dirname(_THIS_DIR)                  # PROJECT-V1/
FEDGKT_ROOT = os.path.join(_PARENT_DIR, 'fedgkt')

assert os.path.isdir(os.path.join(FEDGKT_ROOT, 'src')), (
    f"Could not find FedGKT's src/ folder at {os.path.join(FEDGKT_ROOT, 'src')}.\n"
    f"This script expects fedgkt/ to be a sibling folder of "
    f"fedgkt_baselines_staging/. If your layout differs, edit FEDGKT_ROOT above."
)
if FEDGKT_ROOT not in sys.path:
    sys.path.insert(0, FEDGKT_ROOT)

from src.utils import config as cfg
from src.utils.metrics import per_student_auc, macro_auc
from src.models.fedgkt import FedGKT
from src.data.pkg import PersonalKnowledgeGraph
from src.training.evaluator import evaluate


DEFAULT_CHECKPOINT = os.path.join(
    FEDGKT_ROOT, 'checkpoints_gpu_batched', 'best_model_batched.pt'
)
OUTPUT_DIR = os.path.join(_THIS_DIR, 'fedgkt_reeval')

# the three students used as ground truth throughout this project --
# deliberately mixed lengths (30 / 106 / 9548) so short, medium and very
# long sequences are all exercised
SELF_TEST_STUDENT_IDS = [233536, 45224, 21419]


# ── the replay ───────────────────────────────────────────────────────────
def replay(model, student_ids, edge_index, verbose=False):
    """
    Deliberate duplicate of evaluator.py's evaluate() loop -- SAME
    PersonalKnowledgeGraph contract, SAME order of operations:

        refresh_time_decay(t) -> forward pass -> update(ex, c, t)

    The only difference is what comes back out: this keeps every
    per-interaction prediction, along with which student it belongs to
    and its position within that student's own sequence (so the first
    interaction can be identified and dropped later).

    Returns a dict of flat parallel numpy arrays, one entry per
    interaction, plus per-student bookkeeping.
    """
    model.eval()

    all_predictions = []
    all_targets = []
    all_exercise_indices = []
    all_uids = []
    all_positions = []          # 0-based position within that student's sequence

    per_student_aucs_full = []  # mirrors evaluate()'s own list, for the check
    per_student_n = {}

    start_time = time.time()

    with torch.no_grad():
        for i, user_id in enumerate(student_ids):
            pkg_path = os.path.join(cfg.PKG_DIR, f'pkg_{user_id}.pt')
            assert os.path.exists(pkg_path), (
                f"BUG: PKG file not found for user_id={user_id} at {pkg_path}."
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

                all_predictions.append(pred_val)
                all_targets.append(c)
                all_exercise_indices.append(ex)
                all_uids.append(int(user_id))
                all_positions.append(step)

            per_student_aucs_full.append(per_student_auc(student_preds, student_targets))
            per_student_n[int(user_id)] = n

            if verbose:
                print(f"  [{i + 1}/{len(student_ids)}] user_id={user_id}  "
                      f"n_interactions={n}")

    elapsed = time.time() - start_time

    return {
        'predictions': np.asarray(all_predictions, dtype=np.float64),
        'targets': np.asarray(all_targets, dtype=np.float64),
        'exercise_indices': np.asarray(all_exercise_indices, dtype=np.int64),
        'uids': np.asarray(all_uids, dtype=np.int64),
        'positions': np.asarray(all_positions, dtype=np.int64),
        'per_student_aucs_full': per_student_aucs_full,
        'per_student_n': per_student_n,
        'elapsed_seconds': elapsed,
    }


# ── the four metrics ─────────────────────────────────────────────────────
def compute_metrics(replay_result, student_ids):
    """
    Turns the flat replay arrays into the four reported numbers.

    'full' = every interaction. 'no_first' = each student's position-0
    interaction dropped, so the prediction set matches what a pyKT
    baseline is scored on.
    """
    preds = replay_result['predictions']
    targs = replay_result['targets']
    uids = replay_result['uids']
    positions = replay_result['positions']

    keep_mask = positions != 0          # drop each student's first interaction
    n_dropped = int((~keep_mask).sum())

    assert n_dropped == len(student_ids), (
        f"BUG: expected to drop exactly one first-interaction per student "
        f"({len(student_ids)}), dropped {n_dropped}. Position bookkeeping is wrong."
    )

    def macro_over(mask):
        """One AUC per student over the masked interactions, then averaged."""
        aucs = []
        for uid in student_ids:
            sel = (uids == int(uid)) & mask
            if sel.sum() == 0:
                aucs.append(None)       # nothing left to score for this student
                continue
            aucs.append(per_student_auc(preds[sel], targs[sel]))
        return macro_auc(aucs), aucs

    def pooled_over(mask):
        """One AUC over every masked interaction, pooled across students."""
        t = targs[mask]
        p = preds[mask]
        assert len(np.unique(t)) >= 2, (
            "BUG: pooled targets are single-class -- AUC undefined."
        )
        return float(roc_auc_score(t, p))

    all_mask = np.ones(len(preds), dtype=bool)

    macro_full, per_student_full = macro_over(all_mask)
    macro_no_first, per_student_no_first = macro_over(keep_mask)

    return {
        'macro_auc_full': macro_full['macro_auc'],
        'macro_full_n_valid': macro_full['n_valid'],
        'macro_full_n_skipped': macro_full['n_skipped'],

        'pooled_auc_full': pooled_over(all_mask),

        'macro_auc_no_first': macro_no_first['macro_auc'],
        'macro_no_first_n_valid': macro_no_first['n_valid'],
        'macro_no_first_n_skipped': macro_no_first['n_skipped'],

        'pooled_auc_no_first': pooled_over(keep_mask),

        'n_students': len(student_ids),
        'n_interactions_full': int(len(preds)),
        'n_interactions_no_first': int(keep_mask.sum()),
        'n_dropped_first_interactions': n_dropped,

        '_per_student_full': per_student_full,
        '_per_student_no_first': per_student_no_first,
    }


# ── the self-check against the locked evaluator ──────────────────────────
def verify_against_locked_evaluator(model, student_ids, edge_index, replay_result):
    """
    Runs the locked evaluate() on the same model / students / edge_index
    and requires EXACT agreement. Returns evaluate()'s own result dict so
    its numbers can be recorded alongside.

    Exact, not approximate: evaluate() is model.eval() + torch.no_grad(),
    so there is no dropout and no sampling anywhere. Identical logic on
    identical inputs must produce bit-identical output. A tolerance here
    would quietly hide precisely the kind of drift this check exists to
    catch.
    """
    print("\n--- Verifying replay against the locked evaluator.py ---")
    print("    (second full pass over the same data -- this is the guarantee, "
          "not wasted work)")

    locked = evaluate(model, student_ids, edge_index=edge_index, verbose=False)

    # 1. total interaction count
    assert locked['total_interactions'] == len(replay_result['predictions']), (
        f"BUG: interaction count mismatch -- evaluator.py counted "
        f"{locked['total_interactions']}, replay counted "
        f"{len(replay_result['predictions'])}."
    )
    print(f"    Interaction count matches: {locked['total_interactions']:,}")

    # 2. every per-student AUC, individually
    locked_aucs = [d['auc'] for d in locked['per_student_details']]
    replay_aucs = replay_result['per_student_aucs_full']
    assert len(locked_aucs) == len(replay_aucs), (
        f"BUG: student count mismatch -- {len(locked_aucs)} vs {len(replay_aucs)}"
    )
    for uid, a, b in zip(student_ids, locked_aucs, replay_aucs):
        if a is None or b is None:
            assert a is None and b is None, (
                f"BUG: student {uid} -- evaluator.py gave {a}, replay gave {b}. "
                f"One treated the student as single-class and the other didn't."
            )
            continue
        assert a == b, (
            f"BUG: student {uid} AUC differs -- evaluator.py {a!r}, "
            f"replay {b!r} (difference {abs(a - b):.3e}). Expected EXACT equality."
        )
    print(f"    All {len(locked_aucs)} per-student AUCs match exactly")

    # 3. the macro AUC itself
    diff = abs(locked['macro_auc'] - replay_result['_macro_auc_full'])
    assert diff == 0.0, (
        f"BUG: macro AUC differs -- evaluator.py "
        f"{locked['macro_auc']!r}, replay {replay_result['_macro_auc_full']!r} "
        f"(difference {diff:.3e}). Expected EXACTLY 0.0."
    )
    print(f"    Macro AUC matches exactly: {locked['macro_auc']:.6f} "
          f"(difference 0.0)")
    print("    Replay is provably doing the same thing as the locked code.")

    return locked


# ── checkpoint loading ───────────────────────────────────────────────────
def load_checkpoint(model, checkpoint_path):
    """
    best_model_batched.pt is a raw state_dict (saved via
    torch.save(model.state_dict(), path)), not wrapped in a metadata
    dict. Loading is strict, and the key sets are compared explicitly
    first so a partial or mismatched load fails loudly instead of
    silently leaving some layers at their random initialisation.
    """
    assert os.path.exists(checkpoint_path), (
        f"Checkpoint not found at {checkpoint_path}"
    )
    state = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

    assert isinstance(state, dict), (
        f"Expected a state_dict, got {type(state)}."
    )
    assert not any(k in state for k in ('model_state_dict', 'state_dict')), (
        "This checkpoint looks like a wrapped dict rather than a raw "
        "state_dict. Unwrap it before loading."
    )

    expected = set(model.state_dict().keys())
    found = set(state.keys())
    assert expected == found, (
        f"Checkpoint keys do not match the model.\n"
        f"  missing from checkpoint: {sorted(expected - found)}\n"
        f"  unexpected in checkpoint: {sorted(found - expected)}"
    )

    model.load_state_dict(state, strict=True)
    print(f"Checkpoint loaded: {checkpoint_path}")
    print(f"  {len(found)} parameter tensors, all keys matched exactly")
    return model


# ── main ─────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Re-evaluate a trained FedGKT checkpoint and report "
                    "macro/pooled AUC, both with and without each student's "
                    "first interaction."
    )
    parser.add_argument('--split', default='test', choices=['test', 'val', 'train'],
                        help="Which split from student_splits.json (default: test).")
    parser.add_argument('--checkpoint', default=DEFAULT_CHECKPOINT,
                        help="Path to the trained checkpoint.")
    parser.add_argument('--self-test', action='store_true',
                        help="Sandbox check: untrained model, 3 known students.")
    parser.add_argument('--out', default=None,
                        help="Where to write the JSON result "
                             "(default: fedgkt_reeval/<split>_reeval.json).")
    args = parser.parse_args()

    print("=" * 70)
    print("baseline_step3_fedgkt_reeval.py")
    print("=" * 70)
    print(f"FedGKT root (read-only): {FEDGKT_ROOT}")

    edge_index = torch.load(cfg.EDGE_INDEX_PATH, weights_only=False)
    print(f"edge_index: {tuple(edge_index.shape)}")

    model = FedGKT()

    if args.self_test:
        print("\nSELF-TEST MODE: untrained model, 3 known students "
              "(30 / 106 / 9548 interactions).")
        print("An untrained model scores near 0.5 -- that is expected and not "
              "what is being tested here. What IS being tested is that the "
              "replay reproduces the locked evaluator.py exactly.")
        torch.manual_seed(cfg.RANDOM_SEED)
        model = FedGKT()
        student_ids = SELF_TEST_STUDENT_IDS
        label = 'selftest'
    else:
        model = load_checkpoint(model, args.checkpoint)
        assert os.path.exists(cfg.SPLITS_PATH), (
            f"student_splits.json not found at {cfg.SPLITS_PATH}"
        )
        splits = json.load(open(cfg.SPLITS_PATH))
        student_ids = splits[args.split]
        label = args.split
        print(f"\nSplit: {args.split}  ({len(student_ids)} students)")

    # ── pass 1: the replay ────────────────────────────────────────────
    print(f"\n--- Replaying {len(student_ids)} students ---")
    replay_result = replay(model, student_ids, edge_index, verbose=args.self_test)
    n_int = len(replay_result['predictions'])
    print(f"    {n_int:,} interactions in {replay_result['elapsed_seconds']:.1f}s "
          f"({n_int / max(replay_result['elapsed_seconds'], 1e-9):.0f}/sec)")

    metrics = compute_metrics(replay_result, student_ids)
    replay_result['_macro_auc_full'] = metrics['macro_auc_full']

    # ── pass 2: the guarantee ─────────────────────────────────────────
    locked = verify_against_locked_evaluator(
        model, student_ids, edge_index, replay_result
    )

    # ── report ────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print(f"RESULTS -- {label} split, {metrics['n_students']} students")
    print("=" * 70)
    print(f"\nFull sequences ({metrics['n_interactions_full']:,} predictions -- "
          f"FedGKT's native protocol):")
    print(f"  macro  AUC : {metrics['macro_auc_full']:.4f}   "
          f"({metrics['macro_full_n_valid']} students, "
          f"{metrics['macro_full_n_skipped']} skipped as single-class)")
    print(f"  pooled AUC : {metrics['pooled_auc_full']:.4f}")

    print(f"\nFirst interaction dropped "
          f"({metrics['n_interactions_no_first']:,} predictions -- "
          f"matches what pyKT baselines are scored on):")
    print(f"  macro  AUC : {metrics['macro_auc_no_first']:.4f}   "
          f"({metrics['macro_no_first_n_valid']} students, "
          f"{metrics['macro_no_first_n_skipped']} skipped as single-class)")
    print(f"  pooled AUC : {metrics['pooled_auc_no_first']:.4f}")

    delta_macro = metrics['macro_auc_no_first'] - metrics['macro_auc_full']
    pct = 100.0 * metrics['n_dropped_first_interactions'] / metrics['n_interactions_full']
    print(f"\nDropped {metrics['n_dropped_first_interactions']} predictions "
          f"({pct:.2f}% of the total), one per student.")
    print(f"Effect on macro AUC: {delta_macro:+.4f}")

    print("\nFor the head-to-head table, use the 'first interaction dropped' "
          "macro AUC.")
    print("Report the full-sequence macro AUC as a FedGKT-only secondary line.")

    # ── write JSON ────────────────────────────────────────────────────
    out_path = args.out or os.path.join(OUTPUT_DIR, f'{label}_reeval.json')
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    payload = {k: v for k, v in metrics.items() if not k.startswith('_')}
    payload['split'] = label
    payload['checkpoint'] = None if args.self_test else os.path.abspath(args.checkpoint)
    payload['verified_against_locked_evaluator'] = True
    payload['locked_evaluator_macro_auc'] = locked['macro_auc']
    payload['locked_evaluator_overall_bce'] = locked['overall_bce']
    payload['per_student'] = [
        {
            'user_id': int(uid),
            'n_interactions': replay_result['per_student_n'][int(uid)],
            'auc_full': metrics['_per_student_full'][i],
            'auc_no_first': metrics['_per_student_no_first'][i],
        }
        for i, uid in enumerate(student_ids)
    ]

    with open(out_path, 'w') as f:
        json.dump(payload, f, indent=2)
    print(f"\nWritten: {out_path}")

    print("\n" + "=" * 70)
    print("Done -- replay verified against the locked evaluator, "
          "no assertion failures.")
    print("=" * 70)


if __name__ == '__main__':
    main()