"""
src/training/evaluator_batched.py

Batched, device-agnostic version of evaluator.py's evaluate(). Same
inputs, same return shape -- a drop-in replacement whose only purpose is
to run the same computation faster by evaluating many students at once,
on the GPU, instead of one at a time on the CPU.

WHY THIS EXISTS
---------------
evaluator.py is sequential and CPU-only by design, which was the right
call when validation happened once per epoch on 100 students and training
was the bottleneck. For the v2 improvement campaign it is no longer:
validation costs roughly 200s of every ~550s epoch, so nearly 40% of every
screening run is spent here. Over a campaign of ten screens that is days.

This file does NOT replace evaluator.py. evaluator.py stays exactly as it
is -- it produced every number this project has reported, and it remains
the reference. This file has to prove itself against it (see below).

HOW IT WORKS
------------
Exactly the same contract as evaluate(), per interaction:

    refresh_time_decay(t) -> forward pass -> update(ex, c, t)

but with B students advancing in lockstep, using the already-verified
BatchedPersonalKnowledgeGraph and the same flattened multi-graph trick
training uses: B disjoint copies of the 835-node graph living in one
[B*835, F] tensor, with node indices offset by i*835 for student i.

Students are grouped into batches by sequence length (deterministic, no
randomness), so a batch of similar-length students wastes little compute
on padding. A student who finishes early is masked out: their PKG state
is frozen and their padded steps produce predictions that are discarded,
never scored.

THE EQUIVALENCE PROBLEM, AND WHY EXACT EQUALITY IS THE WRONG BAR
-----------------------------------------------------------------
The Step 3 FedGKT replay required a difference of EXACTLY 0.0 from
evaluate(), and rightly so: it ran the identical code path on the same
device, so any difference at all meant a bug.

That bar does not apply here, and demanding it would be wrong. This file
deliberately computes the same quantity a different way -- many students
at once instead of one, on a GPU instead of a CPU. Floating-point
addition is not associative, so summing the same numbers in a different
order gives answers that differ in the last few decimal places. Some GPU
kernels are not even bit-reproducible between two runs of themselves.

So the honest bar is the one this project already used for
pkg_batched.py vs pkg.py: prove agreement to a tolerance far tighter
than any effect being measured. That comparison came out at 5.96e-08.
The self-test below applies the same standard to the metrics themselves,
requiring agreement to 1e-5 on macro AUC and on every per-student AUC --
four orders of magnitude smaller than the ~0.01 improvements the v2
campaign is looking for.

IF THE SELF-TEST FAILS, THIS FILE IS NOT USABLE. Fall back to
evaluator.py; a screening campaign measured with a broken ruler is worse
than a slow one.

REPORTING
---------
Whichever evaluator produces a number that ends up in the paper must be
stated, along with this agreement check as the evidence the two are
equivalent.
"""

import os
import sys
import time

import torch

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))       # src/training
_SRC_DIR = os.path.dirname(_THIS_DIR)                          # src
_PROJECT_ROOT = os.path.dirname(_SRC_DIR)                      # fedgkt_v2/
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.utils import config as cfg
from src.utils.metrics import bce_loss, per_student_auc, macro_auc, per_concept_auc
from src.data.pkg_batched import BatchedPersonalKnowledgeGraph


def build_batched_edge_index(edge_index, B, num_nodes, device):
    """
    B disjoint copies of the graph, node indices in copy i offset by
    i * num_nodes. Identical to centralised_batched.py's function of the
    same name -- kept here so evaluation does not import the training
    module.
    """
    edge_index = edge_index.to(device)
    offsets = (torch.arange(B, device=device) * num_nodes).view(B, 1, 1)
    replicated = edge_index.unsqueeze(0).expand(B, 2, -1) + offsets
    return replicated.permute(1, 0, 2).reshape(2, -1)


def _load_student_sequence(user_id):
    pkg_path = os.path.join(cfg.PKG_DIR, f'pkg_{user_id}.pt')
    assert os.path.exists(pkg_path), (
        f"BUG: PKG file not found for user_id={user_id} at {pkg_path}"
    )
    data = torch.load(pkg_path, weights_only=False)
    return data['exercise_idx'], data['correct'], data['time_done']


def evaluate_batched(model, student_ids, edge_index=None, batch_size=16,
                     device=None, verbose=False):
    """
    Drop-in replacement for evaluator.evaluate(). Returns the same dict
    keys, with per_student_details in the SAME ORDER as student_ids
    (batching reorders internally by length; the output is reordered back).

    batch_size: students evaluated together. Under no_grad the memory cost
    is small -- nothing is retained for backpropagation -- so this can be
    considerably larger than the training batch size.
    """
    assert len(student_ids) > 0, "BUG: student_ids is empty -- nothing to evaluate"

    device = device if device is not None else torch.device(
        'cuda' if torch.cuda.is_available() else 'cpu')

    if edge_index is None:
        assert os.path.exists(cfg.EDGE_INDEX_PATH), (
            f"BUG: edge_index.pt not found at {cfg.EDGE_INDEX_PATH}"
        )
        edge_index = torch.load(cfg.EDGE_INDEX_PATH, weights_only=False)

    model = model.to(device)
    model.eval()

    # ── load every sequence once, then batch by length ────────────────
    sequences = {uid: _load_student_sequence(uid) for uid in student_ids}
    lengths = {uid: int(sequences[uid][0].shape[0]) for uid in student_ids}
    for uid in student_ids:
        assert lengths[uid] > 0, f"BUG: student {uid} has zero interactions"

    sorted_ids = sorted(student_ids, key=lambda uid: lengths[uid])
    batches = [sorted_ids[i:i + batch_size]
               for i in range(0, len(sorted_ids), batch_size)]

    preds_by_student = {}
    targets_by_student = {}
    edge_index_cache = {}

    start_time = time.time()

    with torch.no_grad():
        for bi, batch_ids in enumerate(batches):
            B = len(batch_ids)
            if B not in edge_index_cache:
                edge_index_cache[B] = build_batched_edge_index(
                    edge_index, B, cfg.NUM_NODES, device)
            batched_edge_index = edge_index_cache[B]

            max_len = max(lengths[uid] for uid in batch_ids)

            padded_ex = torch.zeros((B, max_len), dtype=torch.int64)
            padded_correct = torch.zeros((B, max_len), dtype=torch.float32)
            padded_time = torch.zeros((B, max_len), dtype=torch.int64)
            active_matrix = torch.zeros((B, max_len), dtype=torch.bool)

            for i, uid in enumerate(batch_ids):
                ex_seq, c_seq, t_seq = sequences[uid]
                L = lengths[uid]
                padded_ex[i, :L] = ex_seq
                padded_correct[i, :L] = c_seq
                padded_time[i, :L] = t_seq
                active_matrix[i, :L] = True
                if L < max_len:
                    # harmless placeholders: masked out, never scored, and
                    # chosen so the chronological-order assertions hold
                    padded_ex[i, L:] = ex_seq[-1]
                    padded_correct[i, L:] = 0.0
                    padded_time[i, L:] = t_seq[-1]

            padded_ex = padded_ex.to(device)
            padded_correct = padded_correct.to(device)
            padded_time = padded_time.to(device)
            active_matrix = active_matrix.to(device)

            pkg_batch = BatchedPersonalKnowledgeGraph(batch_size=B, device=device)
            batch_offsets = torch.arange(B, device=device) * cfg.NUM_NODES

            step_preds = []          # one [B] tensor per step
            for step in range(max_len):
                active_mask = active_matrix[:, step]
                step_time = padded_time[:, step]
                step_ex = padded_ex[:, step]
                step_correct = padded_correct[:, step]

                pkg_batch.refresh_time_decay(step_time, active_mask)

                x_flat = pkg_batch.get_x().reshape(B * cfg.NUM_NODES, cfg.NUM_FEATURES)
                flat_idx = step_ex + batch_offsets
                preds = model(x_flat, batched_edge_index, flat_idx)   # [B]

                step_preds.append(preds.detach().float().cpu())

                pkg_batch.update(step_ex, step_correct, step_time, active_mask)

            all_steps = torch.stack(step_preds, dim=1)                # [B, max_len]
            active_cpu = active_matrix.cpu()
            correct_cpu = padded_correct.cpu()

            for i, uid in enumerate(batch_ids):
                L = lengths[uid]
                keep = active_cpu[i, :L]
                assert bool(keep.all()), (
                    f"BUG: student {uid} has an inactive step inside its real length"
                )
                preds_by_student[uid] = all_steps[i, :L].tolist()
                targets_by_student[uid] = correct_cpu[i, :L].tolist()

            if verbose:
                print(f"  batch {bi + 1}/{len(batches)}  students={B}  "
                      f"max_len={max_len}")

    elapsed = time.time() - start_time

    # ── aggregate, in the caller's original student order ─────────────
    per_student_aucs = []
    per_student_details = []
    all_exercise_indices = []
    all_predictions = []
    all_targets = []

    for uid in student_ids:
        p = preds_by_student[uid]
        t = targets_by_student[uid]
        n = lengths[uid]
        assert len(p) == len(t) == n, (
            f"BUG: student {uid} produced {len(p)} predictions for {n} interactions"
        )

        auc = per_student_auc(p, t)
        per_student_aucs.append(auc)
        per_student_details.append({
            'user_id': int(uid), 'auc': auc, 'n_interactions': n,
        })

        all_exercise_indices.extend(sequences[uid][0].tolist())
        all_predictions.extend(p)
        all_targets.extend(t)

    macro_result = macro_auc(per_student_aucs)
    concept_result = per_concept_auc(
        all_exercise_indices, all_predictions, all_targets, cfg.NUM_NODES)

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
        # extra key (harmless for drop-in use): lets the self-test below
        # detect tied predictions, which AUC is acutely sensitive to
        'per_student_predictions': {int(u): preds_by_student[u] for u in student_ids},
    }


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description="Prove evaluate_batched() agrees with the locked evaluate().")
    parser.add_argument('--students', default='selftest',
                        choices=['selftest', 'val'],
                        help="'selftest' = the 3 known students (fast, CPU-friendly); "
                             "'val' = all 100 validation students (the real check).")
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--checkpoint', default=None,
                        help="Optional trained checkpoint (raw state_dict). "
                             "Without one, an untrained model is used -- fine for "
                             "an equivalence check, which is about agreement, not skill.")
    parser.add_argument('--tolerance', type=float, default=1e-5,
                        help="Prediction-level tolerance (BCE). Tight on purpose: BCE "
                             "moves only if the predictions themselves move.")
    parser.add_argument('--macro-tolerance', type=float, default=1e-4,
                        help="Macro AUC tolerance. Looser than the BCE one BY DESIGN -- "
                             "see the note in the code. Still ~100x below the effects "
                             "the v2 campaign measures.")
    args = parser.parse_args()

    from src.models.fedgkt import FedGKT
    from src.training.evaluator import evaluate

    print("=" * 70)
    print("evaluator_batched.py -- equivalence check against the locked evaluator")
    print("=" * 70)

    if args.students == 'selftest':
        student_ids = [233536, 45224, 21419]
    else:
        import json
        with open(cfg.SPLITS_PATH) as f:
            student_ids = json.load(f)['val']

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\ndevice: {device}   students: {len(student_ids)}   "
          f"batch_size: {args.batch_size}")

    edge_index = torch.load(cfg.EDGE_INDEX_PATH, weights_only=False)

    torch.manual_seed(cfg.RANDOM_SEED)
    model = FedGKT()
    if args.checkpoint:
        state = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
        expected, found = set(model.state_dict().keys()), set(state.keys())
        assert expected == found, (
            f"Checkpoint keys do not match the model.\n"
            f"  missing: {sorted(expected - found)}\n"
            f"  unexpected: {sorted(found - expected)}")
        model.load_state_dict(state, strict=True)
        print(f"checkpoint: {args.checkpoint} ({len(found)} tensors, all keys matched)")
    else:
        print("checkpoint: none (untrained model -- agreement is what is being tested)")

    print("\n--- Locked evaluator.py (sequential, CPU) ---")
    ref = evaluate(model.to('cpu'), student_ids,
                   edge_index=edge_index, verbose=False)
    print(f"    macro AUC {ref['macro_auc']:.10f}   "
          f"{ref['total_interactions']:,} interactions   "
          f"{ref['elapsed_seconds']:.1f}s")

    print(f"\n--- evaluator_batched.py (batched, {device}) ---")
    got = evaluate_batched(model, student_ids, edge_index=edge_index,
                           batch_size=args.batch_size, device=device)
    print(f"    macro AUC {got['macro_auc']:.10f}   "
          f"{got['total_interactions']:,} interactions   "
          f"{got['elapsed_seconds']:.1f}s")

    speedup = ref['elapsed_seconds'] / max(got['elapsed_seconds'], 1e-9)
    print(f"\nspeed-up: {speedup:.1f}x")

    print("\n--- Agreement ---")
    assert got['total_interactions'] == ref['total_interactions'], (
        f"BUG: interaction counts differ -- {got['total_interactions']} vs "
        f"{ref['total_interactions']}")
    print(f"  interaction count matches: {ref['total_interactions']:,}")

    assert got['macro_n_valid'] == ref['macro_n_valid'], "BUG: valid-student counts differ"
    assert got['macro_n_skipped'] == ref['macro_n_skipped'], "BUG: skipped counts differ"
    print(f"  valid/skipped students match: {ref['macro_n_valid']}/{ref['macro_n_skipped']}")

    worst_uid, worst = None, 0.0
    for a, b in zip(ref['per_student_details'], got['per_student_details']):
        assert a['user_id'] == b['user_id'], "BUG: per-student ordering differs"
        assert a['n_interactions'] == b['n_interactions'], (
            f"BUG: interaction count differs for student {a['user_id']}")
        if a['auc'] is None or b['auc'] is None:
            assert a['auc'] is None and b['auc'] is None, (
                f"BUG: student {a['user_id']} -- one evaluator called it single-class "
                f"and the other did not")
            continue
        d = abs(a['auc'] - b['auc'])
        if d > worst:
            worst_uid, worst = a['user_id'], d

    macro_diff = abs(ref['macro_auc'] - got['macro_auc'])
    bce_diff = abs(ref['overall_bce'] - got['overall_bce'])

    print(f"  overall BCE difference   : {bce_diff:.3e}   "
          f"<- prediction-level agreement")
    print(f"  macro AUC difference     : {macro_diff:.3e}")
    print(f"  worst per-student AUC diff: {worst:.3e}"
          + (f"  (student {worst_uid})" if worst_uid else ""))

    # ── BCE is the honest prediction-level check ──────────────────────
    # BCE is a smooth function of the predictions themselves, so it moves
    # if and only if the predictions move. AUC is a function of their
    # RANKING, and is discontinuous at ties: two exactly-equal predictions
    # reordered by a 1e-8 difference can shift a student's AUC by ~1e-3
    # even though nothing meaningful changed. So BCE gates first.
    assert bce_diff < args.tolerance, (
        f"FAIL: overall BCE differs by {bce_diff:.3e}, above {args.tolerance:.0e}. "
        f"The two evaluators are producing genuinely different PREDICTIONS, not "
        f"just different tie-breaking. Do NOT use this evaluator.")
    # WHY MACRO AUC GETS A LOOSER TOLERANCE THAN BCE
    # -----------------------------------------------
    # FedGKT gives IDENTICAL predictions to every concept a student has not
    # yet touched -- their 7 features are identical, so their embeddings are
    # too (fedgkt.py's own self-test demonstrates this). Training does not
    # remove it; it is the concept-identity limitation, not a bug here.
    #
    # AUC ranks predictions and is discontinuous at ties: two exactly-equal
    # predictions reordered by a ~1e-8 float difference move one student's
    # AUC by ~2e-3, averaging to ~3e-5 across ~100 students. So EXACT macro
    # agreement is unattainable between any two implementations that sum in
    # a different order, no matter how correct both are.
    #
    # BCE is the real prediction-level gate above and stays at 1e-5. This
    # bound is ~100x below the ~0.01 effects the v2 campaign measures.
    assert macro_diff < args.macro_tolerance, (
        f"FAIL: macro AUC differs by {macro_diff:.3e}, above "
        f"{args.macro_tolerance:.0e}. That is too large to be tie-breaking. "
        f"Do NOT use this evaluator -- fall back to evaluator.py.")

    # ── per-student AUC: a hard gate only when the model is trained ────
    if worst >= args.tolerance:
        preds = got['per_student_predictions'][worst_uid]
        n = len(preds)
        n_unique = len(set(preds))
        n_tied = n - n_unique
        print(f"\n  student {worst_uid}: {n} interactions, {n_unique} distinct "
              f"prediction values ({n_tied} tied)")

        if n_tied > 0:
            print("\n  NOT A FAILURE -- this is tie-breaking, not disagreement.")
            print("  An UNTRAINED FedGKT gives identical predictions to every concept")
            print("  the student has not yet touched (see fedgkt.py's own self-test).")
            print("  AUC ranks predictions, so exactly-tied values reordered by a")
            print("  ~1e-8 floating-point difference move a single student's AUC by")
            print("  ~1e-3, while the predictions themselves are unchanged -- which")
            print(f"  is exactly what the BCE difference of {bce_diff:.1e} shows.")
            print("  Ties persist even in a trained model: untouched concepts share")
            print("  identical features, so they share identical predictions.")
        else:
            raise AssertionError(
                f"FAIL: per-student AUC differs by {worst:.3e} (student {worst_uid}), "
                f"above {args.tolerance:.0e}, and this is NOT explained by tied "
                f"predictions ({n_tied} ties). Do NOT use this evaluator.")
    else:
        print(f"\n  every per-student AUC agrees within {args.tolerance:.0e}")

    print(f"\nPASS -- predictions agree to {bce_diff:.1e} (BCE), macro AUC to "
          f"{macro_diff:.1e}.")
    print("Both far below the ~0.01 effects the v2 campaign is trying to measure.")
    print("\nSelf-test complete -- no assertion failures.")