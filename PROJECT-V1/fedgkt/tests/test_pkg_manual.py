"""
tests/test_pkg_manual.py

Standalone manual verification of PersonalKnowledgeGraph, using a REAL
student's saved interaction sequence (not synthetic data). Prints the full
feature row for the attempted node at every step, so the numbers can be
checked by hand against the update rules in src/data/pkg.py.

Run from anywhere -- this script anchors to its own location, same as
config.py and pkg.py.
"""

import os
import sys

import torch

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))       # tests/
_PROJECT_ROOT = os.path.dirname(_THIS_DIR)                    # fedgkt/
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.utils import config as cfg
from src.data.pkg import PersonalKnowledgeGraph

# the 30-interaction student already manually verified in Step 6
TEST_USER_ID = 233536

PKG_PATH = os.path.join(cfg.PKG_DIR, f'pkg_{TEST_USER_ID}.pt')

print("=" * 70)
print(f"Manual PKG verification -- real student user_id={TEST_USER_ID}")
print("=" * 70)

assert os.path.exists(PKG_PATH), f"File not found: {PKG_PATH}"
data = torch.load(PKG_PATH, weights_only=False)

exercise_idx = data['exercise_idx']
correct = data['correct']
time_done = data['time_done']
n = exercise_idx.shape[0]

print(f"\nLoaded {n} real interactions for user {TEST_USER_ID}")
print(f"Feature order: {cfg.FEATURE_NAMES}")

pkg = PersonalKnowledgeGraph()

print("\n--- Step-by-step replay ---")
for step in range(n):
    ex = int(exercise_idx[step].item())
    c = float(correct[step].item())
    t = int(time_done[step].item())

    pkg.refresh_time_decay(t)
    row_before = [round(v, 4) for v in pkg.x[ex].tolist()]

    pkg.update(ex, c, t)

    # NOTE: this extra refresh call is ONLY for this printout, so the "AFTER"
    # row shows a freshly-attempted node's decay as 1.0 for readability.
    # It is NOT part of the real training contract -- in actual training,
    # refresh_time_decay() is called once per step, right before the NEXT
    # interaction's forward pass, not immediately after update().
    pkg.refresh_time_decay(t)
    row_after = [round(v, 4) for v in pkg.x[ex].tolist()]

    print(f"\nStep {step:>2}: exercise_idx={ex:>4}  correct={int(c)}  time_done={t}")
    print(f"  x[{ex}] BEFORE (used for this step's prediction): {row_before}")
    print(f"  x[{ex}] AFTER  (post-update, decay refreshed for display): {row_after}")

# ── final sanity checks against known facts about this student ──────────────
print("\n" + "=" * 70)
print("Final sanity checks")
print("=" * 70)

first_attempt_col = cfg.FEATURE_IDX['first_attempt']
n_touched_nodes = int((pkg.x[:, first_attempt_col] == 1.0).sum().item())
n_unique_exercises_in_sequence = int(torch.unique(exercise_idx).shape[0])

print(f"\nNodes with first_attempt=1: {n_touched_nodes}")
print(f"Unique exercise_idx values in this student's sequence: {n_unique_exercises_in_sequence}")
assert n_touched_nodes == n_unique_exercises_in_sequence, (
    f"BUG: first_attempt count ({n_touched_nodes}) doesn't match actual "
    f"unique exercises attempted ({n_unique_exercises_in_sequence})"
)
print("MATCH -- first_attempt flags exactly track real unique exercises attempted.")

untouched_mask = pkg.x[:, first_attempt_col] == 0.0
untouched_decay_sum = pkg.x[untouched_mask, cfg.FEATURE_IDX['time_decay']].sum().item()
untouched_mastery_ok = torch.all(
    pkg.x[untouched_mask, cfg.FEATURE_IDX['mastery_score']] == cfg.COLD_START_MASTERY
).item()

print(f"\nUntouched nodes: {int(untouched_mask.sum().item())}")
print(f"  Sum of time_decay across untouched nodes (must be 0.0): {untouched_decay_sum}")
print(f"  All untouched nodes still at cold-start mastery (0.5): {untouched_mastery_ok}")
assert untouched_decay_sum == 0.0, "BUG: an untouched node has nonzero time_decay"
assert untouched_mastery_ok, "BUG: an untouched node's mastery_score was modified"

print("\nAll sanity checks passed.")
print("Manual verification complete -- review the step-by-step rows above by hand.")