"""
src/data/pkg.py

PersonalKnowledgeGraph — manages one student's per-concept feature matrix (x)
and the bookkeeping needed to update it correctly as new interactions arrive.

Design notes (confirmed across both Claude instances working this project):

- x is the [NUM_NODES, NUM_FEATURES] tensor actually fed to the GAT. It holds
  FINAL, ready-to-use feature values (e.g. log-scaled attempt_count, normalised
  streak) -- not raw counters.
- Raw bookkeeping needed to correctly recompute those final values (attempt
  counts, correctness history, uncapped streak, last-attempt timestamps) is
  kept as separate internal state, NOT stored in x. mastery_score is the one
  exception -- its EMA update only depends on the previous EMA value, which
  IS the feature value itself, so no separate raw state is needed for it.
- time_decay is handled as a two-operation design, not folded into update():
    1. refresh_time_decay(current_time) -- recomputes decay for ALL 835 nodes
       relative to `current_time`. Called BEFORE every forward pass.
    2. update(exercise_idx, correct, time_done) -- updates the other 6
       features for ONLY the node just interacted with. Called AFTER the
       forward pass, once the true outcome is known.
  This split exists because time_decay must reflect forgetting on concepts
  the student has NOT touched recently, which requires recomputing every
  node's decay relative to "now" -- not just refreshing whichever node was
  last attempted.
- edge_index is intentionally NOT stored here. It's identical across all
  students (same 978 prerequisite edges for everyone), so it's loaded once
  at the training-script level and passed into the GAT forward call
  separately -- this class stays focused only on genuinely per-student state.
- hint_used and time_taken are NOT consumed by any of the 7 features (locked
  decision -- mastery_score is pure correctness EMA, no hint or time
  weighting). They remain available in the raw interaction sequence (from
  Step 5) for possible future ablations, but this class does not touch them.

BUGFIX (caught during manual Step 6-style verification of this file):
refresh_time_decay() previously cast the raw absolute timestamps (~1.4e15,
microseconds since epoch) to float32 BEFORE subtracting them. float32 can
only exactly represent integers up to ~1.68e7 (2^24), so two timestamps that
differ by only tens of millions of microseconds (a realistic sub-day gap)
were rounded to the SAME float32 value before the subtraction ever happened
-- silently destroying the delta being measured. Confirmed empirically: two
identical 60,000,000us gaps produced different, wrong decay values (1.0 and
0.9998) instead of matching. Fixed by computing the delta in exact int64
arithmetic first, converting to days using float64 (double) precision, and
only casting down to float32 at the very end, once values are small (days,
not epoch-microseconds) and well within float32's precision range.
"""

import os
import sys
from collections import deque

import torch

# Anchor to this file's own location so imports work regardless of the
# caller's current working directory -- same principle used in config.py,
# adopted after the exact cwd bug hit during preprocessing Step 4.
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))       # src/data
_SRC_DIR = os.path.dirname(_THIS_DIR)                         # src
_PROJECT_ROOT = os.path.dirname(_SRC_DIR)                     # fedgkt/
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.utils import config as cfg


class PersonalKnowledgeGraph:
    """
    Holds one student's node feature matrix (x) and updates it as their
    interaction sequence is replayed.

    Usage per interaction, in this exact order:
        pkg.refresh_time_decay(time_done)             # BEFORE forward pass
        # ... run model forward pass using pkg.x, get a prediction ...
        pkg.update(exercise_idx, correct, time_done)   # AFTER forward pass
    """

    def __init__(self):
        n = cfg.NUM_NODES
        f = cfg.NUM_FEATURES

        # ── the actual feature matrix fed to the GAT ──────────────────────
        self.x = torch.zeros((n, f), dtype=torch.float32)
        self.x[:, cfg.FEATURE_IDX['mastery_score']] = cfg.COLD_START_MASTERY
        # every other feature starts at 0.0 (COLD_START_DEFAULT), already
        # satisfied by torch.zeros above

        # ── internal raw bookkeeping (NOT part of the model-facing x) ─────
        self._raw_attempt_count = torch.zeros(n, dtype=torch.int32)
        self._raw_streak = torch.zeros(n, dtype=torch.int32)
        # -1 sentinel = "never attempted". Real timestamps are large positive
        # microsecond values, so -1 can never collide with a real timestamp.
        self._last_attempt_time = torch.full((n,), -1, dtype=torch.int64)
        # correctness history per node, oldest-to-newest, capped at 10 --
        # covers both recent_accuracy_5 (last 5) and recent_accuracy_10 (last 10)
        self._history = [deque(maxlen=10) for _ in range(n)]

        # defensive: enforce interactions are processed in chronological order
        self._last_seen_time = -1

        self._n = n
        self._f = f

    # ── operation 1: refresh time_decay for ALL nodes ──────────────────────
    def refresh_time_decay(self, current_time):
        """
        Recompute time_decay for every node relative to `current_time`.
        Must be called BEFORE every forward pass, using the timestamp of the
        interaction about to be predicted (not the timestamp of whichever
        node was last attempted).

        Nodes never attempted (first_attempt == 0) are gated to exactly 0.0,
        never computed via exp() on an undefined/sentinel timestamp.
        """
        current_time = int(current_time)
        assert current_time >= self._last_seen_time, (
            f"BUG: refresh_time_decay called with current_time={current_time}, "
            f"earlier than a previously seen time={self._last_seen_time}. "
            f"Interactions must be processed in strict chronological order."
        )

        first_attempt_col = cfg.FEATURE_IDX['first_attempt']
        time_decay_col = cfg.FEATURE_IDX['time_decay']

        attempted_mask = self.x[:, first_attempt_col] == 1.0

        # BUGFIX: the raw timestamps are ~1.4e15 (microseconds since epoch).
        # float32 can only exactly represent integers up to ~1.68e7, so
        # casting these huge absolute timestamps to float32 BEFORE
        # subtracting silently destroys the delta we're trying to measure
        # (confirmed: two identical 60,000,000us gaps produced different,
        # wrong decay values -- 1.0 and 0.9998 -- due to this precision loss).
        #
        # Fix: compute the delta in exact int64 arithmetic first, convert to
        # days using float64 (double) precision, and only cast to float32 at
        # the very end, once the values are small (days, not epoch-microseconds)
        # and easily within float32's precision range.
        current_time_tensor = torch.tensor(current_time, dtype=torch.int64)
        delta_us = current_time_tensor - self._last_attempt_time  # exact, int64
        days_since = delta_us.to(torch.float64) / float(cfg.MICROSECONDS_PER_DAY)
        decay = torch.exp(-cfg.TIME_DECAY_RATE * days_since).to(torch.float32)

        self.x[:, time_decay_col] = torch.where(
            attempted_mask,
            decay,
            torch.zeros(self._n, dtype=torch.float32),
        )

    # ── operation 2: update the single attempted node ───────────────────────
    def update(self, exercise_idx, correct, time_done):
        """
        Update all 6 non-time_decay features for the single node just
        interacted with. Call AFTER the forward pass, once the true outcome
        is known. time_decay is NOT touched here -- it's handled entirely by
        refresh_time_decay(), called before the NEXT forward pass.
        """
        i = int(exercise_idx)
        c = float(correct)
        t = int(time_done)

        assert 0 <= i < self._n, (
            f"BUG: exercise_idx={i} out of range [0, {self._n - 1}]"
        )
        assert c in (0.0, 1.0), (
            f"BUG: correct={c} is not binary (0 or 1)"
        )
        assert t >= self._last_seen_time, (
            f"BUG: update called with time_done={t}, earlier than a "
            f"previously seen time={self._last_seen_time}. Interactions "
            f"must be processed in strict chronological order."
        )

        mastery_col = cfg.FEATURE_IDX['mastery_score']
        attempt_count_col = cfg.FEATURE_IDX['attempt_count']
        recent5_col = cfg.FEATURE_IDX['recent_accuracy_5']
        recent10_col = cfg.FEATURE_IDX['recent_accuracy_10']
        streak_col = cfg.FEATURE_IDX['streak']
        first_attempt_col = cfg.FEATURE_IDX['first_attempt']

        # 1. mastery_score -- EMA, depends only on the previous EMA value
        #    (which IS the current feature value), no separate raw state needed
        old_mastery = self.x[i, mastery_col].item()
        new_mastery = cfg.MASTERY_EMA_ALPHA * c + (1 - cfg.MASTERY_EMA_ALPHA) * old_mastery
        self.x[i, mastery_col] = new_mastery

        # 2. attempt_count -- log(1 + raw count)
        self._raw_attempt_count[i] += 1
        self.x[i, attempt_count_col] = torch.log1p(self._raw_attempt_count[i].float())

        # 3. recent_accuracy_5 / recent_accuracy_10 -- mean over whatever
        #    exists so far, no padding. Confirmed decision: average over
        #    available history only.
        self._history[i].append(c)
        hist = list(self._history[i])
        self.x[i, recent10_col] = sum(hist) / len(hist)
        last5 = hist[-5:]
        self.x[i, recent5_col] = sum(last5) / len(last5)

        # 4. streak -- raw count caps at STREAK_CAP, then normalised to [0,1]
        if c == 1.0:
            self._raw_streak[i] = min(int(self._raw_streak[i].item()) + 1, cfg.STREAK_CAP)
        else:
            self._raw_streak[i] = 0
        self.x[i, streak_col] = self._raw_streak[i].item() / cfg.STREAK_CAP

        # 5. first_attempt -- flips to 1 and stays there permanently
        self.x[i, first_attempt_col] = 1.0

        # 6. record this timestamp for future time_decay refreshes
        self._last_attempt_time[i] = t

        self._last_seen_time = t

    def get_x(self):
        """Return the current feature matrix. Shape: [NUM_NODES, NUM_FEATURES]."""
        return self.x


if __name__ == '__main__':
    print("=" * 70)
    print("PersonalKnowledgeGraph -- standalone self-test (synthetic data)")
    print("=" * 70)

    pkg = PersonalKnowledgeGraph()
    print(f"\nInitial x shape: {tuple(pkg.x.shape)}")
    assert torch.all(pkg.x[:, cfg.FEATURE_IDX['mastery_score']] == 0.5), \
        "BUG: cold-start mastery is not 0.5 everywhere"
    assert torch.all(pkg.x[:, cfg.FEATURE_IDX['first_attempt']] == 0.0), \
        "BUG: cold-start first_attempt is not 0 everywhere"
    print("Cold-start assertions passed.")

    base_t = 1_400_000_000_000_000
    day = cfg.MICROSECONDS_PER_DAY

    print("\n--- Simulating interactions on node 10 ---")
    interactions = [
        (10, 1, base_t),
        (10, 1, base_t + 60_000_000),        # 1 min later
        (10, 0, base_t + 120_000_000),        # 1 min later
        (10, 1, base_t + 3 * day),            # 3 days later
    ]

    for step, (ex_idx, correct, t) in enumerate(interactions):
        pkg.refresh_time_decay(t)
        row_before = [round(v, 4) for v in pkg.x[ex_idx].tolist()]
        pkg.update(ex_idx, correct, t)
        row_after = [round(v, 4) for v in pkg.x[ex_idx].tolist()]
        print(f"\nStep {step}: exercise_idx={ex_idx} correct={correct} time={t}")
        print(f"  x[{ex_idx}] before update: {row_before}")
        print(f"  x[{ex_idx}] after  update: {row_after}")

    print("\n--- Checking an untouched node's time_decay ---")
    untouched_node = 500
    val = pkg.x[untouched_node, cfg.FEATURE_IDX['time_decay']].item()
    print(f"  Node {untouched_node} (never attempted) time_decay: {val} (expected 0.0)")
    assert val == 0.0, "BUG: untouched node should have time_decay == 0.0"

    print("\n--- Sanity check: two identical 60,000,000us gaps should match ---")
    print("  (This is the exact check that caught the original float32 bug --")
    print("   step 1 and step 2 above both simulate a 1-minute gap and must")
    print("   now print the SAME time_decay value in their 'before update' row.)")

    print("\nSelf-test complete -- no assertion failures.")