"""
src/data/pkg_batched.py

BatchedPersonalKnowledgeGraph -- vectorized, multi-student version of
PersonalKnowledgeGraph (src/data/pkg.py), built for GPU-batched training in
the Colab experiment. NOT used by the local sequential training run --
that continues using the original, already-verified src/data/pkg.py
unchanged.

DESIGN DECISION (stated explicitly, not silent): this uses a FIXED-SIZE
batch with masking, not dynamic shrinking/re-indexing of edge_index as
students finish. Students who finish their sequence early keep receiving
harmless placeholder forward passes for the remainder of the batch, but
their loss contribution is explicitly zeroed and their internal state is
frozen (no further updates applied). This trades a small amount of wasted
compute (padding) for eliminating an entire category of bug: dynamically
re-indexing a batched graph's edge_index as students drop out is exactly
the kind of silent-corruption risk flagged as high-risk before building
this. A fixed batch size means edge_index (via PyG's Batch.from_data_list)
can be built ONCE per batch and reused for its entire duration -- only the
per-step loss mask changes, never the graph structure.

FEATURE SEMANTICS: every update rule here is intended to be NUMERICALLY
IDENTICAL to src/data/pkg.py for every ACTIVE student at every step --
this is not a reimplementation with different behavior, it's the same
math, vectorized across a batch dimension. This equivalence is verified in
the self-test below by replaying real students through both this batched
class and the original sequential class and comparing results directly,
not just assumed from reading the code.

One deliberate implementation simplification vs. the original: the
recent_accuracy_5/10 rolling history uses a right-aligned SHIFT-based
window (torch.cat + slice) rather than a circular buffer with a pointer.
A circular buffer is more "efficient" in principle but requires tracking
per-(student,node) write-position state and is much easier to get subtly
wrong. Since only one node per active student is updated per step (not all
835), the cost of shifting a length-10 vector is negligible -- so the
simpler, more obviously-correct approach was chosen over the
cleverer/riskier one, consistent with how the rest of this project has
prioritized correctness over micro-optimization throughout.
"""

import os
import sys

import torch

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))       # src/data
_SRC_DIR = os.path.dirname(_THIS_DIR)                          # src
_PROJECT_ROOT = os.path.dirname(_SRC_DIR)                      # fedgkt/ (or colab root)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.utils import config as cfg


class BatchedPersonalKnowledgeGraph:
    """
    Holds B students' node feature matrices (x) at once, as one
    [B, NUM_NODES, NUM_FEATURES] tensor, and updates them in a fully
    vectorized way (no Python loop over the batch dimension).

    Usage per batch-step, in this exact order (mirrors pkg.py's contract,
    just with a batch dimension and an active_mask added to BOTH calls --
    an earlier version only masked update() and left refresh_time_decay()
    unmasked, which caused a real, caught-by-testing bug: a finished
    student's time_decay kept being recomputed on padding steps using a
    frozen timestamp, silently diverging from the sequential ground truth):

        pkg.refresh_time_decay(time_done_batch, active_mask)          # BEFORE forward pass
        # ... run model forward pass using pkg.get_x(), get predictions for all B ...
        pkg.update(exercise_idx_batch, correct_batch, time_done_batch, active_mask)  # AFTER
    """

    def __init__(self, batch_size, device=None):
        self.B = batch_size
        n = cfg.NUM_NODES
        f = cfg.NUM_FEATURES
        self.device = device if device is not None else torch.device('cpu')

        self.x = torch.zeros((self.B, n, f), dtype=torch.float32, device=self.device)
        self.x[:, :, cfg.FEATURE_IDX['mastery_score']] = cfg.COLD_START_MASTERY

        self._raw_attempt_count = torch.zeros((self.B, n), dtype=torch.int32, device=self.device)
        self._raw_streak = torch.zeros((self.B, n), dtype=torch.int32, device=self.device)
        self._last_attempt_time = torch.full((self.B, n), -1, dtype=torch.int64, device=self.device)

        # right-aligned shifting history window: [..., -1] is most recent,
        # [..., 0] is oldest of the up-to-10 stored. Unwritten slots are 0
        # and excluded from means via _history_count, never read past it.
        self._history = torch.zeros((self.B, n, 10), dtype=torch.float32, device=self.device)
        self._history_count = torch.zeros((self.B, n), dtype=torch.int32, device=self.device)

        self._last_seen_time = torch.full((self.B,), -1, dtype=torch.int64, device=self.device)

        self._n = n
        self._f = f

    def refresh_time_decay(self, current_time, active_mask=None):
        """
        current_time: LongTensor [B] -- ONE timestamp per student (not a
        single shared scalar -- different students are at different points
        in their own independent real timelines, even at the same batch-step
        index). Recomputes time_decay for ALL nodes of ACTIVE students only,
        relative to each student's own current_time.

        active_mask: BoolTensor [B], or None (defaults to all-True). Rows
        where active_mask is False are left COMPLETELY UNTOUCHED -- their
        time_decay column stays exactly as it was when they last were
        active. This matters: without this masking, a finished student's
        time_decay would keep being recomputed on every subsequent padding
        step using a frozen/repeated timestamp, which can spuriously flip
        a just-touched node's decay from "not yet refreshed" (matching the
        sequential ground truth, which simply stops after the real
        sequence ends) to "refreshed with zero elapsed time" (decay=1.0,
        a real divergence -- this was caught empirically via the self-test
        below, not anticipated in advance).
        """
        current_time = current_time.to(torch.int64).to(self.device)
        assert current_time.shape == (self.B,), (
            f"BUG: current_time must have shape ({self.B},), got {tuple(current_time.shape)}"
        )

        if active_mask is None:
            active_mask = torch.ones(self.B, dtype=torch.bool, device=self.device)
        active_mask = active_mask.to(self.device)

        # only check chronological order for ACTIVE rows -- inactive rows'
        # padded timestamps are allowed to not advance (they're frozen/ignored)
        assert torch.all(current_time[active_mask] >= self._last_seen_time[active_mask]), (
            "BUG: refresh_time_decay called with a current_time earlier than a "
            "previously seen time for at least one ACTIVE student -- "
            "chronological order violated."
        )

        first_attempt_col = cfg.FEATURE_IDX['first_attempt']
        time_decay_col = cfg.FEATURE_IDX['time_decay']

        attempted_mask = self.x[:, :, first_attempt_col] == 1.0  # [B, n]

        # same int64-exact-delta -> float64-days -> float32 fix as pkg.py,
        # vectorized across the full [B, n] grid in one shot
        current_time_expanded = current_time.view(self.B, 1).expand(self.B, self._n)
        delta_us = current_time_expanded - self._last_attempt_time  # [B, n], exact int64
        days_since = delta_us.to(torch.float64) / float(cfg.MICROSECONDS_PER_DAY)
        decay = torch.exp(-cfg.TIME_DECAY_RATE * days_since).to(torch.float32)

        new_time_decay = torch.where(
            attempted_mask,
            decay,
            torch.zeros((self.B, self._n), dtype=torch.float32, device=self.device),
        )

        # apply the new value ONLY for active rows; inactive rows keep
        # whatever time_decay they already had (frozen)
        active_expanded = active_mask.view(self.B, 1).expand(self.B, self._n)
        self.x[:, :, time_decay_col] = torch.where(
            active_expanded,
            new_time_decay,
            self.x[:, :, time_decay_col],
        )

    def update(self, exercise_idx, correct, time_done, active_mask):
        """
        exercise_idx: LongTensor [B] -- node index each student is answering
                      this step. For INACTIVE (finished) students this value
                      is not read/used at all (any placeholder is fine).
        correct:      FloatTensor [B] -- 0.0/1.0. Same: ignored for inactive.
        time_done:    LongTensor [B]. Same: ignored for inactive.
        active_mask:  BoolTensor [B] -- True for students still within their
                      real sequence length this step, False for students who
                      have already finished (padding). Only True rows are
                      updated; False rows are left completely untouched,
                      freezing their state for the rest of the batch.
        """
        assert exercise_idx.shape == (self.B,)
        assert correct.shape == (self.B,)
        assert time_done.shape == (self.B,)
        assert active_mask.shape == (self.B,)
        assert active_mask.dtype == torch.bool

        if not torch.any(active_mask):
            return  # entire batch finished -- nothing to update

        exercise_idx = exercise_idx.to(torch.int64).to(self.device)
        correct = correct.to(torch.float32).to(self.device)
        time_done = time_done.to(torch.int64).to(self.device)
        active_mask = active_mask.to(self.device)

        active_correct_vals = correct[active_mask]
        assert torch.all((active_correct_vals == 0.0) | (active_correct_vals == 1.0)), (
            "BUG: correct contains non-binary values for an active student"
        )
        active_node_vals = exercise_idx[active_mask]
        assert torch.all((active_node_vals >= 0) & (active_node_vals < self._n)), (
            "BUG: exercise_idx out of range for an active student"
        )

        batch_idx = torch.arange(self.B, device=self.device)[active_mask]  # which students
        node_idx = exercise_idx[active_mask]                                 # their target node each
        c = correct[active_mask]
        t = time_done[active_mask]

        assert torch.all(t >= self._last_seen_time[active_mask]), (
            "BUG: update called with a time_done earlier than a previously "
            "seen time for at least one active student."
        )

        mastery_col = cfg.FEATURE_IDX['mastery_score']
        attempt_count_col = cfg.FEATURE_IDX['attempt_count']
        recent5_col = cfg.FEATURE_IDX['recent_accuracy_5']
        recent10_col = cfg.FEATURE_IDX['recent_accuracy_10']
        streak_col = cfg.FEATURE_IDX['streak']
        first_attempt_col = cfg.FEATURE_IDX['first_attempt']

        # 1. mastery_score -- EMA, vectorized gather-modify-scatter
        old_mastery = self.x[batch_idx, node_idx, mastery_col]
        new_mastery = cfg.MASTERY_EMA_ALPHA * c + (1 - cfg.MASTERY_EMA_ALPHA) * old_mastery
        self.x[batch_idx, node_idx, mastery_col] = new_mastery

        # 2. attempt_count -- log(1 + raw count)
        self._raw_attempt_count[batch_idx, node_idx] += 1
        self.x[batch_idx, node_idx, attempt_count_col] = torch.log1p(
            self._raw_attempt_count[batch_idx, node_idx].float()
        )

        # 3. recent_accuracy_5 / recent_accuracy_10 -- right-aligned shift window
        old_hist = self._history[batch_idx, node_idx]  # [k, 10], k = active count
        shifted = torch.cat([old_hist[:, 1:], c.unsqueeze(-1)], dim=-1)  # drop oldest, append new
        self._history[batch_idx, node_idx] = shifted

        old_count = self._history_count[batch_idx, node_idx]
        new_count = torch.clamp(old_count + 1, max=10)
        self._history_count[batch_idx, node_idx] = new_count

        # mean over the valid (right-aligned) portion of each row -- vectorized
        # via a mask, since different rows may have different valid counts
        idx10 = torch.arange(10, device=self.device).unsqueeze(0)  # [1, 10]
        valid10_mask = idx10 >= (10 - new_count.unsqueeze(-1))       # [k, 10]
        sum10 = (shifted * valid10_mask.float()).sum(dim=-1)
        mean10 = sum10 / new_count.float()
        self.x[batch_idx, node_idx, recent10_col] = mean10

        count5 = torch.clamp(new_count, max=5)
        valid5_mask = idx10 >= (10 - count5.unsqueeze(-1))
        sum5 = (shifted * valid5_mask.float()).sum(dim=-1)
        mean5 = sum5 / count5.float()
        self.x[batch_idx, node_idx, recent5_col] = mean5

        # 4. streak -- raw count caps at STREAK_CAP, then normalised to [0,1]
        old_streak = self._raw_streak[batch_idx, node_idx]
        is_correct = c == 1.0
        new_streak_if_correct = torch.clamp(old_streak + 1, max=cfg.STREAK_CAP)
        new_streak = torch.where(
            is_correct,
            new_streak_if_correct,
            torch.zeros_like(old_streak),
        )
        self._raw_streak[batch_idx, node_idx] = new_streak
        self.x[batch_idx, node_idx, streak_col] = new_streak.float() / cfg.STREAK_CAP

        # 5. first_attempt -- flips to 1 and stays there permanently
        self.x[batch_idx, node_idx, first_attempt_col] = 1.0

        # 6. record this timestamp for future time_decay refreshes
        self._last_attempt_time[batch_idx, node_idx] = t

        self._last_seen_time = torch.where(active_mask, time_done, self._last_seen_time)

    def get_x(self):
        """Return the current feature matrix. Shape: [B, NUM_NODES, NUM_FEATURES]."""
        return self.x


if __name__ == '__main__':
    print("=" * 70)
    print("BatchedPersonalKnowledgeGraph -- self-test against REAL data")
    print("Verifying numerical equivalence with the sequential pkg.py,")
    print("student by student, using students of DIFFERENT lengths in the")
    print("same batch (so padding/masking is genuinely exercised, not just")
    print("a same-length happy path).")
    print("=" * 70)

    from src.data.pkg import PersonalKnowledgeGraph  # the original, already-verified class

    real_ids = [233536, 45224, 21419]  # 30, 106, 9548 interactions -- deliberately mixed lengths
    missing = [
        uid for uid in real_ids
        if not os.path.exists(os.path.join(cfg.PKG_DIR, f'pkg_{uid}.pt'))
    ]
    assert not missing, f"Missing PKG files for self-test: {missing}"

    sequences = {}
    for uid in real_ids:
        data = torch.load(os.path.join(cfg.PKG_DIR, f'pkg_{uid}.pt'), weights_only=False)
        sequences[uid] = (data['exercise_idx'], data['correct'], data['time_done'])

    B = len(real_ids)
    lengths = {uid: sequences[uid][0].shape[0] for uid in real_ids}
    max_len = max(lengths.values())
    print(f"\nBatch of {B} real students: {[(uid, lengths[uid]) for uid in real_ids]}")
    print(f"max_len={max_len} -- shorter students will be masked out (padded) "
          f"once their real sequence ends, exercising the masking logic for "
          f"the majority of the batch's duration.")

    # ── ground truth: run each student through the ORIGINAL sequential class ──
    print("\n--- Computing ground truth via original sequential pkg.py ---")
    ground_truth_final_x = {}
    for uid in real_ids:
        ex_seq, c_seq, t_seq = sequences[uid]
        pkg_seq = PersonalKnowledgeGraph()
        for step in range(ex_seq.shape[0]):
            ex = int(ex_seq[step].item())
            c = float(c_seq[step].item())
            t = int(t_seq[step].item())
            pkg_seq.refresh_time_decay(t)
            pkg_seq.update(ex, c, t)
        ground_truth_final_x[uid] = pkg_seq.get_x().clone()
    print("  Ground truth computed for all 3 students.")

    # ── batched run: all 3 students together, padded to max_len ────────────
    print("\n--- Running the SAME 3 students through BatchedPersonalKnowledgeGraph ---")
    pkg_batch = BatchedPersonalKnowledgeGraph(batch_size=B)

    # For padding steps beyond a student's real length, we need SOME valid
    # placeholder timestamp (>= last real timestamp used) so the strict
    # chronological-order assertions don't fire even for masked-out rows --
    # using each student's own final real timestamp, repeated, is a simple,
    # safe choice (never earlier than anything already seen for that row).
    padded_ex = torch.zeros((B, max_len), dtype=torch.int64)
    padded_correct = torch.zeros((B, max_len), dtype=torch.float32)
    padded_time = torch.zeros((B, max_len), dtype=torch.int64)
    active_matrix = torch.zeros((B, max_len), dtype=torch.bool)

    for i, uid in enumerate(real_ids):
        ex_seq, c_seq, t_seq = sequences[uid]
        L = ex_seq.shape[0]
        padded_ex[i, :L] = ex_seq
        padded_correct[i, :L] = c_seq
        padded_time[i, :L] = t_seq
        active_matrix[i, :L] = True
        if L < max_len:
            padded_ex[i, L:] = ex_seq[-1]        # harmless placeholder node
            padded_correct[i, L:] = 0.0
            padded_time[i, L:] = t_seq[-1]        # repeat last real timestamp -- valid, non-decreasing

    for step in range(max_len):
        active_mask = active_matrix[:, step]
        pkg_batch.refresh_time_decay(padded_time[:, step], active_mask)
        # (forward pass would happen here in real training -- self-test only
        # needs to verify the PKG state itself, not model predictions)
        pkg_batch.update(padded_ex[:, step], padded_correct[:, step], padded_time[:, step], active_mask)

    print("  Batched run complete.")

    # ── compare final state, per student, against ground truth ──────────────
    print("\n--- Comparing batched result vs. sequential ground truth ---")
    max_diff_overall = 0.0
    for i, uid in enumerate(real_ids):
        batched_x = pkg_batch.get_x()[i]
        seq_x = ground_truth_final_x[uid]
        diff = (batched_x - seq_x).abs()
        max_diff = diff.max().item()
        max_diff_overall = max(max_diff_overall, max_diff)
        n_mismatched_cells = (diff > 1e-5).sum().item()
        print(f"  user_id={uid}  max_abs_diff={max_diff:.2e}  "
              f"cells_with_diff>1e-5: {n_mismatched_cells}/{835*7}")
        assert max_diff < 1e-4, (
            f"BUG: batched result diverges from sequential ground truth for "
            f"student {uid} -- max diff {max_diff:.2e} exceeds tolerance. "
            f"The batched update logic does not match the original."
        )

    print(f"\nMax difference across all 3 students, all 835 nodes, all 7 "
          f"features: {max_diff_overall:.2e}")
    print("(Small nonzero values here are expected float32 vs float64 "
          "intermediate rounding, same as CPU-vs-CPU platform drift seen "
          "elsewhere in this project -- not a correctness bug as long as "
          "they stay far below the 1e-4 tolerance checked above.)")

    print("\nSelf-test complete -- batched implementation verified numerically "
          "equivalent to the original sequential pkg.py for all 3 real "
          "students, despite mixed sequence lengths and padding/masking.")