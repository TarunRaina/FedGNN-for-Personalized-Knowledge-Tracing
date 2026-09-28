"""
src/models/fedgkt_v2.py

FedGKT v2 -- the original model plus two additions, aimed at two problems
measured in the v1 model rather than guessed at.

Everything else is unchanged: the same 7 hand-designed features from
pkg.py, the same 978-edge prerequisite graph, the same 3-layer GAT, the
same personal output head, the same federated base/personal split.

THE TWO PROBLEMS
----------------
Measured on a real student (45224, 106 interactions): of 835 concepts,
13 were touched. 822 rows of the input -- 98.4% -- were the identical
cold-start row (mastery 0.5, everything else 0). Across all 835 nodes,
mastery_score had a standard deviation of 0.05 and every other feature
was near zero.

  (1) THE BLANK 98% CARRIES NO INFORMATION.
      Every untouched concept looks identical to every other untouched
      concept, so the model cannot know that one is hard and another
      easy before the student attempts them. fedgkt.py's own self-test
      shows this directly: on a cold-start student, concepts 0, 42 and
      834 receive IDENTICAL predictions. Every pyKT baseline learns a
      per-concept representation; v1 has none.

  (2) THE ONE INFORMATIVE ROW GETS DILUTED.
      A GAT layer mixes each node with its neighbours; three layers mix
      three times. With 98.4% of neighbours blank, the strongest single
      predictor available -- this student's own mastery of the concept
      being asked -- is smeared into mostly-empty neighbourhoods by the
      time it reaches the head. This is over-smoothing, a documented
      failure mode of deep GNNs.

THE TWO ADDITIONS
-----------------
  (1) LEARNED PER-CONCEPT COLD-START PRIOR  (835 parameters)
      cfg.COLD_START_MASTERY is a hand-chosen 0.5 applied to every
      concept. Here it becomes a learned value per concept, used in
      place of 0.5 for concepts the student has not yet attempted.
      Touched concepts keep their real EMA mastery, untouched.

      Stored as a logit and passed through a sigmoid, so the prior is
      always a valid [0,1] mastery value and the gradient never dies at
      the boundary (a hard clamp would). Logits start at 0, so every
      prior starts at sigmoid(0) = 0.5 -- exactly v1's behaviour.

      This is interpretable, arguably more so than the constant it
      replaces: after training, sort the priors and read off which
      concepts the model considers hard. That is a figure for the paper,
      and a sanity check -- if the lowest priors are not genuinely
      difficult topics, something is wrong.

      It belongs to the SHARED base in FedPer: "quadratics are hard" is
      knowledge about the subject, not about any one student.

  (2) RAW-FEATURE SKIP CONNECTION INTO THE HEAD  (~500 parameters)
      The head currently sees only the GAT's 64-dim output for the
      target node. Here it also receives that node's raw 7 features
      directly, through a small linear layer added to the head's first
      layer. The smoothed view is still there; the unsmoothed one is no
      longer lost.

      The skip layer is initialised to ZERO, so at epoch 0 it
      contributes nothing and the model computes exactly what v1
      computes. It only diverges as it learns.

      It belongs to the PERSONAL head: it is part of turning a
      representation into that student's prediction.

PAIRED INITIALISATION -- WHY THE CONSTRUCTION ORDER IS DELIBERATE
------------------------------------------------------------------
The v1 vs v2 comparison is only sharp if both models start from the same
weights. Two runs with the same seed stay within ~0.003 of each other;
two runs with different seeds differ by ~0.008-0.012. Since new
parameters change how the random generator hands out values, a naive v2
would effectively be a different-seed run, and a genuine +0.005 would be
indistinguishable from luck.

So this class constructs the GAT and the head FIRST, with exactly the
shapes and in exactly the order FedGKT does, consuming exactly the same
random draws. Only afterwards does it create the new parts -- and both
of those are filled with constants (logits 0, skip weights 0), which
consume no randomness at all.

Result: every parameter v1 has, v2 has identically, AND at
initialisation v2 computes the identical function. The self-test at the
bottom asserts both, so this is checked rather than claimed.

WHAT THIS DOES NOT CHANGE
-------------------------
  - pkg.py, and the 7 features: untouched
  - the graph, the splits, the .pt data: untouched
  - what leaves a client's device: nothing new. The priors are shared
    base parameters, averaged across clients, like the GAT layers.

ONE HONEST PRIVACY NOTE FOR THE PAPER
--------------------------------------
A client only produces gradients for the prior rows of concepts it
actually attempted. A server observing updates could therefore infer
WHICH CONCEPTS a student worked on (not their answers). This is a real,
if modest, additional exposure over v1, and belongs in the privacy
discussion alongside the standard mitigation (differential privacy,
already a Phase 2 topic). The personal head still never leaves the
device.
"""

import os
import sys

import torch
import torch.nn as nn

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))       # src/models
_SRC_DIR = os.path.dirname(_THIS_DIR)                          # src
_PROJECT_ROOT = os.path.dirname(_SRC_DIR)                      # fedgkt_v2/
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.utils import config as cfg
from src.models.gat import GAT


class FedGKTv2(nn.Module):
    """
    Same interface as FedGKT: forward(x, edge_index, exercise_idx).

    x may be [NUM_NODES, NUM_FEATURES] for a single student, or
    [B * NUM_NODES, NUM_FEATURES] for a batch of B students flattened
    into one disjoint graph (as centralised_batched.py builds it). In the
    batched case row i belongs to concept i % NUM_NODES, which is how the
    per-concept prior is matched to the right row.
    """

    def __init__(self, use_prior=True, use_skip=True):
        super().__init__()

        # ── built FIRST, exactly as FedGKT builds them, so the random
        #    draws -- and therefore the initial weights -- are identical ──
        self.gat = GAT()
        self.head = nn.Sequential(
            nn.Linear(cfg.GAT_LAYER_3_DIM, cfg.HEAD_HIDDEN_DIM),
            nn.ReLU(),
            nn.Dropout(cfg.DROPOUT),
            nn.Linear(cfg.HEAD_HIDDEN_DIM, 1),
            nn.Sigmoid(),
        )

        # ── new parts, created AFTER, filled with constants (no RNG) ──
        self.use_prior = use_prior
        self.use_skip = use_skip

        if use_prior:
            # logit 0 -> sigmoid(0) = 0.5 = cfg.COLD_START_MASTERY
            self.concept_prior_logit = nn.Parameter(
                torch.zeros(cfg.NUM_NODES, dtype=torch.float32))

        if use_skip:
            self.skip = nn.Linear(cfg.NUM_FEATURES, cfg.HEAD_HIDDEN_DIM)
            nn.init.zeros_(self.skip.weight)
            nn.init.zeros_(self.skip.bias)

        self._mastery_col = cfg.FEATURE_IDX['mastery_score']
        self._first_attempt_col = cfg.FEATURE_IDX['first_attempt']

    # ── the prior substitution ────────────────────────────────────────
    def apply_prior(self, x):
        """
        Replaces the cold-start mastery of UNTOUCHED concepts with this
        model's learned per-concept prior. Touched concepts are left
        exactly as pkg.py computed them.

        Returns a new tensor; x is never modified in place (pkg_batched
        reuses its buffer across steps, and training defers backward, so
        in-place edits here would corrupt the graph).
        """
        if not self.use_prior:
            return x

        n_rows = x.shape[0]
        assert n_rows % cfg.NUM_NODES == 0, (
            f"BUG: x has {n_rows} rows, not a multiple of {cfg.NUM_NODES}. "
            f"Cannot map rows to concepts.")

        # row i belongs to concept i % NUM_NODES -- this is the mapping the
        # flattened batched graph relies on; getting it wrong would silently
        # give every student after the first the wrong concept priors
        concept_ids = torch.arange(n_rows, device=x.device) % cfg.NUM_NODES

        prior = torch.sigmoid(self.concept_prior_logit)[concept_ids]
        untouched = x[:, self._first_attempt_col] == 0.0
        new_mastery = torch.where(untouched, prior, x[:, self._mastery_col])

        x = x.clone()
        x[:, self._mastery_col] = new_mastery
        return x

    def forward(self, x, edge_index, exercise_idx):
        x = self.apply_prior(x)

        node_embeddings = self.gat(x, edge_index)

        if isinstance(exercise_idx, int):
            idx_tensor = torch.tensor([exercise_idx], dtype=torch.long,
                                      device=node_embeddings.device)
            return_scalar = True
        else:
            idx_tensor = exercise_idx
            return_scalar = False

        assert idx_tensor.dtype == torch.long, (
            f"BUG: exercise_idx tensor must be dtype long, got {idx_tensor.dtype}")
        assert torch.all(idx_tensor >= 0) and torch.all(idx_tensor < node_embeddings.shape[0]), (
            f"BUG: exercise_idx contains values out of range "
            f"[0, {node_embeddings.shape[0] - 1}]")

        selected = node_embeddings[idx_tensor]          # [k, GAT_LAYER_3_DIM]

        # head, with the raw-feature skip added into its first layer
        h = self.head[0](selected)                       # Linear(64 -> 32)
        if self.use_skip:
            h = h + self.skip(x[idx_tensor])             # Linear(7 -> 32), zero-init
        for layer in self.head[1:]:                      # ReLU, Dropout, Linear, Sigmoid
            h = layer(h)
        preds = h.squeeze(-1)

        return preds[0] if return_scalar else preds

    # ── FedPer split ──────────────────────────────────────────────────
    def base_parameters(self):
        """Shared across clients: the GAT, plus the per-concept priors."""
        params = list(self.gat.parameters())
        if self.use_prior:
            params.append(self.concept_prior_logit)
        return iter(params)

    def head_parameters(self):
        """Stays local to each student: the output head and its skip."""
        params = list(self.head.parameters())
        if self.use_skip:
            params += list(self.skip.parameters())
        return iter(params)

    # ── interpretability ──────────────────────────────────────────────
    def learned_priors(self):
        """The learned cold-start mastery per concept, as a [835] tensor in [0,1]."""
        return torch.sigmoid(self.concept_prior_logit).detach()


if __name__ == '__main__':
    from src.models.fedgkt import FedGKT

    print("=" * 70)
    print("FedGKTv2 -- self-test")
    print("=" * 70)

    edge_index = torch.load(cfg.EDGE_INDEX_PATH, weights_only=False)
    print(f"\nedge_index: {tuple(edge_index.shape)}")

    torch.manual_seed(cfg.RANDOM_SEED)
    v1 = FedGKT()
    torch.manual_seed(cfg.RANDOM_SEED)
    v2 = FedGKTv2()

    n1 = sum(p.numel() for p in v1.parameters())
    n2 = sum(p.numel() for p in v2.parameters())
    base2 = sum(p.numel() for p in v2.base_parameters())
    head2 = sum(p.numel() for p in v2.head_parameters())
    print(f"\nv1 parameters: {n1:,}")
    print(f"v2 parameters: {n2:,}   (+{n2 - n1:,} = {cfg.NUM_NODES} priors + "
          f"{(cfg.NUM_FEATURES + 1) * cfg.HEAD_HIDDEN_DIM} skip)")
    print(f"  v2 base (shared)  : {base2:,}")
    print(f"  v2 head (personal): {head2:,}")
    assert base2 + head2 == n2, "BUG: base + head don't sum to total"

    # ── 1. paired initialisation: every shared parameter identical ────
    print("\n--- 1. Paired initialisation ---")
    p1 = dict(v1.named_parameters())
    p2 = dict(v2.named_parameters())
    shared = sorted(set(p1) & set(p2))
    assert set(p1) - set(p2) == set(), f"v1 has parameters v2 lacks: {set(p1) - set(p2)}"
    worst = 0.0
    for name in shared:
        d = (p1[name] - p2[name]).abs().max().item()
        worst = max(worst, d)
    print(f"  {len(shared)} shared parameter tensors, max difference: {worst:.3e}")
    assert worst == 0.0, (
        "BUG: v1 and v2 do not start from identical weights. The paired "
        "comparison would be a different-seed comparison, with ~3x the noise.")
    print(f"  v2-only: {sorted(set(p2) - set(p1))}")

    # ── 2. at init, v2 computes exactly what v1 computes ──────────────
    print("\n--- 2. Identical function at initialisation ---")
    torch.manual_seed(0)
    x = torch.zeros((cfg.NUM_NODES, cfg.NUM_FEATURES))
    x[:, cfg.FEATURE_IDX['mastery_score']] = cfg.COLD_START_MASTERY
    for node in (10, 42, 500):                      # a few touched concepts
        x[node, cfg.FEATURE_IDX['mastery_score']] = 0.8
        x[node, cfg.FEATURE_IDX['first_attempt']] = 1.0
        x[node, cfg.FEATURE_IDX['streak']] = 0.3

    idx = torch.tensor([0, 10, 42, 500, 834], dtype=torch.long)
    v1.eval(); v2.eval()
    with torch.no_grad():
        y1 = v1(x, edge_index, idx)
        y2 = v2(x, edge_index, idx)
    d = (y1 - y2).abs().max().item()
    print(f"  v1: {[round(v, 6) for v in y1.tolist()]}")
    print(f"  v2: {[round(v, 6) for v in y2.tolist()]}")
    print(f"  max difference: {d:.3e}")
    assert d == 0.0, (
        "BUG: v2 does not reproduce v1 at initialisation. The priors start at "
        "0.5 and the skip starts at zero, so it must.")

    # ── 3. the prior actually does something once it differs ──────────
    print("\n--- 3. The prior changes untouched concepts (and only those) ---")
    with torch.no_grad():
        v2.concept_prior_logit[0] = 2.0          # concept 0 -> sigmoid(2) = 0.88
        v2.concept_prior_logit[834] = -2.0       # concept 834 -> 0.12
        y2b = v2(x, edge_index, idx)
    print(f"  learned prior for concept 0  : {v2.learned_priors()[0]:.4f}")
    print(f"  learned prior for concept 834: {v2.learned_priors()[834]:.4f}")
    changed = [(int(i), abs(a - b) > 1e-6) for i, a, b in zip(idx, y2.tolist(), y2b.tolist())]
    print(f"  predictions changed: {changed}")
    assert abs(y2b[0] - y2[0]) > 1e-6, "BUG: changing an untouched concept's prior did nothing"
    print("  (touched concepts 10/42/500 may shift too -- correctly so: they are")
    print("   graph neighbours of nodes whose priors changed.)")

    # ── 4. gradients reach the new parameters ─────────────────────────
    print("\n--- 4. Gradient flow ---")
    v2.train()
    out = v2(x, edge_index, torch.tensor([42], dtype=torch.long))
    out.sum().backward()
    g_prior = v2.concept_prior_logit.grad
    g_skip = v2.skip.weight.grad
    print(f"  concept_prior_logit grad: nonzero on {int((g_prior.abs() > 0).sum())} "
          f"of {cfg.NUM_NODES} concepts")
    print(f"  skip.weight grad max    : {g_skip.abs().max().item():.3e}")
    assert g_prior is not None and g_prior.abs().sum() > 0, "BUG: no gradient to the priors"
    assert g_skip is not None and g_skip.abs().sum() > 0, (
        "BUG: no gradient to the skip layer -- zero-init must not block learning")
    for name, p in v2.named_parameters():
        assert p.grad is not None, f"BUG: no gradient reached {name}"

    # ── 5. batched/flattened layout maps rows to the right concepts ───
    print("\n--- 5. Flattened batch layout ---")
    B = 3
    x_flat = x.repeat(B, 1)                              # [B*835, 7]
    assert x_flat.shape == (B * cfg.NUM_NODES, cfg.NUM_FEATURES)
    with torch.no_grad():
        applied = v2.apply_prior(x_flat)
    mcol = cfg.FEATURE_IDX['mastery_score']
    for b in range(B):
        off = b * cfg.NUM_NODES
        assert torch.allclose(applied[off + 0, mcol], v2.learned_priors()[0]), (
            f"BUG: student {b}'s concept 0 did not get concept 0's prior")
        assert torch.allclose(applied[off + 834, mcol], v2.learned_priors()[834]), (
            f"BUG: student {b}'s concept 834 did not get concept 834's prior")
        assert torch.allclose(applied[off + 10, mcol], torch.tensor(0.8)), (
            f"BUG: student {b}'s TOUCHED concept 10 was overwritten by the prior")
    print(f"  all {B} students: untouched concepts got their own concept's prior,")
    print("  touched concepts kept their real mastery -- OK")

    # ── 6. the ablation switches ──────────────────────────────────────
    print("\n--- 6. Ablation switches ---")
    torch.manual_seed(cfg.RANDOM_SEED)
    prior_only = FedGKTv2(use_prior=True, use_skip=False)
    torch.manual_seed(cfg.RANDOM_SEED)
    skip_only = FedGKTv2(use_prior=False, use_skip=True)
    prior_only.eval(); skip_only.eval()   # dropout off, or the comparison is random
    print(f"  prior only: {sum(p.numel() for p in prior_only.parameters()):,} params")
    print(f"  skip only : {sum(p.numel() for p in skip_only.parameters()):,} params")
    with torch.no_grad():
        assert (prior_only(x, edge_index, idx) - y1).abs().max().item() == 0.0
        assert (skip_only(x, edge_index, idx) - y1).abs().max().item() == 0.0
    print("  both reproduce v1 at initialisation -- OK")

    print("\nSelf-test complete -- no assertion failures.")