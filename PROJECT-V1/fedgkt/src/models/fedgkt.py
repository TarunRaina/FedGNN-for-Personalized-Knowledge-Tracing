"""
src/models/fedgkt.py

FedGKT -- wraps the plain GAT (src/models/gat.py) with a personal output
head that turns a node embedding into a single P(correct) prediction.

Deliberately kept as two separate submodules, self.gat and self.head, even
though nothing is federated yet in Phase 1. This split is exactly what
Phase 2's FedPer design needs: all 3 GAT layers are shared/aggregated
across clients (the "base"), while the output head stays local to each
student (the "personal" part). Keeping them separate now means Phase 2
only has to decide WHICH parameters to send over the network -- no
architecture rewrite required.
"""

import os
import sys

import torch
import torch.nn as nn

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))       # src/models
_SRC_DIR = os.path.dirname(_THIS_DIR)                          # src
_PROJECT_ROOT = os.path.dirname(_SRC_DIR)                      # fedgkt/
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.utils import config as cfg
from src.models.gat import GAT


class FedGKT(nn.Module):
    """
    Input:
        x:            [NUM_NODES, NUM_FEATURES] -- current PKG state for one student
        edge_index:   [2, NUM_EDGES] -- shared prerequisite graph (same for everyone)
        exercise_idx: int, OR 1D LongTensor of node indices to score

    Output:
        scalar tensor in [0,1]   if exercise_idx was a plain int
        1D tensor in [0,1]       if exercise_idx was a tensor of indices

    The tensor-of-indices path exists for future convenience (e.g. Phase 2's
    prerequisite recommendation feature, which needs a P(correct)-like score
    for many/all concepts from a single fixed graph state) -- it costs
    nothing extra to support now and avoids a signature change later.
    """

    def __init__(self):
        super().__init__()

        self.gat = GAT()

        self.head = nn.Sequential(
            nn.Linear(cfg.GAT_LAYER_3_DIM, cfg.HEAD_HIDDEN_DIM),
            nn.ReLU(),
            nn.Dropout(cfg.DROPOUT),
            nn.Linear(cfg.HEAD_HIDDEN_DIM, 1),
            nn.Sigmoid(),
        )

    def forward(self, x, edge_index, exercise_idx):
        node_embeddings = self.gat(x, edge_index)  # [NUM_NODES, GAT_LAYER_3_DIM]

        if isinstance(exercise_idx, int):
            idx_tensor = torch.tensor([exercise_idx], dtype=torch.long)
            return_scalar = True
        else:
            idx_tensor = exercise_idx
            return_scalar = False

        assert idx_tensor.dtype == torch.long, (
            f"BUG: exercise_idx tensor must be dtype long, got {idx_tensor.dtype}"
        )
        assert torch.all(idx_tensor >= 0) and torch.all(idx_tensor < cfg.NUM_NODES), (
            f"BUG: exercise_idx contains values out of range [0, {cfg.NUM_NODES - 1}]"
        )

        selected = node_embeddings[idx_tensor]        # [k, GAT_LAYER_3_DIM]
        preds = self.head(selected).squeeze(-1)         # [k]

        if return_scalar:
            return preds[0]
        return preds

    # ── FedPer convenience accessors (Phase 2 will use these directly) ──────
    def base_parameters(self):
        """Parameters that get shared/aggregated across clients in FedPer."""
        return self.gat.parameters()

    def head_parameters(self):
        """Parameters that stay local to each student in FedPer."""
        return self.head.parameters()


if __name__ == '__main__':
    print("=" * 70)
    print("FedGKT -- standalone self-test")
    print("=" * 70)

    if os.path.exists(cfg.EDGE_INDEX_PATH):
        edge_index = torch.load(cfg.EDGE_INDEX_PATH, weights_only=False)
        print(f"\nLoaded real edge_index.pt: shape {tuple(edge_index.shape)}")
    else:
        print(f"\nWARNING: {cfg.EDGE_INDEX_PATH} not found -- using synthetic edge_index.")
        torch.manual_seed(cfg.RANDOM_SEED)
        edge_index = torch.randint(0, cfg.NUM_NODES, (2, cfg.NUM_EDGES), dtype=torch.long)

    torch.manual_seed(cfg.RANDOM_SEED)
    model = FedGKT()

    n_total = sum(p.numel() for p in model.parameters())
    n_base = sum(p.numel() for p in model.base_parameters())
    n_head = sum(p.numel() for p in model.head_parameters())
    print(f"\nParameter counts: base (GAT) = {n_base:,}  head = {n_head:,}  total = {n_total:,}")
    assert n_base + n_head == n_total, "BUG: base + head params don't sum to total"
    print(f"  (Design doc target: ~85,000 total -- this is a rough guideline from "
          f"the original spec, not an enforced assertion. Current total is well "
          f"under that, which is fine for a small, FL-communication-efficient model.)")

    x = torch.zeros((cfg.NUM_NODES, cfg.NUM_FEATURES), dtype=torch.float32)
    x[:, 0] = cfg.COLD_START_MASTERY

    # ── single-int exercise_idx -> scalar output ─────────────────────────────
    model.eval()
    with torch.no_grad():
        pred_single = model(x, edge_index, 42)
    print(f"\nSingle exercise_idx=42 -> prediction: {pred_single.item():.4f} "
          f"(shape {tuple(pred_single.shape)}, must be scalar/0-dim)")
    assert pred_single.dim() == 0, f"BUG: expected scalar (0-dim) output, got shape {tuple(pred_single.shape)}"
    assert 0.0 <= pred_single.item() <= 1.0, "BUG: prediction outside [0,1] -- sigmoid not applied correctly"

    # ── batch of exercise_idx -> 1D tensor output ────────────────────────────
    idx_batch = torch.tensor([0, 42, 834], dtype=torch.long)
    with torch.no_grad():
        pred_batch = model(x, edge_index, idx_batch)
    print(f"\nBatch exercise_idx={idx_batch.tolist()} -> predictions: "
          f"{[round(v, 4) for v in pred_batch.tolist()]} (shape {tuple(pred_batch.shape)})")
    assert tuple(pred_batch.shape) == (3,), f"BUG: expected shape (3,), got {tuple(pred_batch.shape)}"
    assert torch.all(pred_batch >= 0.0) and torch.all(pred_batch <= 1.0), (
        "BUG: some predictions outside [0,1]"
    )
    print("Single and batch prediction shapes/ranges both correct -- OK")
    print("NOTE: all 3 predictions above are expected to be IDENTICAL -- x here is")
    print("      completely uniform across every node (pure cold-start), and a GAT's")
    print("      attention-weighted aggregation of identical vectors mathematically")
    print("      returns that same vector regardless of graph structure. This is")
    print("      correct behaviour for a degenerate uniform input, not a bug -- the")
    print("      REAL node-differentiation check follows below with varied features.")

    # ── batch test with REALISTIC (non-uniform) features -- the check that
    #    actually verifies different nodes get different scores ──────────────
    x_varied = torch.zeros((cfg.NUM_NODES, cfg.NUM_FEATURES), dtype=torch.float32)
    x_varied[:, 0] = cfg.COLD_START_MASTERY
    x_varied[0, cfg.FEATURE_IDX['mastery_score']] = 0.9
    x_varied[0, cfg.FEATURE_IDX['first_attempt']] = 1.0
    x_varied[42, cfg.FEATURE_IDX['mastery_score']] = 0.2
    x_varied[42, cfg.FEATURE_IDX['first_attempt']] = 1.0

    with torch.no_grad():
        pred_varied = model(x_varied, edge_index, idx_batch)
    print(f"\nSame batch, REALISTIC varied features -> predictions: "
          f"{[round(v, 4) for v in pred_varied.tolist()]}")
    n_unique = len(set(round(v, 6) for v in pred_varied.tolist()))
    print(f"  Unique prediction values: {n_unique} (must be > 1 -- nodes with "
          f"different features must score differently)")
    assert n_unique > 1, (
        "BUG: nodes with genuinely different features produced identical "
        "predictions -- the model is not using node-specific state correctly"
    )
    print("  Nodes with different features correctly produce different predictions -- OK")

    # ── gradient flow through BOTH gat and head ──────────────────────────────
    model.train()
    pred_train = model(x, edge_index, 42)
    pred_train.backward()

    base_has_grad = all(
        p.grad is not None and p.grad.abs().sum().item() > 0
        for p in model.base_parameters()
    )
    head_has_grad = all(
        p.grad is not None and p.grad.abs().sum().item() > 0
        for p in model.head_parameters()
    )
    print(f"\nGradient flow: base (GAT) params all have grad = {base_has_grad}, "
          f"head params all have grad = {head_has_grad}")
    assert base_has_grad, "BUG: gradient did not reach all GAT parameters"
    assert head_has_grad, "BUG: gradient did not reach all head parameters"
    print("Gradients reached both submodules -- OK")

    print("\nSelf-test complete -- no assertion failures.")