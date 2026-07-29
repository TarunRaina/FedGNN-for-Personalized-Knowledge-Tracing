"""
src/models/gat.py

Plain 3-layer Graph Attention Network (GAT). Pure architecture only -- no
personalisation head, no FL-specific logic. This class exists purely to
turn a student's node feature matrix into node embeddings; everything about
turning those embeddings into a prediction lives in fedgkt.py.

Kept as a standalone module (not merged into fedgkt.py) deliberately: Phase
2's FedPer design shares these GAT layers across all clients while keeping
the output head local to each student. Having the GAT as its own class now
means that split requires no rewrite later -- just swapping which submodule
gets aggregated in the FL strategy.

Architecture (locked, from original design doc):
  Layer 1: 4 heads x 32 dim, concatenated -> 128-dim output
  Layer 2: 4 heads x 32 dim, concatenated -> 128-dim output
  Layer 3: 1 head  x 64 dim -> 64-dim output (final node embedding)
  ELU activation + dropout between layers 1->2 and 2->3 (standard GAT paper
  convention, Velickovic et al. ICLR 2018). No activation after the final
  layer -- raw embeddings are returned for the personal head to consume.
"""

import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))       # src/models
_SRC_DIR = os.path.dirname(_THIS_DIR)                          # src
_PROJECT_ROOT = os.path.dirname(_SRC_DIR)                      # fedgkt/
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.utils import config as cfg


class GAT(nn.Module):
    """
    Input:  x [NUM_NODES, NUM_FEATURES], edge_index [2, NUM_EDGES]
    Output: node embeddings [NUM_NODES, GAT_LAYER_3_DIM]
    """

    def __init__(self):
        super().__init__()

        self.conv1 = GATConv(
            in_channels=cfg.NUM_FEATURES,
            out_channels=cfg.GAT_LAYER_1_DIM,
            heads=cfg.GAT_LAYER_1_HEADS,
            concat=True,
            dropout=cfg.DROPOUT,
        )
        layer1_out_dim = cfg.GAT_LAYER_1_DIM * cfg.GAT_LAYER_1_HEADS  # 128

        self.conv2 = GATConv(
            in_channels=layer1_out_dim,
            out_channels=cfg.GAT_LAYER_2_DIM,
            heads=cfg.GAT_LAYER_2_HEADS,
            concat=True,
            dropout=cfg.DROPOUT,
        )
        layer2_out_dim = cfg.GAT_LAYER_2_DIM * cfg.GAT_LAYER_2_HEADS  # 128

        self.conv3 = GATConv(
            in_channels=layer2_out_dim,
            out_channels=cfg.GAT_LAYER_3_DIM,
            heads=cfg.GAT_LAYER_3_HEADS,
            concat=False,  # heads=1, so concat vs mean is moot, but explicit
            dropout=cfg.DROPOUT,
        )

        self.dropout_p = cfg.DROPOUT
        self.output_dim = cfg.GAT_LAYER_3_DIM

        self._layer1_out_dim = layer1_out_dim
        self._layer2_out_dim = layer2_out_dim

    def forward(self, x, edge_index):
        h = self.conv1(x, edge_index)
        h = F.elu(h)
        h = F.dropout(h, p=self.dropout_p, training=self.training)

        h = self.conv2(h, edge_index)
        h = F.elu(h)
        h = F.dropout(h, p=self.dropout_p, training=self.training)

        h = self.conv3(h, edge_index)
        # no activation on the final layer -- raw embeddings passed onward

        return h


if __name__ == '__main__':
    print("=" * 70)
    print("GAT -- standalone self-test")
    print("=" * 70)

    # ── load REAL edge_index.pt if available, else fall back to synthetic ──
    if os.path.exists(cfg.EDGE_INDEX_PATH):
        edge_index = torch.load(cfg.EDGE_INDEX_PATH, weights_only=False)
        print(f"\nLoaded real edge_index.pt: shape {tuple(edge_index.shape)}")
    else:
        print(f"\nWARNING: {cfg.EDGE_INDEX_PATH} not found -- using synthetic "
              f"edge_index for this self-test instead. Run preprocessing "
              f"Step 5 first if you want to test against the real graph.")
        torch.manual_seed(cfg.RANDOM_SEED)
        edge_index = torch.randint(0, cfg.NUM_NODES, (2, cfg.NUM_EDGES), dtype=torch.long)

    assert edge_index.shape[0] == 2, f"BUG: edge_index should be [2, E], got {tuple(edge_index.shape)}"
    assert edge_index.dtype == torch.long, f"BUG: edge_index dtype should be long, got {edge_index.dtype}"

    torch.manual_seed(cfg.RANDOM_SEED)
    model = GAT()

    n_params = sum(p.numel() for p in model.parameters())
    print(f"\nGAT parameter count: {n_params:,}")
    print(f"  (Design doc target for GAT + personal head combined: ~85,000. "
          f"The head is added separately in fedgkt.py, so this number alone "
          f"should be somewhat less than 85,000.)")

    # ── forward pass with a cold-start-shaped feature matrix ────────────────
    x = torch.zeros((cfg.NUM_NODES, cfg.NUM_FEATURES), dtype=torch.float32)
    x[:, 0] = 0.5  # mastery_score cold-start value, matches PersonalKnowledgeGraph

    model.eval()  # disable dropout for a deterministic shape/sanity check
    with torch.no_grad():
        out = model(x, edge_index)

    print(f"\nInput x shape:  {tuple(x.shape)}")
    print(f"Output shape:   {tuple(out.shape)}")
    expected_shape = (cfg.NUM_NODES, cfg.GAT_LAYER_3_DIM)
    assert tuple(out.shape) == expected_shape, (
        f"BUG: expected output shape {expected_shape}, got {tuple(out.shape)}"
    )
    print(f"Shape check passed: matches expected {expected_shape}")

    assert not torch.isnan(out).any(), "BUG: NaN values in GAT output"
    assert not torch.isinf(out).any(), "BUG: Inf values in GAT output"
    print("No NaN/Inf values in output -- OK")

    # ── gradient flow check ──────────────────────────────────────────────────
    model.train()
    x_grad_test = x.clone().requires_grad_(False)  # x itself isn't learned, only model weights are
    out_train = model(x_grad_test, edge_index)
    loss = out_train.sum()
    loss.backward()

    n_params_with_grad = sum(
        1 for p in model.parameters() if p.grad is not None and p.grad.abs().sum().item() > 0
    )
    n_total_params_tensors = sum(1 for p in model.parameters())
    print(f"\nGradient flow check: {n_params_with_grad}/{n_total_params_tensors} "
          f"parameter tensors received nonzero gradients after backward()")
    assert n_params_with_grad == n_total_params_tensors, (
        "BUG: some parameters received no gradient -- a layer may be disconnected"
    )
    print("All parameter tensors received gradients -- OK")

    # ── two different students (different x) should give different embeddings ──
    model.eval()
    x_student_a = torch.zeros((cfg.NUM_NODES, cfg.NUM_FEATURES), dtype=torch.float32)
    x_student_a[:, 0] = 0.5
    x_student_a[10, 0] = 0.9  # student A has high mastery on node 10

    x_student_b = torch.zeros((cfg.NUM_NODES, cfg.NUM_FEATURES), dtype=torch.float32)
    x_student_b[:, 0] = 0.5
    x_student_b[10, 0] = 0.1  # student B has low mastery on node 10

    with torch.no_grad():
        out_a = model(x_student_a, edge_index)
        out_b = model(x_student_b, edge_index)

    diff = (out_a - out_b).abs().sum().item()
    print(f"\nDifferent-feature sanity check: two students with different "
          f"mastery on node 10 produce embeddings differing by total abs "
          f"diff = {diff:.4f} (must be > 0)")
    assert diff > 0, "BUG: different input features produced identical output -- model is not using x"
    print("Output correctly differs based on input features -- OK")

    print("\nSelf-test complete -- no assertion failures.")