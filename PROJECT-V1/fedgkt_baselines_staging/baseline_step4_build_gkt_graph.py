"""
baseline_step4_build_gkt_graph.py

Builds the graph file pyKT's GKT baseline loads: FedGKT's own 978-edge
prerequisite graph, packaged exactly the way pyKT packages every graph it
builds for GKT. Runs LOCALLY, once, like Steps 1 and 3 -- the output file
then travels to Colab inside the zip.

    input : graph_for_gkt/prerequisite_edges.csv        (from_idx,to_idx)
    check : ../fedgkt/data/processed/edge_index.pt      (read-only)
    output: pykt_sequences/gkt_graph_prerequisite.npz   (key 'matrix')

WHY THE GRAPH IS BUILT THIS WAY  (verified against pyKT 0.0.38 source)
----------------------------------------------------------------------
pyKT has two built-in GKT graphs, in pykt/models/gkt_utils.py. Both are
row-stochastic, and both force a zero diagonal:

    build_transition_graph():
        ...
        np.fill_diagonal(graph, 0)
        # row normalization
        rowsum = np.array(graph.sum(1))
        def inv(x):
            if x == 0:
                return x
            return 1. / x
        ...
        graph = r_mat_inv.dot(graph)

    build_dense_graph():
        graph = 1. / (concept_num - 1) * np.ones(...)
        np.fill_diagonal(graph, 0)

So this file does the same to FedGKT's edges:
  - graph[from_idx, to_idx] = 1        (prerequisite -> dependent, FedGKT's
                                        own direction, unchanged)
  - diagonal zero                      (pyKT's convention; FedGKT's graph has
                                        no self-loops to begin with)
  - row-normalised with pyKT's exact rule, INCLUDING its handling of empty
    rows: inv(0) returns 0, so a concept with no outgoing edges keeps an
    all-zero row. No division by zero, no NaN.

NO SELF-LOOPS ARE ADDED. An earlier recommendation proposed self-loops on
the 59 isolated concepts to avoid division by zero during normalisation.
Checking pyKT's code showed that reasoning was wrong on three counts:
pyKT never divides by zero (inv(0) = 0); pyKT deliberately zeroes the
diagonal in both of its graphs; and normalisation leaves 293 empty rows
here, not 59 -- every concept that is never a prerequisite of anything,
of which the 59 isolated ones are a subset. Empty rows are routine in
pyKT's own transition graphs, so GKT is built to handle them.

HOW GKT READS THE GRAPH (pykt/models/gkt.py, lines 139-140)
------------------------------------------------------------
    adj         = self.graph[masked_qt.long(), :]   # the answered concept's ROW
    reverse_adj = self.graph[:, masked_qt.long()]   # ...and its COLUMN

So GKT sees both a concept's dependents (row) and its prerequisites
(column). Direction decides which side carries the row-normalised
weights, not whether a neighbour is seen at all.

FILE NAME
---------
The file is named gkt_graph_prerequisite.npz, and the driver passes
graph_type='prerequisite'. pyKT's init_model() loads
gkt_graph_{graph_type}.npz from the data folder if it exists, and
otherwise BUILDS a graph itself. pyKT's own default graph_type is
'transition' -- so naming ours after that would mean a missing file
silently produces pyKT's data-derived graph instead of FedGKT's, and GKT
would train on the wrong graph without any error. With a name pyKT has no
builder for, a missing file fails loudly instead.

FOR THE PAPER: both FedGKT and GKT use the identical edge set; each
applies its own standard convention to it. FedGKT's GAT layers add
self-loops (PyG's GATConv default); GKT, following pyKT, does not.

USAGE (locally, from fedgkt_baselines_staging/)
-----------------------------------------------
    python baseline_step4_build_gkt_graph.py
"""

import hashlib
import os

import numpy as np
import pandas as pd

NUM_CONCEPTS = 835
EXPECTED_EDGES = 978
GRAPH_TYPE = 'prerequisite'

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
EDGES_CSV = os.path.join(_THIS_DIR, 'graph_for_gkt', 'prerequisite_edges.csv')
OUT_PATH = os.path.join(_THIS_DIR, 'pykt_sequences', f'gkt_graph_{GRAPH_TYPE}.npz')


def find_fedgkt_edge_index():
    """Walks upward from this file to find fedgkt/data/processed/edge_index.pt."""
    here = _THIS_DIR
    for _ in range(4):
        candidate = os.path.join(here, 'fedgkt', 'data', 'processed', 'edge_index.pt')
        if os.path.exists(candidate):
            return candidate
        here = os.path.dirname(here)
    return None


def pykt_row_normalize(graph):
    """
    Copied line-for-line from pyKT 0.0.38's build_transition_graph(),
    minus the final torch conversion. Kept verbatim on purpose, so this is
    pyKT's own rule rather than a re-derivation of it.
    """
    graph = np.array(graph, dtype=np.float64)
    np.fill_diagonal(graph, 0)
    # row normalization
    rowsum = np.array(graph.sum(1))

    def inv(x):
        if x == 0:
            return x
        return 1. / x

    inv_func = np.vectorize(inv)
    r_inv = inv_func(rowsum).flatten()
    r_mat_inv = np.diag(r_inv)
    graph = r_mat_inv.dot(graph)
    return graph


def independent_row_normalize(graph):
    """A second, independently written version, used only to cross-check."""
    graph = np.array(graph, dtype=np.float64)
    np.fill_diagonal(graph, 0)
    out = np.zeros_like(graph)
    sums = graph.sum(axis=1)
    nonzero = sums > 0
    out[nonzero] = graph[nonzero] / sums[nonzero][:, None]
    return out


def graph_sha256(matrix):
    return hashlib.sha256(np.ascontiguousarray(matrix, dtype=np.float32).tobytes()).hexdigest()


def main():
    print("=" * 70)
    print("baseline_step4_build_gkt_graph.py")
    print("=" * 70)

    # ── 1. read and validate the edge list ────────────────────────────
    assert os.path.exists(EDGES_CSV), f"Edge list not found: {EDGES_CSV}"
    df = pd.read_csv(EDGES_CSV)
    assert list(df.columns) == ['from_idx', 'to_idx'], (
        f"Expected columns ['from_idx', 'to_idx'], got {list(df.columns)}"
    )
    src = df['from_idx'].to_numpy(dtype=np.int64)
    dst = df['to_idx'].to_numpy(dtype=np.int64)

    print(f"\nEdge list: {EDGES_CSV}")
    print(f"  edges: {len(df)}")
    assert len(df) == EXPECTED_EDGES, f"Expected {EXPECTED_EDGES} edges, got {len(df)}"
    assert src.min() >= 0 and dst.min() >= 0, "Negative concept index in edge list"
    assert src.max() < NUM_CONCEPTS and dst.max() < NUM_CONCEPTS, (
        f"Concept index >= {NUM_CONCEPTS} in edge list"
    )
    assert not np.any(src == dst), "Edge list contains self-loops"
    pairs = set(zip(src.tolist(), dst.tolist()))
    assert len(pairs) == len(df), f"Edge list contains {len(df) - len(pairs)} duplicate edges"
    print("  indices in 0..834, no self-loops, no duplicates -- OK")

    # ── 2. cross-check against FedGKT's own graph ─────────────────────
    ei_path = find_fedgkt_edge_index()
    assert ei_path, (
        "Could not find fedgkt/data/processed/edge_index.pt above this folder. "
        "This check proves GKT gets exactly FedGKT's graph, so it runs locally, "
        "where fedgkt/ exists."
    )
    import torch
    ei = torch.load(ei_path, weights_only=False).numpy()
    fedgkt_pairs = set(zip(ei[0].tolist(), ei[1].tolist()))
    print(f"\nFedGKT's graph: {ei_path}")
    print(f"  edges: {ei.shape[1]}")
    assert fedgkt_pairs == pairs, (
        f"Edge list does NOT match FedGKT's edge_index.pt.\n"
        f"  only in CSV:    {len(pairs - fedgkt_pairs)}\n"
        f"  only in FedGKT: {len(fedgkt_pairs - pairs)}\n"
        f"GKT must use exactly FedGKT's graph -- stopping."
    )
    print("  identical edge set, same direction -- OK")

    # ── 3. build and normalise ────────────────────────────────────────
    adjacency = np.zeros((NUM_CONCEPTS, NUM_CONCEPTS), dtype=np.float64)
    adjacency[src, dst] = 1.0

    graph = pykt_row_normalize(adjacency)
    check = independent_row_normalize(adjacency)
    max_diff = float(np.abs(graph - check).max())
    assert max_diff < 1e-12, f"pyKT's rule and the independent version disagree by {max_diff}"

    # ── 4. verify the result ──────────────────────────────────────────
    out_deg = (adjacency > 0).sum(axis=1)
    in_deg = (adjacency > 0).sum(axis=0)
    row_sums = graph.sum(axis=1)
    n_empty = int((row_sums == 0).sum())

    assert graph.shape == (NUM_CONCEPTS, NUM_CONCEPTS)
    assert np.all(np.diag(graph) == 0), "Diagonal is not zero"
    assert int((graph > 0).sum()) == EXPECTED_EDGES, "Nonzero count changed during normalisation"
    assert np.all(np.isfinite(graph)), "Graph contains NaN or Inf"
    assert np.allclose(row_sums[out_deg > 0], 1.0), "A non-empty row does not sum to 1"
    assert n_empty == int((out_deg == 0).sum()), "Empty rows don't match zero out-degree"

    print("\nNormalised graph:")
    print(f"  shape               : {graph.shape}")
    print(f"  nonzero entries     : {int((graph > 0).sum())}")
    print(f"  diagonal            : all zero")
    print(f"  non-empty rows      : {NUM_CONCEPTS - n_empty}, each summing to 1")
    print(f"  empty rows          : {n_empty}   (concepts that are never a prerequisite)")
    print(f"  fully isolated      : {int(((out_deg == 0) & (in_deg == 0)).sum())}")
    print(f"  pyKT's rule vs independent version: max difference {max_diff:.1e}")

    # ── 5. save, then reload EXACTLY the way pyKT's init_model does ───
    matrix = graph.astype(np.float32)
    np.savez(OUT_PATH, matrix=matrix)

    import torch
    reloaded = torch.tensor(np.load(OUT_PATH, allow_pickle=True)['matrix']).float()
    assert torch.equal(reloaded, torch.from_numpy(matrix)), "Reloaded graph differs from saved graph"

    print(f"\nWritten : {OUT_PATH}")
    print(f"  reloaded the way pyKT's init_model() loads it -- identical")
    print(f"  sha256  : {graph_sha256(matrix)}")
    print("  (run_gkt.py recomputes this and records it, so a changed graph is caught)")

    print("\nDone -- no assertion failures.")


if __name__ == '__main__':
    main()