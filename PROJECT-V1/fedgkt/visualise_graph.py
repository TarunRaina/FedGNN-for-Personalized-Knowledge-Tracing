import os
import torch
import networkx as nx
import matplotlib.pyplot as plt

from src.utils import config as cfg
from src.models.fedgkt import FedGKT
from src.data.pkg_batched import BatchedPersonalKnowledgeGraph

# --- Paths ---
EDGE_INDEX_PATH = os.path.join("data", "processed", "edge_index.pt")
CHECKPOINT_PATH = os.path.join("checkpoints_gpu_batched", "best_model_batched.pt")
STUDENT_PKG_PATH = os.path.join("data", "processed", "pkgs", "pkg_630.pt")

# Output dir for high-res saves (so you can zoom in beyond what plt.show() gives you)
OUT_DIR = "viz_output"
os.makedirs(OUT_DIR, exist_ok=True)


def get_layout(G):
    """
    Computes a layout suited for THIS graph's actual shape: a single large
    DAG (~750 nodes in one component) with 38 root nodes, max chain depth
    ~65, and modest branching (most nodes have out-degree 0-2, a handful
    of hub nodes reach out-degree ~14). It's a long, narrow, deep DAG --
    not a wide bushy tree and not an undirected blob.

    That ruled out the two "generic" options:
    - spring_layout / sfdp: force-directed, ignore hierarchy entirely,
      collapse the graph into a hairball or scatter unrelated fragments.
    - Graphviz `dot`: built for wide, shallow hierarchies. On a graph this
      deep and narrow it still tangles the dense mid-graph confluence
      where multiple long chains merge.

    Instead this uses a layered ("Sugiyama-style") layout tailored to the
    graph's real structure:
      1. Assign every node a depth = longest path length from any root.
         This naturally produces the same left-to-right "flow" you'd
         expect from a prerequisite chain.
      2. Within each depth level, repeatedly reorder nodes by the average
         position of their predecessors/successors (alternating forward
         and backward passes). This is the standard technique for
         minimizing edge crossings in layered graph drawing -- it's what
         pulls apart the tangled mid-graph hub into legible strands
         instead of a smear.
    Runs in well under a second at this graph's size (a few hundred to
    ~1000 nodes) -- no external layout engine or binary required.
    """
    depth = {}
    for n in nx.topological_sort(G):
        preds = list(G.predecessors(n))
        depth[n] = 0 if not preds else max(depth[p] for p in preds) + 1

    by_depth = {}
    for n, d in depth.items():
        by_depth.setdefault(d, []).append(n)

    order = {d: sorted(ns) for d, ns in by_depth.items()}
    y_pos = {}
    for d in sorted(order.keys()):
        for i, n in enumerate(order[d]):
            y_pos[n] = float(i)

    # Alternating forward/backward barycenter passes to reduce edge crossings.
    for _ in range(20):
        for d in sorted(order.keys()):
            if d == 0:
                continue
            def bary_pred(n):
                preds = list(G.predecessors(n))
                return sum(y_pos[p] for p in preds) / len(preds) if preds else y_pos[n]
            ns_sorted = sorted(order[d], key=bary_pred)
            order[d] = ns_sorted
            for i, n in enumerate(ns_sorted):
                y_pos[n] = float(i) - len(ns_sorted) / 2
        for d in sorted(order.keys(), reverse=True):
            def bary_succ(n):
                succs = list(G.successors(n))
                return sum(y_pos[s] for s in succs) / len(succs) if succs else y_pos[n]
            ns_sorted = sorted(order[d], key=bary_succ)
            order[d] = ns_sorted
            for i, n in enumerate(ns_sorted):
                y_pos[n] = float(i) - len(ns_sorted) / 2

    pos = {}
    for d, ns in order.items():
        for i, n in enumerate(ns):
            pos[n] = (d * 1.4, y_pos[n])  # slight x-stretch keeps edges legible

    print(f"Layered layout: {len(by_depth)} depth levels, "
          f"widest level has {max(len(v) for v in by_depth.values())} nodes.")
    return pos, depth


def _figsize_for_layout(pos, base_height=24, min_width=30, max_width=160):
    """
    This layout is naturally very wide (one axis per depth level -- up to
    ~65 levels for this graph) and much shorter vertically. Sizing the
    figure to the real aspect ratio keeps nodes from being squeezed
    together, capped so the output file doesn't balloon indefinitely.
    """
    xs = [p[0] for p in pos.values()]
    ys = [p[1] for p in pos.values()]
    width = max(xs) - min(xs) if xs else 1
    height = max(ys) - min(ys) if ys else 1
    aspect = width / max(height, 1)
    fig_w = min(max(base_height * aspect / 2.2, min_width), max_width)
    return fig_w, base_height


def plot_global_graph(edge_index):
    """Plots the static concept prerequisite map with a large-graph-friendly layout."""
    G = nx.DiGraph()
    edges = edge_index.t().tolist()
    G.add_edges_from(edges)

    pos, depth = get_layout(G)
    fig_w, fig_h = _figsize_for_layout(pos)
    print(f"Rendering at figsize {fig_w:.0f} x {fig_h} -- this graph is naturally long "
          f"and narrow (many depth levels), so the saved PNG is meant to be zoomed "
          f"into, not viewed all at once.")

    plt.figure(figsize=(fig_w, fig_h), dpi=150)

    # Color by depth (root=dark purple -> deep leaf=yellow) so the prerequisite
    # "flow" through the graph is visible at a glance, not just the topology.
    node_colors = [depth[n] for n in G.nodes()]

    nx.draw_networkx_nodes(G, pos, node_color=node_colors, cmap=plt.cm.viridis,
                            node_size=55, alpha=0.9, linewidths=0)
    nx.draw_networkx_edges(G, pos, edge_color='dimgray', arrows=True,
                            arrowsize=7, width=0.6, alpha=0.6)
    # Labels off by default -- at 800+ nodes text is the main source of clutter.
    # Set SHOW_LABELS = True below if you want them (best combined with zooming
    # into the saved high-res PNG rather than the plt.show() window).
    SHOW_LABELS = False
    if SHOW_LABELS:
        nx.draw_networkx_labels(G, pos, font_size=4)

    plt.title("1. Global Concept Knowledge Graph", fontsize=20, fontweight='bold')
    plt.figtext(0.5, 0.01, "Color = prerequisite depth from root (dark = foundational, yellow = advanced)",
                ha='center', fontsize=11, style='italic')
    plt.axis('off')
    plt.tight_layout()

    out_path = os.path.join(OUT_DIR, "global_graph.png")
    # dpi=150 here (not higher) because figsize is already scaled to the layout's
    # true aspect ratio -- at fig_w up to 160in, a higher dpi produces a
    # multi-hundred-MB file for no real gain. Open the PNG in an image viewer
    # and zoom in; it stays crisp because of the large native pixel dimensions.
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"Saved high-res global graph to {out_path} (open in an image viewer and zoom in)")

    plt.show()
    return G, pos, depth


def plot_student_graph(G, pos, depth, model, student_pkg_path, edge_index):
    """Rebuilds the student's PKG state and plots the model's predicted mastery."""
    device = torch.device('cpu')
    student_data = torch.load(student_pkg_path, weights_only=False)

    ex_seq = student_data['exercise_idx']
    c_seq = student_data['correct']
    t_seq = student_data['time_done']
    seq_len = ex_seq.shape[0]

    # 1. Rebuild the student's final PKG state by simulating their interaction history
    pkg = BatchedPersonalKnowledgeGraph(batch_size=1, device=device)
    active_mask = torch.tensor([True], dtype=torch.bool, device=device)

    for step in range(seq_len):
        step_ex = ex_seq[step].unsqueeze(0)
        step_c = c_seq[step].unsqueeze(0)
        step_t = t_seq[step].unsqueeze(0)

        pkg.refresh_time_decay(step_t, active_mask)
        pkg.update(step_ex, step_c, step_t, active_mask)

    # Extract the fully updated feature matrix [NUM_NODES, NUM_FEATURES]
    x_final = pkg.get_x().reshape(cfg.NUM_NODES, cfg.NUM_FEATURES)

    # 2. Run the model forward pass to get mastery predictions for ALL concepts
    model.eval()
    with torch.no_grad():
        all_nodes_idx = torch.arange(cfg.NUM_NODES, dtype=torch.long, device=device)
        mastery_probs = model(x_final, edge_index, all_nodes_idx).numpy()

    node_colors = [mastery_probs[int(node)] if int(node) < len(mastery_probs) else 0.0 for node in G.nodes()]

    attempted_nodes = set(ex_seq.tolist())
    edge_colors = ['black' if n in attempted_nodes else 'none' for n in G.nodes()]
    edge_widths = [1.2 if n in attempted_nodes else 0.0 for n in G.nodes()]

    fig_w, fig_h = _figsize_for_layout(pos)
    fig = plt.figure(figsize=(fig_w, fig_h), dpi=150)
    ax = plt.gca()

    nodes = nx.draw_networkx_nodes(G, pos, node_color=node_colors, cmap=plt.cm.RdYlGn,
                                    node_size=50, vmin=0, vmax=1,
                                    edgecolors=edge_colors, linewidths=edge_widths,
                                    ax=ax)

    nx.draw_networkx_edges(G, pos, edge_color='dimgray', arrows=True,
                            arrowsize=7, width=0.6, alpha=0.5, ax=ax)

    SHOW_LABELS = False
    if SHOW_LABELS:
        nx.draw_networkx_labels(G, pos, font_size=4, ax=ax)

    cbar = plt.colorbar(nodes, ax=ax, fraction=0.03, pad=0.01)
    cbar.set_label("Predicted Mastery (Red=Low, Green=High)", fontsize=12)

    plt.title(f"2. FedGKT Predicted Mastery Graph ({os.path.basename(student_pkg_path)})",
              fontsize=20, fontweight='bold')
    plt.figtext(0.5, 0.01, "Layout position = prerequisite depth from root; color = predicted mastery",
                ha='center', fontsize=11, style='italic')
    plt.axis('off')
    plt.tight_layout()

    out_path = os.path.join(OUT_DIR, f"student_graph_{os.path.basename(student_pkg_path)}.png")
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"Saved high-res student mastery graph to {out_path} (open in an image viewer and zoom in)")

    plt.show()


if __name__ == "__main__":
    print("Loading global edges...")
    edge_index = torch.load(EDGE_INDEX_PATH, weights_only=False)

    print("Plotting Global Graph (Close the window to proceed to the Student Graph)...")
    G, pos, depth = plot_global_graph(edge_index)

    print("Loading trained model weights...")
    model = FedGKT()
    model.load_state_dict(torch.load(CHECKPOINT_PATH, weights_only=True, map_location='cpu'))

    print(f"Rebuilding PKG state and plotting Student Graph for {os.path.basename(STUDENT_PKG_PATH)}...")
    plot_student_graph(G, pos, depth, model, STUDENT_PKG_PATH, edge_index)