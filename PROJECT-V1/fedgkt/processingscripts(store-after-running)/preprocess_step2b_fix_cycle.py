import pandas as pd
import json
import networkx as nx
import os

# ── paths ─────────────────────────────────────────────────────────────────────
PROCESSED_DIR    = os.path.join('data', 'processed')
VOCAB_FILE       = os.path.join(PROCESSED_DIR, 'exercise_vocab.json')
EDGES_IDX        = os.path.join(PROCESSED_DIR, 'prerequisite_edges.csv')
EDGES_NAMED      = os.path.join(PROCESSED_DIR, 'prerequisite_edges_with_names.csv')

# ── load vocab ────────────────────────────────────────────────────────────────
with open(VOCAB_FILE, 'r', encoding='utf-8') as f:
    vocab = json.load(f)

name_to_idx = vocab['name_to_idx']
idx_to_name = vocab['idx_to_name']   # keys are strings in JSON
num_exercises = vocab['num_exercises']

# ── load existing edge files ───────────────────────────────────────────────────
edges_idx   = pd.read_csv(EDGES_IDX)
edges_named = pd.read_csv(EDGES_NAMED)

print(f"Edges before fix: {len(edges_idx)}")

# ── define the erroneous edge to remove ───────────────────────────────────────
# Reason: the raw Junyi data contains a cycle:
#   simplifying_radicals (688)
#     -> adding_and_subtracting_radicals (13)
#     -> radical_multiplication_and_division (582)
#     -> simplifying_radicals (688)   <-- this third edge is wrong
#
# Logically: you learn to simplify radicals first, then add/subtract them,
# then multiply/divide them. radical_multiplication_and_division cannot be
# a prerequisite of simplifying_radicals -- that is backwards.
# The correct direction is 688 -> 13 -> 582. The edge 582 -> 688 is removed.

BAD_FROM_NAME = 'radical_multiplication_and_division'
BAD_TO_NAME   = 'simplifying_radicals'

# Confirm the indices match the names using the vocab
bad_from_idx = name_to_idx[BAD_FROM_NAME]
bad_to_idx   = name_to_idx[BAD_TO_NAME]

print(f"\nEdge to remove:")
print(f"  from_idx={bad_from_idx} ({BAD_FROM_NAME})")
print(f"  to_idx  ={bad_to_idx} ({BAD_TO_NAME})")

# Safety check: confirm this edge actually exists before we try to remove it
exists_in_idx   = len(edges_idx[
    (edges_idx['from_idx'] == bad_from_idx) &
    (edges_idx['to_idx']   == bad_to_idx)
]) > 0

exists_in_named = len(edges_named[
    (edges_named['from_idx'] == bad_from_idx) &
    (edges_named['to_idx']   == bad_to_idx)
]) > 0

print(f"\nEdge found in prerequisite_edges.csv:            {exists_in_idx}")
print(f"Edge found in prerequisite_edges_with_names.csv: {exists_in_named}")

if not exists_in_idx or not exists_in_named:
    print("\nERROR: Edge not found in one or both files. Stopping -- nothing changed.")
    exit(1)

# ── remove the bad edge from both files ───────────────────────────────────────
edges_idx_fixed = edges_idx[
    ~((edges_idx['from_idx'] == bad_from_idx) &
      (edges_idx['to_idx']   == bad_to_idx))
].reset_index(drop=True)

edges_named_fixed = edges_named[
    ~((edges_named['from_idx'] == bad_from_idx) &
      (edges_named['to_idx']   == bad_to_idx))
].reset_index(drop=True)

print(f"\nEdges after fix: {len(edges_idx_fixed)}")
print(f"Edges removed:   {len(edges_idx) - len(edges_idx_fixed)}  (expected: 1)")

# ── rebuild the graph and validate ───────────────────────────────────────────
G = nx.DiGraph()
G.add_nodes_from(range(num_exercises))
for _, row in edges_idx_fixed.iterrows():
    G.add_edge(int(row['from_idx']), int(row['to_idx']))

is_dag = nx.is_directed_acyclic_graph(G)
print(f"\nIs valid DAG after fix: {is_dag}")

if not is_dag:
    # If we still have cycles, something is wrong -- stop and report
    cycles = list(nx.simple_cycles(G))
    print(f"ERROR: Still {len(cycles)} cycle(s) remaining. Stopping -- nothing saved.")
    for cycle in cycles:
        names = [idx_to_name[str(i)] for i in cycle]
        print(f"  {' -> '.join(names)}")
    exit(1)

# ── verify the two edges we expect to remain are still there ──────────────────
# The other two edges in the original cycle must still exist
check_pairs = [
    ('simplifying_radicals',          'adding_and_subtracting_radicals'),
    ('adding_and_subtracting_radicals','radical_multiplication_and_division'),
]

print("\nChecking the two correct edges in the cycle are preserved:")
for from_name, to_name in check_pairs:
    f_idx = name_to_idx[from_name]
    t_idx = name_to_idx[to_name]
    present = len(edges_idx_fixed[
        (edges_idx_fixed['from_idx'] == f_idx) &
        (edges_idx_fixed['to_idx']   == t_idx)
    ]) > 0
    print(f"  {from_name} -> {to_name}: {'OK' if present else 'MISSING -- ERROR'}")

# ── final graph stats ─────────────────────────────────────────────────────────
print(f"\n--- Final Graph Stats ---")
print(f"Nodes:                       {G.number_of_nodes()}")
print(f"Edges:                       {G.number_of_edges()}")
print(f"Is valid DAG:                True")
print(f"Weakly connected components: {nx.number_weakly_connected_components(G)}")
root_nodes = [n for n in G.nodes() if G.in_degree(n) == 0]
isolated   = [n for n in G.nodes() if G.degree(n) == 0]
print(f"Root nodes (no prereqs):     {len(root_nodes)}")
print(f"Isolated nodes (no edges):   {len(isolated)}")

# ── save both fixed files (overwrite originals) ───────────────────────────────
edges_idx_fixed[['from_idx', 'to_idx']].to_csv(EDGES_IDX, index=False)
edges_named_fixed[['from_idx', 'to_idx', 'from_name', 'to_name']].to_csv(EDGES_NAMED, index=False)

print(f"\nOverwritten: {EDGES_IDX}")
print(f"Overwritten: {EDGES_NAMED}")
print("Step 2b complete. Prerequisite graph is now a clean DAG.")