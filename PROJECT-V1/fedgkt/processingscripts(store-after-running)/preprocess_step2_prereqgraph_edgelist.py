import pandas as pd
import json
import os
import networkx as nx

# ── paths ─────────────────────────────────────────────────────────────────────
RAW_DIR       = os.path.join('data', 'raw')
PROCESSED_DIR = os.path.join('data', 'processed')

EXERCISE_TABLE   = os.path.join(RAW_DIR, 'junyi_Exercise_table.csv')
VOCAB_FILE       = os.path.join(PROCESSED_DIR, 'exercise_vocab.json')
OUTPUT_EDGES     = os.path.join(PROCESSED_DIR, 'prerequisite_edges.csv')
OUTPUT_EDGES_NM  = os.path.join(PROCESSED_DIR, 'prerequisite_edges_with_names.csv')

# ── load vocab ────────────────────────────────────────────────────────────────
print("Loading vocab...")
with open(VOCAB_FILE, 'r', encoding='utf-8') as f:
    vocab = json.load(f)

name_to_idx   = vocab['name_to_idx']
idx_to_name   = vocab['idx_to_name']  # keys are strings in JSON
num_exercises = vocab['num_exercises']

print(f"Vocab: {num_exercises} exercises, indices 0 to {num_exercises - 1}")

# ── load exercise table ───────────────────────────────────────────────────────
# Must deduplicate exactly the same way as step 1,
# so we stay consistent with the vocab that was built there.
print("\nLoading exercise table...")
ex = pd.read_csv(EXERCISE_TABLE, encoding='utf-8')
ex_deduped = (ex.sort_values('name')
                .drop_duplicates(subset='name', keep='first')
                .reset_index(drop=True))

print(f"Rows loaded: {len(ex)} -> {len(ex_deduped)} after dedup (consistent with step 1)")

# ── build edge list ───────────────────────────────────────────────────────────
# Direction: prerequisite ──> exercise
# i.e. from_idx must be learned before to_idx
print("\nBuilding prerequisite edges...")

edges      = []   # valid edges
skipped    = []   # names not found in vocab (should be 0)
self_loops = []   # exercise listed as its own prereq (should be 0)

for _, row in ex_deduped.iterrows():

    # root node — no prerequisites, no edges to add
    if pd.isna(row['prerequisites']):
        continue

    child_name = row['name']

    # defensive check — child must be in vocab
    if child_name not in name_to_idx:
        skipped.append(f"child not in vocab: {child_name}")
        continue

    child_idx = name_to_idx[child_name]

    # split on comma to handle multiple prerequisites
    raw_prereqs = str(row['prerequisites']).split(',')
    prereq_names = [p.strip() for p in raw_prereqs if p.strip()]

    for prereq_name in prereq_names:

        # defensive check — prereq must be in vocab
        if prereq_name not in name_to_idx:
            skipped.append(f"prereq not in vocab: '{prereq_name}' (needed by '{child_name}')")
            continue

        prereq_idx = name_to_idx[prereq_name]

        # self-loop check — exercise cannot be its own prerequisite
        if prereq_idx == child_idx:
            self_loops.append(f"{child_name} (idx {child_idx}) lists itself as prereq")
            continue

        edges.append({
            'from_idx':  prereq_idx,
            'to_idx':    child_idx,
            'from_name': prereq_name,
            'to_name':   child_name,
        })

print(f"Raw edges collected:      {len(edges)}")
print(f"Self-loops found:         {len(self_loops)}")
print(f"Skipped (not in vocab):   {len(skipped)}")

if self_loops:
    print("  Self-loops:")
    for s in self_loops:
        print(f"    {s}")

if skipped:
    print("  Skipped entries:")
    for s in skipped:
        print(f"    {s}")

# ── build dataframe and remove duplicate edges ────────────────────────────────
edges_df = pd.DataFrame(edges)

n_before = len(edges_df)
edges_df = edges_df.drop_duplicates(subset=['from_idx', 'to_idx']).reset_index(drop=True)
n_after  = len(edges_df)

print(f"\nDuplicate edges removed:  {n_before - n_after}")
print(f"Final edge count:         {len(edges_df)}")

# ── validate all indices are in range ─────────────────────────────────────────
all_idx_used = set(edges_df['from_idx'].tolist() + edges_df['to_idx'].tolist())
out_of_range = [i for i in all_idx_used if i < 0 or i >= num_exercises]

if out_of_range:
    print(f"\nWARNING: {len(out_of_range)} out-of-range indices: {out_of_range}")
else:
    print(f"All indices in range [0, {num_exercises - 1}]: OK")

# ── graph analysis using networkx ─────────────────────────────────────────────
print("\n--- Graph Analysis ---")
G = nx.DiGraph()
G.add_nodes_from(range(num_exercises))
for _, row in edges_df.iterrows():
    G.add_edge(int(row['from_idx']), int(row['to_idx']))

print(f"Nodes:                           {G.number_of_nodes()}")
print(f"Edges:                           {G.number_of_edges()}")

is_dag = nx.is_directed_acyclic_graph(G)
print(f"Is valid DAG (no cycles):        {is_dag}")

if not is_dag:
    print("\nWARNING: Graph contains cycles. These are data quality issues in the source.")
    cycles = list(nx.simple_cycles(G))
    print(f"Number of cycles: {len(cycles)}")
    print("First 5 cycles (with names):")
    for cycle in cycles[:5]:
        names = [idx_to_name[str(i)] for i in cycle]
        print(f"  {' -> '.join(names)} -> (back to {names[0]})")

# node degree stats
root_nodes     = [n for n in G.nodes() if G.in_degree(n) == 0]
leaf_nodes     = [n for n in G.nodes() if G.out_degree(n) == 0]
isolated_nodes = [n for n in G.nodes() if G.degree(n) == 0]

print(f"Root nodes (no prerequisites):   {len(root_nodes)}")
print(f"Leaf nodes (no dependents):      {len(leaf_nodes)}")
print(f"Isolated nodes (no edges):       {len(isolated_nodes)}")
print(f"Weakly connected components:     {nx.number_weakly_connected_components(G)}")

# most depended-upon exercises (highest in-degree = most things need them)
in_degrees = sorted([(n, G.in_degree(n)) for n in G.nodes()],
                    key=lambda x: x[1], reverse=True)
print("\nTop 10 most-prerequisite-of exercises (highest in-degree):")
for node, deg in in_degrees[:10]:
    if deg > 0:
        print(f"  [{node:3d}] {idx_to_name[str(node)]} — needed by {deg} exercises")

# most demanding exercises (highest out-degree = needs the most prerequisites)
out_degrees = sorted([(n, G.out_degree(n)) for n in G.nodes()],
                     key=lambda x: x[1], reverse=True)
print("\nTop 10 exercises with most prerequisites (highest out-degree):")
for node, deg in out_degrees[:10]:
    if deg > 0:
        print(f"  [{node:3d}] {idx_to_name[str(node)]} — requires {deg} prerequisites")

# ── sample edge preview ───────────────────────────────────────────────────────
print("\n--- Sample Edges (first 15) ---")
for _, row in edges_df.head(15).iterrows():
    print(f"  [{row['from_idx']:3d}] {row['from_name']:<45s} --> "
          f"[{row['to_idx']:3d}] {row['to_name']}")

# ── save ──────────────────────────────────────────────────────────────────────
# model-facing file: indices only, this is what the GNN reads
edges_df[['from_idx', 'to_idx']].to_csv(OUTPUT_EDGES, index=False)

# human-readable file: includes names, for inspection and debugging
edges_df[['from_idx', 'to_idx', 'from_name', 'to_name']].to_csv(
    OUTPUT_EDGES_NM, index=False
)

print(f"\nSaved: {OUTPUT_EDGES}          (model input — indices only)")
print(f"Saved: {OUTPUT_EDGES_NM}  (human readable — with names)")
print("Step 2 complete.")