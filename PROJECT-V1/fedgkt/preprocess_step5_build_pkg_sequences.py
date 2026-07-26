import pandas as pd
import numpy as np
import torch
import json
import os

# ── config ─────────────────────────────────────────────────────────────────────
PROCESSED_DIR   = os.path.join('data', 'processed')
INTERACTIONS    = os.path.join(PROCESSED_DIR, 'filtered_interactions.csv')
EDGES           = os.path.join(PROCESSED_DIR, 'prerequisite_edges.csv')
SPLITS          = os.path.join(PROCESSED_DIR, 'student_splits.json')
PKG_DIR         = os.path.join(PROCESSED_DIR, 'pkgs')
OUT_EDGE_INDEX  = os.path.join(PROCESSED_DIR, 'edge_index.pt')

EXPECTED_NODES  = 835
EXPECTED_EDGES  = 978
EXPECTED_STUDENTS = 1000
MIN_INTERACTIONS  = 30   # sanity re-check only, not re-applied as a filter here

print("=" * 70)
print("STEP 5 — Build per-student PKG interaction sequences + shared edge_index")
print("=" * 70)

# ── load student_splits.json and build target ID set ──────────────────────────
print("\nLoading student_splits.json...")
with open(SPLITS, 'r') as f:
    splits = json.load(f)

train_ids = splits['train']
val_ids   = splits['val']
test_ids  = splits['test']

print(f"  train: {len(train_ids)}  val: {len(val_ids)}  test: {len(test_ids)}")

target_ids = set(train_ids) | set(val_ids) | set(test_ids)

n_expected_union = len(train_ids) + len(val_ids) + len(test_ids)
assert len(target_ids) == n_expected_union, (
    f"BUG: union of train/val/test has {len(target_ids)} unique IDs, "
    f"expected {n_expected_union} (train+val+test with zero overlap). "
    f"There is an overlap between splits that should not exist."
)
assert len(target_ids) == EXPECTED_STUDENTS, (
    f"BUG: expected exactly {EXPECTED_STUDENTS} working-subset students, "
    f"got {len(target_ids)}"
)
print(f"  Union (train+val+test), no overlap: {len(target_ids)} students — OK")

# ── build and save shared edge_index.pt ────────────────────────────────────────
print("\nLoading prerequisite_edges.csv...")
edges_df = pd.read_csv(EDGES)

assert list(edges_df.columns[:2]) == ['from_idx', 'to_idx'], (
    f"BUG: unexpected columns in prerequisite_edges.csv: {edges_df.columns.tolist()}"
)
assert len(edges_df) == EXPECTED_EDGES, (
    f"BUG: expected {EXPECTED_EDGES} edges, found {len(edges_df)}. "
    f"Has prerequisite_edges.csv been regenerated without the Step 2b cycle fix?"
)

min_idx = min(edges_df['from_idx'].min(), edges_df['to_idx'].min())
max_idx = max(edges_df['from_idx'].max(), edges_df['to_idx'].max())
assert min_idx >= 0 and max_idx <= EXPECTED_NODES - 1, (
    f"BUG: edge indices out of expected range [0, {EXPECTED_NODES - 1}], "
    f"found range [{min_idx}, {max_idx}]"
)

edge_index = torch.tensor(
    np.stack([edges_df['from_idx'].values, edges_df['to_idx'].values]),
    dtype=torch.long
)
assert edge_index.shape == (2, EXPECTED_EDGES), (
    f"BUG: edge_index has shape {tuple(edge_index.shape)}, "
    f"expected (2, {EXPECTED_EDGES})"
)

os.makedirs(PROCESSED_DIR, exist_ok=True)
torch.save(edge_index, OUT_EDGE_INDEX)
print(f"  edge_index shape: {tuple(edge_index.shape)}  dtype: {edge_index.dtype}")
print(f"  Saved: {OUT_EDGE_INDEX}")

# ── load filtered_interactions.csv, filter to working subset only ─────────────
print("\nLoading filtered_interactions.csv (full load, then filtering to 1,000 IDs)...")
print("This mirrors Step 4's approach — ~500-700 MB expected, well within budget.")

dtypes = {
    'user_id':      'int32',
    'exercise_idx': 'int16',
    'correct':      'int8',
    'time_done':    'int64',
    'time_taken':   'float32',
    'hint_used':    'int8',
}

df = pd.read_csv(INTERACTIONS, dtype=dtypes)
print(f"  Full file loaded: {len(df):,} rows, {df['user_id'].nunique():,} students")

df_subset = df[df['user_id'].isin(target_ids)].copy()
del df  # free the full 2GB+ dataframe now that we only need the subset

n_students_found = df_subset['user_id'].nunique()
print(f"  Filtered to working subset: {len(df_subset):,} rows, "
      f"{n_students_found:,} students")

assert n_students_found == EXPECTED_STUDENTS, (
    f"BUG: expected interactions for {EXPECTED_STUDENTS} students in the working "
    f"subset, found {n_students_found}. Some student_ids from student_splits.json "
    f"may be missing from filtered_interactions.csv — check for a mismatch between "
    f"Step 3 and Step 4 outputs."
)

# exercise_idx range check on the subset actually used
min_ex = df_subset['exercise_idx'].min()
max_ex = df_subset['exercise_idx'].max()
assert min_ex >= 0 and max_ex <= EXPECTED_NODES - 1, (
    f"BUG: exercise_idx out of range in subset: [{min_ex}, {max_ex}], "
    f"expected within [0, {EXPECTED_NODES - 1}]"
)

# no nulls
n_nulls = df_subset.isnull().sum().sum()
assert n_nulls == 0, f"BUG: {n_nulls} null values found in working-subset interactions"
print(f"  Null check: no nulls — OK")

# ── build per-student PKG sequence files ───────────────────────────────────────
print(f"\nBuilding {EXPECTED_STUDENTS} per-student sequence files...")
os.makedirs(PKG_DIR, exist_ok=True)

interaction_counts = []
n_saved = 0
n_progress_interval = 100

grouped = df_subset.groupby('user_id')

for user_id, group in grouped:
    # defensive: sort chronologically even though Step 3 already sorted globally —
    # guarantees per-student order is correct regardless of upstream assumptions
    g = group.sort_values('time_done', ascending=True).reset_index(drop=True)

    n_interactions = len(g)
    assert n_interactions >= MIN_INTERACTIONS, (
        f"BUG: student {user_id} has only {n_interactions} interactions in the "
        f"working subset, below the {MIN_INTERACTIONS} qualification threshold. "
        f"This should be impossible given Step 4's filtering."
    )

    pkg_data = {
        'user_id':      int(user_id),
        'exercise_idx': torch.tensor(g['exercise_idx'].values, dtype=torch.long),
        'correct':      torch.tensor(g['correct'].values, dtype=torch.float32),
        'time_done':    torch.tensor(g['time_done'].values, dtype=torch.int64),
        'time_taken':   torch.tensor(g['time_taken'].values, dtype=torch.float32),
        'hint_used':    torch.tensor(g['hint_used'].values, dtype=torch.float32),
    }

    # sanity: every tensor must have exactly n_interactions elements
    for key in ['exercise_idx', 'correct', 'time_done', 'time_taken', 'hint_used']:
        assert pkg_data[key].shape[0] == n_interactions, (
            f"BUG: tensor '{key}' for student {user_id} has length "
            f"{pkg_data[key].shape[0]}, expected {n_interactions}"
        )

    # sanity: chronological order actually holds after our own sort
    td = pkg_data['time_done']
    assert torch.all(td[1:] >= td[:-1]), (
        f"BUG: time_done not monotonically non-decreasing for student {user_id} "
        f"after sorting — data integrity issue"
    )

    out_path = os.path.join(PKG_DIR, f'pkg_{user_id}.pt')
    torch.save(pkg_data, out_path)

    interaction_counts.append(n_interactions)
    n_saved += 1

    if n_saved % n_progress_interval == 0:
        print(f"  Saved {n_saved}/{EXPECTED_STUDENTS} student PKG files...")

print(f"  Saved {n_saved}/{EXPECTED_STUDENTS} student PKG files — done")

# ── final verification ──────────────────────────────────────────────────────────
assert n_saved == EXPECTED_STUDENTS, (
    f"BUG: saved {n_saved} files, expected {EXPECTED_STUDENTS}"
)

saved_files = [f for f in os.listdir(PKG_DIR) if f.startswith('pkg_') and f.endswith('.pt')]
assert len(saved_files) == EXPECTED_STUDENTS, (
    f"BUG: {len(saved_files)} .pt files found on disk in {PKG_DIR}, "
    f"expected {EXPECTED_STUDENTS}"
)
print(f"\nFile count on disk confirmed: {len(saved_files)} files in {PKG_DIR}")

# spot-check: reload 3 random files and confirm they load and match expectations
print("\n--- Spot-check: reloading 3 saved files ---")
rng = np.random.default_rng(42)
sample_ids = rng.choice(list(target_ids), size=3, replace=False)

for uid in sample_ids:
    path = os.path.join(PKG_DIR, f'pkg_{int(uid)}.pt')
    loaded = torch.load(path, weights_only=False)
    print(f"  user_id={loaded['user_id']}  "
          f"n_interactions={loaded['exercise_idx'].shape[0]}  "
          f"exercise_idx dtype={loaded['exercise_idx'].dtype}  "
          f"correct dtype={loaded['correct'].dtype}")
    assert loaded['user_id'] == int(uid), "BUG: user_id mismatch after reload"

print("\n--- Interaction count summary (working subset, from saved files) ---")
counts = pd.Series(interaction_counts)
print(counts.describe().round(1).to_string())
print(f"\n  Min: {counts.min()}  Max: {counts.max()}  Total: {counts.sum():,}")

print(f"\nedge_index.pt: shape {tuple(edge_index.shape)}")
print(f"pkgs/ folder: {n_saved} files")
print("\nStep 5 complete.")