import pandas as pd
import numpy as np
import json
import os

# ── config ─────────────────────────────────────────────────────────────────────
PROCESSED_DIR    = os.path.join('data', 'processed')
INTERACTIONS     = os.path.join(PROCESSED_DIR, 'filtered_interactions.csv')
OUT_STATS        = os.path.join(PROCESSED_DIR, 'student_stats.csv')
OUT_SPLITS       = os.path.join(PROCESSED_DIR, 'student_splits.json')

MIN_INTERACTIONS     = 30     # minimum to be a qualified student
WORKING_SUBSET       = 1000   # total students in FL working set
N_STRATA             = 10     # number of deciles for stratified sampling
RANDOM_SEED          = 42
TRAIN_RATIO          = 0.80   # → 800 train clients
VAL_RATIO            = 0.10   # → 100 val students
# TEST gets the remainder → 100 test students

MICROSECONDS_PER_DAY = 86_400_000_000

# Derived — verify divisibility up front so we don't get a silent rounding error
PER_STRATUM = WORKING_SUBSET // N_STRATA
assert PER_STRATUM * N_STRATA == WORKING_SUBSET, (
    f"WORKING_SUBSET ({WORKING_SUBSET}) must be exactly divisible by "
    f"N_STRATA ({N_STRATA})"
)

N_TRAIN_PER_STRATUM = int(PER_STRATUM * TRAIN_RATIO)    # 80
N_VAL_PER_STRATUM   = int(PER_STRATUM * VAL_RATIO)      # 10
N_TEST_PER_STRATUM  = PER_STRATUM - N_TRAIN_PER_STRATUM - N_VAL_PER_STRATUM  # 10

assert N_TRAIN_PER_STRATUM + N_VAL_PER_STRATUM + N_TEST_PER_STRATUM == PER_STRATUM, (
    "Per-stratum split does not add up. Check TRAIN_RATIO / VAL_RATIO."
)

print(f"Config:")
print(f"  Min interactions threshold : {MIN_INTERACTIONS}")
print(f"  Working subset size         : {WORKING_SUBSET}")
print(f"  Strata (deciles)            : {N_STRATA}")
print(f"  Per stratum                 : {PER_STRATUM}  "
      f"({N_TRAIN_PER_STRATUM} train / {N_VAL_PER_STRATUM} val / {N_TEST_PER_STRATUM} test)")
print(f"  Total split                 : {N_TRAIN_PER_STRATUM*N_STRATA} train / "
      f"{N_VAL_PER_STRATUM*N_STRATA} val / {N_TEST_PER_STRATUM*N_STRATA} test")
print(f"  Random seed                 : {RANDOM_SEED}")

# ── load interactions with memory-efficient dtypes ────────────────────────────
print("\nLoading filtered_interactions.csv...")
print("Using explicit dtypes to minimise RAM (~500-700 MB expected).")

dtypes = {
    'user_id':      'int32',
    'exercise_idx': 'int16',
    'correct':      'int8',
    'time_done':    'int64',
    'time_taken':   'float32',
    'hint_used':    'int8',
}

df = pd.read_csv(INTERACTIONS, dtype=dtypes)

mem_mb = df.memory_usage(deep=True).sum() / 1e6
print(f"Loaded: {len(df):,} rows | {df['user_id'].nunique():,} students | {mem_mb:.0f} MB")

# ── per-student statistics ─────────────────────────────────────────────────────
print("\nComputing per-student statistics (this may take a minute)...")
g = df.groupby('user_id')

stats = pd.DataFrame({
    'total_interactions': g.size(),
    'unique_exercises':   g['exercise_idx'].nunique(),
    'accuracy':           g['correct'].mean().round(4),
    'hint_rate':          g['hint_used'].mean().round(4),
    'active_days': (
        (g['time_done'].max() - g['time_done'].min()) / MICROSECONDS_PER_DAY
    ).round(1),
}).reset_index()

print(f"Stats computed for all {len(stats):,} students")

# ── filter qualified students ──────────────────────────────────────────────────
qualified = (
    stats[stats['total_interactions'] >= MIN_INTERACTIONS]
    .copy()
    .reset_index(drop=True)
)

n_disqualified = len(stats) - len(qualified)

print(f"\n--- Qualification filter (>= {MIN_INTERACTIONS} interactions) ---")
print(f"  Total students:        {len(stats):>8,}")
print(f"  Qualified:             {len(qualified):>8,}")
print(f"  Disqualified:          {n_disqualified:>8,}  ({n_disqualified/len(stats)*100:.1f}%)")

print(f"\n--- Qualified student stats ---")
desc = qualified[['total_interactions','unique_exercises','accuracy','active_days']].describe()
print(desc.round(2).to_string())

# Sanity: all qualified students must have >= MIN_INTERACTIONS
assert qualified['total_interactions'].min() >= MIN_INTERACTIONS, (
    "BUG: a student below the threshold slipped through the filter"
)
print(f"\nFilter sanity check: min interactions = "
      f"{qualified['total_interactions'].min()} >= {MIN_INTERACTIONS} — OK")

# ── stratified sampling ────────────────────────────────────────────────────────
print(f"\n--- Stratified sampling of {WORKING_SUBSET} students ({N_STRATA} deciles) ---")

rng = np.random.default_rng(RANDOM_SEED)

# Assign each qualified student to a decile by total_interactions
# pd.qcut splits into equal-frequency bins (each bin has ~same number of students)
# duplicates='drop' handles ties at bin boundaries gracefully
qualified['decile'] = pd.qcut(
    qualified['total_interactions'],
    q=N_STRATA,
    labels=False,
    duplicates='drop'
)

n_actual_strata = qualified['decile'].nunique()

print(f"\nDecile breakdown (qualified students per decile):")
decile_info = qualified.groupby('decile').agg(
    n_students=('user_id','count'),
    min_interactions=('total_interactions','min'),
    max_interactions=('total_interactions','max'),
)
print(decile_info.to_string())

if n_actual_strata < N_STRATA:
    print(f"\nNOTE: pd.qcut produced {n_actual_strata} strata (not {N_STRATA}) due to "
          f"tied values at boundaries. This is expected and acceptable.")

# verify every decile has at least PER_STRATUM students (it should, given 81K total)
min_stratum_size = qualified.groupby('decile').size().min()
assert min_stratum_size >= PER_STRATUM, (
    f"BUG: smallest decile has only {min_stratum_size} students — "
    f"cannot sample {PER_STRATUM}"
)
print(f"\nSmallest decile has {min_stratum_size:,} students — "
      f"sampling {PER_STRATUM} per decile is safe")

# Sample within each decile, split each into train / val / test
train_ids = []
val_ids   = []
test_ids  = []

for decile_label in sorted(qualified['decile'].unique()):
    pool = qualified[qualified['decile'] == decile_label]['user_id'].values.copy()

    # Shuffle using our seeded RNG so results are reproducible
    rng.shuffle(pool)

    sampled = pool[:PER_STRATUM]

    train_ids.extend(sampled[:N_TRAIN_PER_STRATUM].tolist())
    val_ids.extend(  sampled[N_TRAIN_PER_STRATUM : N_TRAIN_PER_STRATUM + N_VAL_PER_STRATUM].tolist())
    test_ids.extend( sampled[N_TRAIN_PER_STRATUM + N_VAL_PER_STRATUM:].tolist())

# Shuffle each split so ordering no longer encodes decile membership
rng.shuffle(train_ids)
rng.shuffle(val_ids)
rng.shuffle(test_ids)

print(f"\n--- Working subset split ---")
print(f"  Train : {len(train_ids):>5}  ({len(train_ids)/WORKING_SUBSET*100:.0f}%)")
print(f"  Val   : {len(val_ids):>5}  ({len(val_ids)/WORKING_SUBSET*100:.0f}%)")
print(f"  Test  : {len(test_ids):>5}  ({len(test_ids)/WORKING_SUBSET*100:.0f}%)")
print(f"  Total : {len(train_ids)+len(val_ids)+len(test_ids):>5}")

# ── overlap checks ─────────────────────────────────────────────────────────────
train_set = set(train_ids)
val_set   = set(val_ids)
test_set  = set(test_ids)

assert len(train_set & val_set)  == 0, "ERROR: overlap between train and val"
assert len(train_set & test_set) == 0, "ERROR: overlap between train and test"
assert len(val_set   & test_set) == 0, "ERROR: overlap between val and test"
assert len(train_set) == len(train_ids), "ERROR: duplicate IDs in train split"
assert len(val_set)   == len(val_ids),   "ERROR: duplicate IDs in val split"
assert len(test_set)  == len(test_ids),  "ERROR: duplicate IDs in test split"

print("\nOverlap / duplicate checks: all passed — OK")

# ── verify working subset covers the full interaction range ────────────────────
all_working    = train_ids + val_ids + test_ids
working_stats  = qualified[qualified['user_id'].isin(set(all_working))]

print(f"\n--- Working subset interaction distribution ---")
print(working_stats['total_interactions'].describe().round(1).to_string())

print(f"\n  Min interactions in subset : {working_stats['total_interactions'].min()}")
print(f"  Max interactions in subset : {working_stats['total_interactions'].max()}")
print(f"  Mean                        : {working_stats['total_interactions'].mean():.1f}")
print(f"  Unique exercises (mean/student): {working_stats['unique_exercises'].mean():.1f}")
print(f"  Mean accuracy in subset     : {working_stats['accuracy'].mean():.4f}")

# ── save student_stats.csv ────────────────────────────────────────────────────
qualified_out = qualified.drop(columns=['decile']).copy()
qualified_out.to_csv(OUT_STATS, index=False)
print(f"\nSaved: {OUT_STATS}")
print(f"  {len(qualified_out):,} qualified students | "
      f"columns: {qualified_out.columns.tolist()}")

# ── save student_splits.json ──────────────────────────────────────────────────
splits = {
    'min_interactions_threshold': int(MIN_INTERACTIONS),
    'total_qualified_students':   int(len(qualified)),
    'working_subset_size':        int(WORKING_SUBSET),
    'sampling_strategy':          'stratified_decile',
    'n_strata':                   int(N_STRATA),
    'random_seed':                int(RANDOM_SEED),
    'train_size':                 int(len(train_ids)),
    'val_size':                   int(len(val_ids)),
    'test_size':                  int(len(test_ids)),
    'train': [int(x) for x in train_ids],
    'val':   [int(x) for x in val_ids],
    'test':  [int(x) for x in test_ids],
}

with open(OUT_SPLITS, 'w') as f:
    json.dump(splits, f, indent=2)

print(f"\nSaved: {OUT_SPLITS}")
print(f"  Train : {len(train_ids)} student IDs")
print(f"  Val   : {len(val_ids)} student IDs")
print(f"  Test  : {len(test_ids)} student IDs")

print("\nStep 4 complete.")