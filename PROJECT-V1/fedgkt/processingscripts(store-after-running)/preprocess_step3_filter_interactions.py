import pandas as pd
import json
import os

# ── paths ─────────────────────────────────────────────────────────────────────
PROCESSED_DIR = os.path.join('data', 'processed')
VOCAB_FILE    = os.path.join(PROCESSED_DIR, 'exercise_vocab.json')
OUTPUT        = os.path.join(PROCESSED_DIR, 'filtered_interactions.csv')

# !! Update this to wherever your CSV actually is on your machine !!
LOG_PATH = os.path.join('data', 'raw', 'junyi_ProblemLog_original.csv')

CHUNK_SIZE = 500_000

# ── load vocab ─────────────────────────────────────────────────────────────────
print("Loading vocab...")
with open(VOCAB_FILE, 'r', encoding='utf-8') as f:
    vocab = json.load(f)

name_to_idx     = vocab['name_to_idx']
valid_exercises = set(name_to_idx.keys())
print(f"Valid exercises in vocab: {len(valid_exercises)}")

# ── process in chunks ──────────────────────────────────────────────────────────
# Only load the 6 columns we actually need — speeds up reading significantly
COLS_TO_READ = ['user_id', 'exercise', 'correct', 'time_done', 'time_taken', 'hint_used']

print(f"\nProcessing log in chunks of {CHUNK_SIZE:,} rows...")
print("This will take several minutes. Progress printed per chunk.\n")

all_chunks       = []
total_read       = 0
dropped_exercise = 0
dropped_correct  = 0
dropped_bad_map  = 0
chunk_num        = 0

for chunk in pd.read_csv(LOG_PATH,
                          usecols=COLS_TO_READ,
                          chunksize=CHUNK_SIZE,
                          low_memory=False):

    chunk_num += 1
    n_in       = len(chunk)
    total_read += n_in

    # ── filter 1: keep only exercises that exist in our vocab ─────────────────
    before = len(chunk)
    chunk  = chunk[chunk['exercise'].isin(valid_exercises)]
    dropped_exercise += before - len(chunk)

    if len(chunk) == 0:
        print(f"  Chunk {chunk_num:3d}: {n_in:>8,} in → 0 kept after exercise filter")
        continue

    # ── filter 2: drop rows where correct is null ─────────────────────────────
    before = len(chunk)
    chunk  = chunk.dropna(subset=['correct'])
    dropped_correct += before - len(chunk)

    if len(chunk) == 0:
        print(f"  Chunk {chunk_num:3d}: {n_in:>8,} in → 0 kept after null-correct filter")
        continue

    # ── convert correct to 0/1 ────────────────────────────────────────────────
    # The column contains Python booleans or 'True'/'False' strings
    chunk['correct'] = chunk['correct'].map(
        {True: 1, False: 0, 'True': 1, 'False': 0,
         1: 1, 0: 0, '1': 1, '0': 0}
    )

    # Drop any rows where correct didn't map to a known value
    before = len(chunk)
    chunk  = chunk.dropna(subset=['correct'])
    dropped_bad_map += before - len(chunk)

    if len(chunk) == 0:
        continue

    chunk['correct'] = chunk['correct'].astype(int)

    # ── convert hint_used to 0/1 ──────────────────────────────────────────────
    # Same boolean handling; fill any unmapped or null values with 0
    chunk['hint_used'] = chunk['hint_used'].map(
        {True: 1, False: 0, 'True': 1, 'False': 0,
         1: 1, 0: 0, '1': 1, '0': 0}
    ).fillna(0).astype(int)

    # ── convert exercise name to integer index ────────────────────────────────
    chunk['exercise_idx'] = chunk['exercise'].map(name_to_idx).astype(int)

    # ── keep only the final columns in the right order ────────────────────────
    chunk = chunk[['user_id', 'exercise_idx', 'correct',
                   'time_done', 'time_taken', 'hint_used']]

    all_chunks.append(chunk)

    print(f"  Chunk {chunk_num:3d}: {n_in:>8,} in → {len(chunk):>7,} kept  "
          f"| total read so far: {total_read:>10,}")

# ── combine ───────────────────────────────────────────────────────────────────
print(f"\nCombining {len(all_chunks)} filtered chunks...")
df = pd.concat(all_chunks, ignore_index=True)
del all_chunks   # free memory immediately after concat
print(f"Combined: {len(df):,} rows, {df['user_id'].nunique():,} unique students")

# ── sort chronologically within each student ──────────────────────────────────
# time_done is a Unix timestamp in microseconds — larger = later, so ascending sort
# is correct chronological order
print("Sorting by user_id then time_done (chronological)...")
df = df.sort_values(['user_id', 'time_done']).reset_index(drop=True)

# ── sanity check: no nulls remain ────────────────────────────────────────────
null_counts = df.isnull().sum()
if null_counts.sum() > 0:
    print(f"\nWARNING: null values remain after filtering:")
    print(null_counts[null_counts > 0])
else:
    print("Null check: no nulls in final dataframe — clean")

# ── statistics ────────────────────────────────────────────────────────────────
print(f"\n{'='*50}")
print(f"FILTERING SUMMARY")
print(f"{'='*50}")
print(f"Total rows read:                     {total_read:>10,}")
print(f"Dropped (exercise not in vocab):      {dropped_exercise:>10,}")
print(f"Dropped (correct was null):           {dropped_correct:>10,}")
print(f"Dropped (correct was unmappable):     {dropped_bad_map:>10,}")
print(f"Final rows kept:                      {len(df):>10,}")
print(f"Unique students:                      {df['user_id'].nunique():>10,}")

print(f"\n--- Interactions per student ---")
per_student = df.groupby('user_id').size()
print(per_student.describe().to_string())
print(f"\nStudents with >=  10 interactions:  {(per_student >= 10).sum():>8,}")
print(f"Students with >=  20 interactions:  {(per_student >= 20).sum():>8,}")
print(f"Students with >=  30 interactions:  {(per_student >= 30).sum():>8,}")
print(f"Students with >=  50 interactions:  {(per_student >= 50).sum():>8,}")
print(f"Students with >= 100 interactions:  {(per_student >= 100).sum():>8,}")
print(f"Students with >= 200 interactions:  {(per_student >= 200).sum():>8,}")

print(f"\n--- Data quality checks ---")
print(f"Correct column unique values:  {sorted(df['correct'].unique().tolist())}")
print(f"hint_used column unique values: {sorted(df['hint_used'].unique().tolist())}")
print(f"Overall accuracy:  {df['correct'].mean():.4f}")
print(f"Hint used rate:    {df['hint_used'].mean():.4f}")

print(f"\n--- exercise_idx range ---")
print(f"Min idx: {df['exercise_idx'].min()}  (expected 0)")
print(f"Max idx: {df['exercise_idx'].max()}  (expected <= 834)")
print(f"Unique exercises in log: {df['exercise_idx'].nunique()}  (out of 835)")

print(f"\n--- time_done range ---")
print(f"Earliest: {df['time_done'].min()}")
print(f"Latest:   {df['time_done'].max()}")

# ── save ──────────────────────────────────────────────────────────────────────
print(f"\nSaving to {OUTPUT} ...")
df.to_csv(OUTPUT, index=False)
print(f"Saved: {len(df):,} rows")
print("Step 3 complete.")