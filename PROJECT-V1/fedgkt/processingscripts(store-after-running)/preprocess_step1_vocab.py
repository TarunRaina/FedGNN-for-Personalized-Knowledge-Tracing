import pandas as pd
import json
import os

# ── paths ────────────────────────────────────────────────────
RAW_DIR       = os.path.join('data', 'raw')
PROCESSED_DIR = os.path.join('data', 'processed')
os.makedirs(PROCESSED_DIR, exist_ok=True)

EXERCISE_TABLE = os.path.join(RAW_DIR, 'junyi_Exercise_table.csv')
OUTPUT_VOCAB   = os.path.join(PROCESSED_DIR, 'exercise_vocab.json')

# ── load exercise table ───────────────────────────────────────
print("Loading exercise table...")
ex = pd.read_csv(EXERCISE_TABLE, encoding='utf-8')
print(f"Rows loaded: {len(ex)}")

# ── sanity checks ─────────────────────────────────────────────
print("\n--- Sanity Checks ---")
print(f"Total exercises:        {len(ex)}")
print(f"Unique exercise names:  {ex['name'].nunique()}")
print(f"Duplicate names:        {len(ex) - ex['name'].nunique()}")
print(f"Live exercises:         {ex['live'].sum()}")
print(f"Non-live exercises:     {(~ex['live']).sum()}")
print(f"Exercises with prereqs: {ex['prerequisites'].notna().sum()}")
print(f"Root nodes (no prereq): {ex['prerequisites'].isna().sum()}")

# ── build vocab ───────────────────────────────────────────────
# Sort by name so vocab is deterministic every time you rebuild it
ex_sorted = ex.sort_values('name').drop_duplicates(subset='name', keep='first').reset_index(drop=True)

name_to_idx = {row['name']: idx for idx, row in ex_sorted.iterrows()}
idx_to_name = {idx: row['name'] for idx, row in ex_sorted.iterrows()}

print(f"\n--- Vocab Built ---")
print(f"Vocab size: {len(name_to_idx)}")
print(f"Index range: 0 to {len(name_to_idx) - 1}")

# ── check all prerequisite references exist in vocab ─────────
print("\n--- Checking prerequisite references ---")
missing_refs = []
for _, row in ex.iterrows():
    if pd.notna(row['prerequisites']):
        prereqs = [p.strip() for p in str(row['prerequisites']).split(',')]
        for p in prereqs:
            if p not in name_to_idx:
                missing_refs.append(p)

if missing_refs:
    print(f"WARNING: {len(missing_refs)} prerequisite references not in vocab:")
    for m in missing_refs:
        print(f"  - {m}")
else:
    print("All prerequisite references exist in vocab. Graph will be complete.")

# ── preview ───────────────────────────────────────────────────
print("\n--- First 10 vocab entries ---")
for name, idx in list(name_to_idx.items())[:10]:
    print(f"  {idx:3d}  {name}")

# ── save ─────────────────────────────────────────────────────
vocab = {
    'name_to_idx': name_to_idx,
    'idx_to_name': idx_to_name,
    'num_exercises': len(name_to_idx)
}

with open(OUTPUT_VOCAB, 'w', encoding='utf-8') as f:
    json.dump(vocab, f, indent=2, ensure_ascii=False)

print(f"\nSaved to: {OUTPUT_VOCAB}")
print("Step 1 complete.")