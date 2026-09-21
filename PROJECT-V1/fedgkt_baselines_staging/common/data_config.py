"""
common/data_config.py

Builds (and writes to disk) the pyKT `data_config` entry for our Junyi
subset, plus a guard against pyKT's silent sequence cache.

WHY THIS FILE EXISTS
--------------------
Step 1 bypassed pyKT's own preprocessor entirely and produced the three
sequence CSVs directly, for reasons that are locked: pyKT's Junyi
preprocessing starts from a different export, applies its own random
splitter, and re-maps concept ids through id_mapping -- all of which
would have destroyed the student-level split and the 0-834 vocabulary
that FedGKT is locked to.

The cost of bypassing it is that nothing ever wrote pyKT's
`data_config` entry, which is the dict pyKT reads to find the CSVs and
to know how many concepts exist. Step 1 has a build_data_config_entry()
helper but never saves it. This file closes that gap.

WHAT pyKT ACTUALLY READS (verified against the installed 0.0.38 source,
not assumed):

  dpath              folder holding the CSVs; also where GKT looks for
                     its graph .npz
  num_c              concept count -> passed straight into every model
                     constructor in init_model.py
  num_q              question count. 0 for us: concept-only registration.
                     Junyi's exercise vocabulary IS the finest identifier
                     that exists in our locked data, so there is no
                     separate question id to supply. This is why AKT's
                     Rasch difficulty term operates at concept
                     granularity in our setup -- a real, disclosable
                     limitation of the dataset, not a bug.
  input_type         ["concepts"] -- KTDataset only populates cseqs
  max_concepts       1 -- one concept per interaction
  train_valid_file   chunked train+val sequences, carries the fold column
  test_file          chunked test sequences, fold == -1
  test_window_file   sliding-window test sequences, fold == -1
  folds              [0, 1]. init_dataset4train trains on
                     (all folds - {i}) and validates on {i}, so passing
                     i=1 gives train=fold 0 (800 students),
                     valid=fold 1 (100 students). Exactly our split.
  emb_path           "" -- no pretrained concept embeddings

THE CACHE TRAP (verified in pykt/datasets/data_loader.py)
---------------------------------------------------------
KTDataset writes a processed copy of each CSV next to it, named
<csv>_<folds>.pkl, and on every later run reads the .pkl and never
looks at the CSV again:

    if not os.path.exists(processed_data):
        ... parse the CSV ...
        pd.to_pickle(save_data, processed_data)
    else:
        self.dori = pd.read_pickle(processed_data)

So if the CSVs are ever regenerated and the .pkl files are left behind,
pyKT keeps training on the OLD data, silently, with no warning. Every
baseline would be affected identically and the results would look
perfectly plausible. clear_sequence_cache() below exists for exactly
that case and should be called whenever Step 1 is re-run.

THE WINDOW-LOADER TRAP (verified in pykt/datasets/init_dataset.py)
-------------------------------------------------------------------
init_dataset4train() -- the function that sets up training -- has its
windowed test loader COMMENTED OUT and hard-returns None:

    # test_window_loader = DataLoader(test_window_dataset, ...)
    test_window_loader = None

Only init_test_datasets() builds it. Since the chunked test file drops
sequence tails shorter than min_seq_len while the windowed file does
not, the windowed file is the one our reported test numbers must come
from. Drivers therefore call init_test_datasets() explicitly for final
evaluation rather than using whatever init_dataset4train() hands back.
Noted here because it is the kind of thing that silently puts headline
numbers on the wrong file.
"""

import json
import os


DATASET_NAME = 'junyi_fedgkt'

NUM_CONCEPTS = 835          # locked: exercise vocabulary, indices 0-834
MAXLEN = 200                # locked: matches Step 1's chunking
MIN_SEQ_LEN = 3             # locked: matches Step 1

TRAIN_VALID_FILE = 'train_valid_sequences.csv'
TEST_FILE = 'test_sequences.csv'
TEST_WINDOW_FILE = 'test_window_sequences.csv'

# fold 0 = 800 train students, fold 1 = 100 validation students.
# Pass i=1 to init_dataset4train to get that split.
FOLDS = [0, 1]
VALIDATION_FOLD = 1


def build_data_config(sequences_dir):
    """
    Returns the data_config dict for our dataset. sequences_dir is the
    folder holding the three CSVs (pykt_sequences/).

    Returned in pyKT's own nested shape -- {dataset_name: {...}} --
    because init_dataset4train() immediately does
    data_config = data_config[dataset_name].
    """
    dpath = os.path.abspath(sequences_dir)

    for fname in (TRAIN_VALID_FILE, TEST_FILE, TEST_WINDOW_FILE):
        path = os.path.join(dpath, fname)
        assert os.path.exists(path), (
            f"Sequence file missing: {path}\n"
            f"Run baseline_step1_build_pykt_sequences.py first."
        )

    return {
        DATASET_NAME: {
            'dpath': dpath,
            'num_c': NUM_CONCEPTS,
            'num_q': 0,
            'input_type': ['concepts'],
            'max_concepts': 1,
            'train_valid_file': TRAIN_VALID_FILE,
            'test_file': TEST_FILE,
            'test_window_file': TEST_WINDOW_FILE,
            'folds': FOLDS,
            'emb_path': '',
            'maxlen': MAXLEN,
            'min_seq_len': MIN_SEQ_LEN,
        }
    }


def write_data_config(sequences_dir, out_path):
    """Writes the data_config to disk as JSON and returns the dict."""
    config = build_data_config(sequences_dir)
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(config, f, indent=2)
    return config


def clear_sequence_cache(sequences_dir, verbose=True):
    """
    Deletes pyKT's cached .pkl sidecars so the CSVs are re-parsed.

    Call this after ANY regeneration of the sequence CSVs. Not calling it
    means pyKT silently keeps using the previous data -- see the module
    docstring. Returns the list of files removed.
    """
    dpath = os.path.abspath(sequences_dir)
    removed = []
    for fname in sorted(os.listdir(dpath)):
        if fname.endswith('.pkl'):
            path = os.path.join(dpath, fname)
            os.remove(path)
            removed.append(fname)
    if verbose:
        if removed:
            print(f"Cleared {len(removed)} cached .pkl file(s): {removed}")
        else:
            print("No cached .pkl files to clear.")
    return removed


def verify_sequences(sequences_dir, verbose=True):
    """
    Structural check of the three CSVs before any training touches them.

    IMPORTANT -- what -1 means in selectmasks (verified against
    generate_window_sequences in 0.0.38; an earlier version of this
    function got it wrong and raised 16,008 false alarms):

      -1 in selectmasks does NOT always mean padding. In a sliding-window
      row it means "real interaction, present as context, but not scored":

          dres["selectmasks"].append(
              ",".join([str(pad_val)] * (maxlen - 1) + ["1"]))

      while concepts/responses in that same row hold genuine values
      (dcur[key][j-maxlen : j]). Only a row's trailing region, where the
      CONCEPT itself is -1, is true padding.

    So the invariant is driven off concepts, not selectmasks:
      concept == -1  -> padding; response and selectmask must also be -1
      concept != -1  -> real; response must be 0/1, selectmask may be
                        either 1 (scored) or -1 (context only)

    Also reports the number of predictions each file will actually be
    scored on AFTER pyKT's dataloader slices smasks[:, 1:], which drops
    position 0 of every row. For the windowed test file this comes to
    exactly one per interaction minus one per student -- the
    first-interaction asymmetry, counted from the data rather than
    assumed.
    """
    import pandas as pd

    dpath = os.path.abspath(sequences_dir)
    expected_folds = {
        TRAIN_VALID_FILE: {0, 1},
        TEST_FILE: {-1},
        TEST_WINDOW_FILE: {-1},
    }

    all_ok = True
    for fname, want_folds in expected_folds.items():
        path = os.path.join(dpath, fname)
        df = pd.read_csv(path)

        assert 'fold' in df.columns, (
            f"{fname}: no 'fold' column (columns are {list(df.columns)}). "
            f"KTDataset does df[df['fold'].isin(folds)] and would crash."
        )

        got_folds = set(int(f) for f in df['fold'].unique())
        folds_ok = got_folds == want_folds

        bad_len = bad_pad = bad_concept = bad_response = bad_mask = 0
        n_real = 0        # real interactions present in the file
        n_scored = 0      # positions pyKT will actually score (after smasks[1:])

        for _, row in df.iterrows():
            c = [int(v) for v in str(row['concepts']).split(',')]
            r = [int(v) for v in str(row['responses']).split(',')]
            s = [int(v) for v in str(row['selectmasks']).split(',')]

            if not (len(c) == len(r) == len(s) == MAXLEN):
                bad_len += 1
                continue

            for ci, ri, si in zip(c, r, s):
                if ci == -1:
                    # true padding -- everything else must be padding too
                    if ri != -1 or si != -1:
                        bad_pad += 1
                        break
                else:
                    if not (0 <= ci < NUM_CONCEPTS):
                        bad_concept += 1
                        break
                    if ri not in (0, 1):
                        bad_response += 1
                        break
                    if si not in (1, -1):
                        bad_mask += 1
                        break
                    n_real += 1

            # pyKT drops position 0 of every row: dori["smasks"][:, 1:]
            n_scored += sum(1 for v in s[1:] if v == 1)

        row_ok = (bad_len == bad_pad == bad_concept
                  == bad_response == bad_mask == 0)
        all_ok = all_ok and folds_ok and row_ok

        if verbose:
            status = 'OK' if (folds_ok and row_ok) else 'PROBLEM'
            print(f"  {fname:32s} rows={len(df):6,}  students={df['uid'].nunique():4}  "
                  f"folds={sorted(got_folds)}  [{status}]")
            print(f"      real interactions in file: {n_real:,}   "
                  f"positions pyKT will score: {n_scored:,}")
            if not folds_ok:
                print(f"      fold mismatch -- expected {sorted(want_folds)}, "
                      f"got {sorted(got_folds)}")
            if not row_ok:
                print(f"      bad_len={bad_len} bad_pad={bad_pad} "
                      f"bad_concept={bad_concept} bad_response={bad_response} "
                      f"bad_mask={bad_mask}")

    return all_ok


if __name__ == '__main__':
    print("=" * 70)
    print("common/data_config.py -- self-test")
    print("=" * 70)

    _THIS_DIR = os.path.dirname(os.path.abspath(__file__))       # common/
    _STAGING = os.path.dirname(_THIS_DIR)                         # staging root
    SEQ_DIR = os.path.join(_STAGING, 'pykt_sequences')

    print(f"\nSequences: {SEQ_DIR}")

    print("\n--- Structural check of the three CSVs ---")
    ok = verify_sequences(SEQ_DIR)
    assert ok, "Sequence files failed structural verification -- see above."
    print("  All three files structurally valid.")

    print("\n--- data_config ---")
    out_path = os.path.join(_STAGING, 'pykt_sequences', 'data_config.json')
    config = write_data_config(SEQ_DIR, out_path)
    for k, v in config[DATASET_NAME].items():
        print(f"  {k:24s} {v}")
    print(f"\nWritten: {out_path}")

    print("\nSelf-test complete -- no assertion failures.")