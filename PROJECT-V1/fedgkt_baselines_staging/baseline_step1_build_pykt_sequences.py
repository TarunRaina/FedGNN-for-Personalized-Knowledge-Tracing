"""
baseline_step1_build_pykt_sequences.py

Lives in fedgkt_baselines_staging/ (a NEW sibling folder to fedgkt/,
deliberately NOT nested inside it and NOT reusing fedgkt_colab_staging/ --
that folder is scoped to the FedGKT GPU speed experiment specifically;
this is a separate workstream: baseline model comparison).

Converts FedGKT's own locked, already-vocabulary-mapped student data into
the exact CSV format pyKT's own Dataset classes expect -- WITHOUT running
pyKT's own preprocessing pipeline (pykt.preprocess.split_datasets.main()).

WHY NOT pyKT's own main()/read_data()/id_mapping() pipeline:
  1. It expects a raw "intermediate" 6-line-per-student text file built
     from a *different* Junyi export than ours (DataShop's ProblemLog,
     not whatever pyKT's own junyi2015_preprocess.py targets).
  2. Its train_test_split()/KFold_split() are random (seed=1024, NOT
     stratified by decile), which would NOT reproduce FedGKT's own
     student_splits.json (800/100/100, stratified-decile, seed=42) --
     using pyKT's own splitter here would make every baseline comparison
     apples-to-oranges, exactly what this project has been careful to
     avoid throughout.
  3. Its id_mapping() reassigns concept ids by FIRST-ENCOUNTERED ORDER
     across the whole dataset -- since our exercise_idx values (0-834)
     are ALREADY FedGKT's own locked vocabulary indices (matching
     exercise_vocab.json and, later, the GKT custom graph's node order),
     running id_mapping() on them would silently SCRAMBLE that ordering
     into a new, different 0-834 assignment. We must bypass it entirely
     and write our own identity keyid2idx.json instead.

WHAT WE DO INSTEAD: build the same in-memory dataframe shape that
read_data()+id_mapping() would have produced (one row per student, with
'fold', 'uid', 'concepts' [comma-joined exercise_idx as strings],
'responses' [comma-joined 0/1]), then call pyKT's OWN, already-tested
generate_sequences() / generate_window_sequences() functions directly on
it. This reuses pyKT's proven chunking/windowing code byte-for-byte
rather than reimplementing it.

CONCEPT-ONLY REGISTRATION: FedGKT has no separate question-vs-skill
layer -- one Junyi exercise IS one concept, 835 total. pyKT natively
supports this via concept-only datasets (e.g. assist2015: num_q=0,
input_type=["concepts"]). DKT/DKVMN/SAKT/AKT/GKT are all classic
concept-only architectures, so this is a correct registration, not a
workaround.

FOLD ASSIGNMENT:
  - train_valid_sequences.csv contains ONLY the 800 train + 100 val
    students, fold=0 for train, fold=1 for val. Training scripts run
    with `--fold 1`, so init_dataset4train() reads:
        curvalid = df[df['fold'] == 1]       -> our 100 val students
        curtrain = df[df['fold'].isin({0})]  -> our 800 train students
  - test_sequences.csv / test_window_sequences.csv are a COMPLETELY
    SEPARATE pair of files, built from the 100 held-out test students
    via their own call to generate_sequences()/generate_window_sequences().
    fold=-1 is written on these rows only because that's pyKT's own
    convention for test rows -- it is NOT read by init_dataset4train()
    at all, since that function only ever looks inside
    train_valid_sequences.csv.

IGNORED FIELDS: time_taken and hint_used exist in every pkg_<uid>.pt (6
total keys) but are used by neither FedGKT's own locked training code nor
any of the 5 baselines here. time_done is also not needed: none of our 5
locked baselines consume timestamps/usetimes (only pyKT's lpkt model
does) -- so "timestamps"/"usetimes" are left out of effective_keys
entirely.

ISOLATION MODEL (confirmed with the student before this was written):
  fedgkt/                          <- LOCKED. Read from exactly ONCE, by
                                       stage_input_data(), and never again.
  fedgkt_baselines_staging/        <- everything below lives here
      staged_input/                <- one-time COPY of pkgs/,
                                       student_splits.json,
                                       exercise_vocab.json out of fedgkt/
      pykt_sequences/               <- this script's actual output

This is a structural guarantee, not just a comment: stage_input_data()
is the ONLY function in this file that takes fedgkt_root as a parameter.
Every function after it (build_all_sequences and everything it calls)
takes only staged_input_dir/output_dir paths -- there is no code path
through which fedgkt_root could leak back in and get written to.

USAGE:
  python baseline_step1_build_pykt_sequences.py
      Real run. Defaults to the confirmed layout: fedgkt/ is a sibling
      of the folder this script itself lives in.
  python baseline_step1_build_pykt_sequences.py --fedgkt-root <path>
      Real run, with fedgkt/ at a different path than the default guess.
  python baseline_step1_build_pykt_sequences.py --selftest
      Runs the self-test (3 known students) instead of the real
      conversion -- verifies stage_input_data() and the sequence-building
      logic against hand-derived expected values.
"""

import argparse
import importlib.metadata
import importlib.util
import json
import os
import shutil
import tempfile

import numpy as np
import pandas as pd
import torch


def _load_pykt_split_datasets():
    """
    Loads pykt.preprocess.split_datasets WITHOUT triggering a normal
    `import pykt...`, which would run pykt/__init__.py -> pykt.models ->
    qdkt.py's `from turtle import forward` -- a real bug confirmed in the
    current PyPI release (pykt-toolkit 0.0.38): this crashes with
    "ModuleNotFoundError: No module named 'tkinter'" on any headless
    environment, including a fresh Colab runtime, and has nothing to do
    with anything this script needs. split_datasets.py itself has no
    pykt-internal imports, so it loads cleanly in isolation via its file
    path, found through package metadata (which does not execute any of
    the package's code, unlike a real import).
    """
    try:
        from pykt.preprocess.split_datasets import (
            generate_sequences,
            generate_window_sequences,
        )
        return generate_sequences, generate_window_sequences
    except Exception:
        pass

    dist = importlib.metadata.distribution("pykt-toolkit")
    module_path = dist.locate_file("pykt/preprocess/split_datasets.py")
    spec = importlib.util.spec_from_file_location(
        "pykt_split_datasets_standalone", module_path
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.generate_sequences, module.generate_window_sequences


generate_sequences, generate_window_sequences = _load_pykt_split_datasets()

NUM_CONCEPTS = 835          # locked, from exercise_vocab.json's num_exercises
MAXLEN = 200                 # pyKT default, matches its standard baseline configs
MIN_SEQ_LEN = 3              # pyKT default
DATASET_NAME = "fedgkt_junyi2015"

# effective_keys deliberately excludes "questions", "timestamps", "usetimes":
# concept-only dataset, and none of the 5 locked baselines consume time info.
TRAIN_VALID_EFFECTIVE_KEYS = ["fold", "uid", "concepts", "responses"]
TEST_EFFECTIVE_KEYS = ["fold", "uid", "concepts", "responses"]


# ── STEP 0: the one-time copy-in. The ONLY function that ever sees fedgkt_root ──
def stage_input_data(fedgkt_root, staging_root):
    """
    Copies exactly three things out of FedGKT's locked data/processed/
    into staging_root/staged_input/: pkgs/, student_splits.json,
    exercise_vocab.json. This is the ONLY function in this entire file
    that reads from fedgkt_root -- every function below takes
    staged_input_dir/output_dir paths only. Safe to re-run (overwrites
    the staged copy via dirs_exist_ok, never touches fedgkt_root itself).
    """
    src_processed = os.path.join(fedgkt_root, "data", "processed")
    staged_input_dir = os.path.join(staging_root, "staged_input")
    os.makedirs(staged_input_dir, exist_ok=True)

    src_pkgs = os.path.join(src_processed, "pkgs")
    dst_pkgs = os.path.join(staged_input_dir, "pkgs")
    assert os.path.isdir(src_pkgs), f"BUG: source pkgs/ not found at {src_pkgs}"
    shutil.copytree(src_pkgs, dst_pkgs, dirs_exist_ok=True)

    for fname in ("student_splits.json", "exercise_vocab.json"):
        src = os.path.join(src_processed, fname)
        assert os.path.isfile(src), f"BUG: source file not found at {src}"
        shutil.copy2(src, os.path.join(staged_input_dir, fname))

    n_pkg_files = len([f for f in os.listdir(dst_pkgs) if f.endswith(".pt")])
    print(f"[stage_input_data] staged {n_pkg_files} pkg files + "
          f"student_splits.json + exercise_vocab.json -> {staged_input_dir}")
    return staged_input_dir


def load_pkg_sequence(pkg_path):
    """
    Reads one pkg_<uid>.pt (6-key dict: user_id, exercise_idx, correct,
    time_done, time_taken, hint_used -- confirmed against real files).
    Returns (user_id, exercise_idx_list[int], correct_list[int 0/1]).
    Only the 3 fields the 5 baselines can actually use are read.
    """
    d = torch.load(pkg_path, weights_only=False)
    user_id = int(d["user_id"])
    exercise_idx = d["exercise_idx"].tolist()
    correct = [int(round(c)) for c in d["correct"].tolist()]
    assert len(exercise_idx) == len(correct), (
        f"BUG: user {user_id} exercise_idx/correct length mismatch "
        f"({len(exercise_idx)} vs {len(correct)})"
    )
    for e in exercise_idx:
        assert 0 <= e < NUM_CONCEPTS, (
            f"BUG: user {user_id} has exercise_idx={e} out of range "
            f"[0, {NUM_CONCEPTS - 1}]"
        )
    for c in correct:
        assert c in (0, 1), f"BUG: user {user_id} has non-binary correct value {c}"
    return user_id, exercise_idx, correct


def build_row(user_id, exercise_idx, correct, fold):
    """One row of the pre-generate_sequences() dataframe. 'concepts' and
    'responses' are comma-joined strings -- save_dcur() (inside pyKT's
    generate_sequences/generate_window_sequences) splits them back into
    python lists internally."""
    return {
        "fold": fold,
        "uid": str(user_id),
        "concepts": ",".join(str(e) for e in exercise_idx),
        "responses": ",".join(str(c) for c in correct),
    }


def build_keyid2idx():
    """
    Identity mapping, deliberately NOT pyKT's own id_mapping(): our
    exercise_idx values already ARE the final 0-834 indices, so this just
    documents that fact for pyKT's config/loading code, not remap
    anything.
    """
    return {
        "concepts": {str(i): i for i in range(NUM_CONCEPTS)},
        "max_concepts": 1,   # single concept per exercise, never multi-concept
    }


def build_data_config_entry(dpath):
    return {
        DATASET_NAME: {
            "dpath": dpath,
            "num_q": 0,
            "num_c": NUM_CONCEPTS,
            "input_type": ["concepts"],
            "max_concepts": 1,
            "min_seq_len": MIN_SEQ_LEN,
            "maxlen": MAXLEN,
            "emb_path": "",
            "train_valid_original_file": "train_valid.csv",
            "train_valid_file": "train_valid_sequences.csv",
            "folds": [0, 1],
            "test_original_file": "test.csv",
            "test_file": "test_sequences.csv",
            "test_window_file": "test_window_sequences.csv",
        }
    }


def build_train_valid_sequences(train_ids, val_ids, pkg_dir):
    rows = []
    for uid in train_ids:
        user_id, ex, co = load_pkg_sequence(os.path.join(pkg_dir, f"pkg_{uid}.pt"))
        rows.append(build_row(user_id, ex, co, fold=0))
    for uid in val_ids:
        user_id, ex, co = load_pkg_sequence(os.path.join(pkg_dir, f"pkg_{uid}.pt"))
        rows.append(build_row(user_id, ex, co, fold=1))
    df = pd.DataFrame(rows)
    return generate_sequences(
        df, TRAIN_VALID_EFFECTIVE_KEYS, min_seq_len=MIN_SEQ_LEN, maxlen=MAXLEN
    )


def build_test_sequences(test_ids, pkg_dir):
    rows = []
    for uid in test_ids:
        user_id, ex, co = load_pkg_sequence(os.path.join(pkg_dir, f"pkg_{uid}.pt"))
        rows.append(build_row(user_id, ex, co, fold=-1))
    df = pd.DataFrame(rows)
    chunked = generate_sequences(
        df, TEST_EFFECTIVE_KEYS, min_seq_len=MIN_SEQ_LEN, maxlen=MAXLEN
    )
    windowed = generate_window_sequences(df, TEST_EFFECTIVE_KEYS, maxlen=MAXLEN)
    return chunked, windowed


# ── everything from here down takes ONLY staged_input_dir/output_dir --
#    fedgkt_root is not a parameter anywhere below this line, by
#    construction, not by convention ──────────────────────────────────
def build_all_sequences(staged_input_dir, output_dir):
    with open(os.path.join(staged_input_dir, "student_splits.json")) as f:
        splits = json.load(f)
    pkg_dir = os.path.join(staged_input_dir, "pkgs")
    os.makedirs(output_dir, exist_ok=True)

    train_valid_seqs = build_train_valid_sequences(splits["train"], splits["val"], pkg_dir)
    train_valid_seqs.to_csv(os.path.join(output_dir, "train_valid_sequences.csv"), index=None)

    test_chunked, test_windowed = build_test_sequences(splits["test"], pkg_dir)
    test_chunked.to_csv(os.path.join(output_dir, "test_sequences.csv"), index=None)
    test_windowed.to_csv(os.path.join(output_dir, "test_window_sequences.csv"), index=None)

    with open(os.path.join(output_dir, "keyid2idx.json"), "w") as f:
        json.dump(build_keyid2idx(), f, indent=2)

    print(f"train_valid_sequences.csv: {len(train_valid_seqs)} rows")
    print(f"test_sequences.csv: {len(test_chunked)} rows")
    print(f"test_window_sequences.csv: {len(test_windowed)} rows")
    return train_valid_seqs, test_chunked, test_windowed


def run_real(fedgkt_root, staging_root):
    """The real entry point. fedgkt_root is used exactly once, right
    here, to call stage_input_data() -- then it is never referenced
    again for the rest of the run."""
    staged_input_dir = stage_input_data(fedgkt_root, staging_root)
    output_dir = os.path.join(staging_root, "pykt_sequences")
    return build_all_sequences(staged_input_dir, output_dir)


def run_self_test():
    print("=" * 70)
    print("baseline_step1_build_pykt_sequences.py -- self-test against REAL data")
    print("Using the 3 known-good real students (233536, 45224, 21419),")
    print("in their ACTUAL locked split roles (233536: train, 45224: train,")
    print("21419: val). ALSO exercises stage_input_data() itself, via a")
    print("fake fedgkt_root built in a temp directory -- not skipped or")
    print("bypassed, since that's the actual new logic being verified now.")
    print("=" * 70)

    real_pkg_dir = os.path.dirname(os.path.abspath(__file__))  # the 3 .pt files live flat here

    with tempfile.TemporaryDirectory() as tmp:
        fake_fedgkt_root = os.path.join(tmp, "fedgkt")
        fake_staging_root = os.path.join(tmp, "fedgkt_baselines_staging")
        fake_processed = os.path.join(fake_fedgkt_root, "data", "processed")
        fake_pkgs = os.path.join(fake_processed, "pkgs")
        os.makedirs(fake_pkgs, exist_ok=True)

        # ---- build the FAKE fedgkt_root source, using only real data ----
        for uid in (233536, 45224, 21419):
            shutil.copy2(
                os.path.join(real_pkg_dir, f"pkg_{uid}.pt"),
                os.path.join(fake_pkgs, f"pkg_{uid}.pt"),
            )

        # self-test-only splits file: real roles for 233536/45224/21419,
        # PLUS 233536 reused as a pretend test-split member -- clearly
        # NOT its real role, used only to exercise the separate
        # test-file code path structurally (documented explicitly, same
        # as in the original self-test).
        fake_splits = {"train": [233536, 45224], "val": [21419], "test": [233536]}
        with open(os.path.join(fake_processed, "student_splits.json"), "w") as f:
            json.dump(fake_splits, f)

        real_vocab_path = os.path.join(
            os.path.dirname(real_pkg_dir), "exercise_vocab.json"
        )
        if os.path.isfile(real_vocab_path):
            shutil.copy2(real_vocab_path, os.path.join(fake_processed, "exercise_vocab.json"))
        else:
            # exercise_vocab.json isn't actually read by this script's
            # logic (exercise_idx values in the .pt files are already
            # final indices) -- it's staged only so downstream steps
            # (e.g. GKT's graph) have it available. A minimal stand-in
            # is fine for this self-test if the real file isn't present
            # at this path in the sandbox.
            with open(os.path.join(fake_processed, "exercise_vocab.json"), "w") as f:
                json.dump({"num_exercises": NUM_CONCEPTS}, f)

        # ---- 1. run the REAL entry point end-to-end, including staging ----
        print("\n--- Running stage_input_data() + build_all_sequences() end-to-end ---")
        train_valid_seqs, test_chunked, test_windowed = run_real(fake_fedgkt_root, fake_staging_root)

        staged_input_dir = os.path.join(fake_staging_root, "staged_input")
        output_dir = os.path.join(fake_staging_root, "pykt_sequences")

        # ---- 2. verify the staged copy is faithful and complete ----------
        print("\n--- Verifying staged_input/ matches the source exactly ---")
        for uid in (233536, 45224, 21419):
            src = os.path.join(fake_pkgs, f"pkg_{uid}.pt")
            dst = os.path.join(staged_input_dir, "pkgs", f"pkg_{uid}.pt")
            assert os.path.isfile(dst), f"BUG: {dst} was not staged"
            assert os.path.getsize(src) == os.path.getsize(dst), (
                f"BUG: staged copy of pkg_{uid}.pt has different size than source"
            )
        assert os.path.isfile(os.path.join(staged_input_dir, "student_splits.json"))
        assert os.path.isfile(os.path.join(staged_input_dir, "exercise_vocab.json"))
        print("  All 3 pkg files + student_splits.json + exercise_vocab.json "
              "staged correctly, byte-size-verified against source -- OK")

        # ---- 3. re-run staging a second time, confirm idempotency --------
        print("\n--- Re-running stage_input_data() a second time (idempotency check) ---")
        stage_input_data(fake_fedgkt_root, fake_staging_root)
        n_pkg_files_after = len(
            [f for f in os.listdir(os.path.join(staged_input_dir, "pkgs")) if f.endswith(".pt")]
        )
        assert n_pkg_files_after == 3, (
            f"BUG: re-running staging changed the pkg file count "
            f"(expected 3, got {n_pkg_files_after}) -- not idempotent"
        )
        print("  Second run did not duplicate or corrupt staged files -- OK")

        # ---- 4. train_valid_sequences.csv row counts / fold assignment ---
        print("\n--- Checking train_valid_sequences.csv ---")
        print(train_valid_seqs[["fold", "uid"]].to_string(index=False))
        expected_counts = {"233536": 1, "45224": 1, "21419": 48}
        actual_counts = train_valid_seqs["uid"].value_counts().to_dict()
        print(f"Expected row counts per student: {expected_counts}")
        print(f"Actual row counts per student:   {actual_counts}")
        for uid_str, expected in expected_counts.items():
            assert actual_counts.get(uid_str) == expected, (
                f"BUG: student {uid_str} expected {expected} sequence rows, "
                f"got {actual_counts.get(uid_str)}"
            )
        assert len(train_valid_seqs) == 50, f"BUG: expected 50 total rows, got {len(train_valid_seqs)}"
        print("Row counts match hand-derived expectations exactly -- OK")

        fold_by_uid = dict(zip(train_valid_seqs["uid"], train_valid_seqs["fold"]))
        assert fold_by_uid["233536"] == 0 and fold_by_uid["45224"] == 0
        assert fold_by_uid["21419"] == 1
        print("Fold assignment matches real split membership -- OK")

        # ---- 5. exact content round-trip for the short student -----------
        print("\n--- Verifying exact content round-trip for student 233536 ---")
        real_ex, real_co = load_pkg_sequence(os.path.join(real_pkg_dir, "pkg_233536.pt"))[1:]
        row = train_valid_seqs[train_valid_seqs["uid"] == "233536"].iloc[0]
        concepts_out = [int(x) for x in row["concepts"].split(",") if x != "-1"]
        responses_out = [int(x) for x in row["responses"].split(",") if x != "-1"]
        selectmask_out = [int(x) for x in row["selectmasks"].split(",")]
        assert concepts_out == list(real_ex), "BUG: concepts round-trip mismatch for 233536"
        assert responses_out == list(real_co), "BUG: responses round-trip mismatch for 233536"
        assert selectmask_out.count(1) == 30 and selectmask_out.count(-1) == 170, (
            "BUG: selectmask for 233536 should be 30x'1' + 170x'-1'"
        )
        print("  Exact exercise_idx/correct sequences recovered unchanged, "
              "padding correct (30 real + 170 pad) -- OK")

        # ---- 6. test_sequences.csv / test_window_sequences.csv -----------
        print("\n--- Checking test_sequences.csv / test_window_sequences.csv ---")
        print("NOTE: 233536 is reused here as a PRETEND test-split member "
              "(its real role is train) purely to exercise this separate "
              "code path -- not a real test-split run.")
        assert set(test_chunked["fold"].unique()) == {-1}
        assert set(test_windowed["fold"].unique()) == {-1}
        assert len(test_chunked) == 1 and len(test_windowed) == 1, (
            f"BUG: expected 1 row each (233536 has 30 <= maxlen), "
            f"got chunked={len(test_chunked)} windowed={len(test_windowed)}"
        )
        print("  fold=-1 correctly tagged, 1 row each as expected for a "
              "30-interaction student (<=maxlen, no chunking/windowing "
              "needed) -- OK")

        # ---- 7. output files actually written to pykt_sequences/ ---------
        print("\n--- Checking pykt_sequences/ output files exist ---")
        for fname in ("train_valid_sequences.csv", "test_sequences.csv",
                      "test_window_sequences.csv", "keyid2idx.json"):
            fpath = os.path.join(output_dir, fname)
            assert os.path.isfile(fpath), f"BUG: {fpath} was not written"
        print("  All 4 output files present in pykt_sequences/ -- OK")

        # ---- 8. keyid2idx.json / data_config entry sanity -----------------
        print("\n--- keyid2idx.json / data_config entry ---")
        with open(os.path.join(output_dir, "keyid2idx.json")) as f:
            k2i = json.load(f)
        assert len(k2i["concepts"]) == NUM_CONCEPTS
        assert k2i["concepts"]["0"] == 0 and k2i["concepts"]["834"] == 834
        assert k2i["max_concepts"] == 1
        print(f"  keyid2idx.json: identity mapping over {len(k2i['concepts'])} "
              f"concepts, max_concepts=1 -- OK")

        cfg_entry = build_data_config_entry(dpath="dummy/path")
        assert cfg_entry[DATASET_NAME]["num_q"] == 0
        assert cfg_entry[DATASET_NAME]["num_c"] == NUM_CONCEPTS
        assert cfg_entry[DATASET_NAME]["input_type"] == ["concepts"]
        assert cfg_entry[DATASET_NAME]["folds"] == [0, 1]
        print("  data_config entry: num_q=0, num_c=835, input_type=['concepts'], "
              "folds=[0,1] -- OK (train with --fold 1 -> val=fold1, train=fold0)")

    print("\n" + "=" * 70)
    print("Self-test complete -- no assertion failures.")
    print("stage_input_data() verified: faithful copy, idempotent re-run,")
    print("and structurally the only function that ever sees fedgkt_root.")
    print("Sequence-building verified against 3 real students in their")
    print("real split roles. A full real run still requires the complete")
    print("pkgs/ folder for all 1000 working-subset students.")
    print("=" * 70)


if __name__ == "__main__":
    _SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    # Matches the confirmed layout: this script lives inside
    # fedgkt_baselines_staging/, which sits as a SIBLING to fedgkt/ --
    # so fedgkt/ is just "one level up, then into fedgkt/" from here,
    # and the staging root is simply wherever this script itself lives.
    _DEFAULT_FEDGKT_ROOT = os.path.normpath(os.path.join(_SCRIPT_DIR, "..", "fedgkt"))
    _DEFAULT_STAGING_ROOT = _SCRIPT_DIR

    parser = argparse.ArgumentParser(
        description="Build pyKT-format sequence CSVs from FedGKT's locked data."
    )
    parser.add_argument(
        "--fedgkt-root", type=str, default=_DEFAULT_FEDGKT_ROOT,
        help=f"Path to the real, locked fedgkt/ project root (read-only). "
             f"Default: {_DEFAULT_FEDGKT_ROOT} (assumes the confirmed sibling layout).",
    )
    parser.add_argument(
        "--staging-root", type=str, default=_DEFAULT_STAGING_ROOT,
        help=f"Path to fedgkt_baselines_staging/. Default: {_DEFAULT_STAGING_ROOT} "
             f"(wherever this script itself is running from).",
    )
    parser.add_argument(
        "--selftest", action="store_true",
        help="Run the self-test (3 known students) instead of the real conversion.",
    )
    args = parser.parse_args()

    if args.selftest:
        run_self_test()
    else:
        if not os.path.isdir(args.fedgkt_root):
            raise SystemExit(
                f"fedgkt/ not found at the expected default location:\n"
                f"  {args.fedgkt_root}\n"
                f"If your fedgkt/ folder isn't a direct sibling of this script's "
                f"folder, pass its real path explicitly with --fedgkt-root."
            )
        run_real(args.fedgkt_root, args.staging_root)