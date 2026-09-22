"""
drivers/run_dkt.py

Trains the DKT baseline and scores it on the test set. Meant to be run
in Colab with a GPU runtime (pyKT's dataloader builds CUDA tensors and
will not run on CPU).

All shared behaviour -- early stopping on validation macro AUC, resume
after disconnect, student matching, macro + pooled test scoring -- lives
in common/train_utils.py. This file only says which model to train and
with which settings.

SETTINGS (verified against pyKT 0.0.38, not recalled)
-----------------------------------------------------
Taken from what pyKT's own DKT script does when run with no flags:

  examples/wandb_dkt_train.py  (argparse defaults)
      --emb_size       200
      --dropout        0.2
      --learning_rate  1e-3
  configs/kt_config.json -> train_config
      batch_size       256   (wandb_train.py only lowers this for
                              dkvmn/sakt/akt -> 64 and gkt -> 16; DKT
                              keeps 256)
  emb_type             'qid' (pyKT's default)

These are pyKT's standard settings, used untuned, per the no-tuning
policy agreed for all baselines. Justify them in the paper as "pyKT's
own defaults" -- NOT as "matched to FedGKT". (kt_config.json lists
dropout 0.1 for DKT, but the training script's argparse default of 0.2
is what actually reaches the model, since wandb_train.py builds
model_config from the command-line params.)

Learning rate 1e-3 happens to match FedGKT's config.py as well.

Deliberately OURS rather than pyKT's (shared across all baselines, set
in train_utils for parity with FedGKT): early stopping with patience 5
on validation macro AUC, max 100 epochs, no shuffling, seed 42. pyKT's
own script instead runs a fixed 200 epochs and picks its best epoch by
pooled AUC -- see train_utils.py for why that would not be a fair
comparison.

USAGE (in Colab, from the staging folder)
-----------------------------------------
    # quick check -- 2 epochs, into a SEPARATE folder (dkt_smoke)
    python drivers/run_dkt.py --smoke

    # the real run -- resumes automatically if Colab disconnected
    python drivers/run_dkt.py

Results go to  <run-root>/dkt/  (or dkt_smoke/ for --smoke).
"""

import argparse
import os
import sys

# ── make common/ importable ──────────────────────────────────────────────
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))          # drivers/
STAGING_ROOT = os.path.dirname(_THIS_DIR)                        # staging root
COMMON_DIR = os.path.join(STAGING_ROOT, 'common')
if COMMON_DIR not in sys.path:
    sys.path.insert(0, COMMON_DIR)

import data_config as dc          # noqa: E402
import train_utils as tu          # noqa: E402
from pykt.models import init_model  # noqa: E402


MODEL_NAME = 'dkt'

# pyKT 0.0.38 defaults for DKT -- see module docstring for sources
MODEL_CONFIG = {
    'emb_size': 200,
    'dropout': 0.2,
}
BATCH_SIZE = 256
LEARNING_RATE = 1e-3
EMB_TYPE = 'qid'

# shared parity settings (identical across all five baselines)
MAX_EPOCHS = 100
PATIENCE = 5
MIN_DELTA = 0.0
SEED = 42

DEFAULT_RUN_ROOT = '/content/drive/MyDrive/fedgkt_baselines_runs'
DEFAULT_SEQUENCES_DIR = os.path.join(STAGING_ROOT, 'pykt_sequences')


def main():
    parser = argparse.ArgumentParser(description="Train the DKT baseline.")
    parser.add_argument('--smoke', action='store_true',
                        help="Quick 2-epoch check into a separate dkt_smoke folder.")
    parser.add_argument('--run-root', default=DEFAULT_RUN_ROOT,
                        help="Parent folder for run outputs (Google Drive in Colab).")
    parser.add_argument('--sequences-dir', default=DEFAULT_SEQUENCES_DIR,
                        help="Folder holding the three pyKT sequence CSVs.")
    parser.add_argument('--skip-verify', action='store_true',
                        help="Skip the structural check of the CSVs (not recommended).")
    args = parser.parse_args()

    # Refuse to write to an unmounted Drive path. Without this, Colab would
    # happily create /content/drive/... on its own local disk, and every
    # result would vanish when the session ends.
    if args.run_root == DEFAULT_RUN_ROOT:
        assert os.path.isdir('/content/drive/MyDrive'), (
            "Google Drive is not mounted, so results would be lost when "
            "Colab disconnects. Mount it first:\n"
            "    from google.colab import drive\n"
            "    drive.mount('/content/drive')"
        )

    # The smoke run MUST use its own folder. If it wrote into dkt/, its
    # latest_model.pt would make the real run "resume" from the smoke run's
    # epoch 2 -- same settings, so nothing would stop it -- silently
    # splicing a 2-epoch test into the real result.
    folder = f'{MODEL_NAME}_smoke' if args.smoke else MODEL_NAME
    run_dir = os.path.join(args.run_root, folder)
    max_epochs = 2 if args.smoke else MAX_EPOCHS

    print(f"Sequences : {os.path.abspath(args.sequences_dir)}")
    print(f"Run folder: {run_dir}")
    print(f"Mode      : {'SMOKE TEST (2 epochs)' if args.smoke else 'full run'}")

    if not args.skip_verify:
        print("\n--- Checking the sequence CSVs ---")
        ok = dc.verify_sequences(args.sequences_dir)
        assert ok, "Sequence CSVs failed the structural check -- see above."

    cfg = dc.build_data_config(args.sequences_dir)[dc.DATASET_NAME]

    tu.set_seed(SEED)   # seed BEFORE building the model: weight init is random
    model = init_model(MODEL_NAME, MODEL_CONFIG, cfg, EMB_TYPE)

    results = tu.fit_and_evaluate(
        model, MODEL_NAME, cfg,
        batch_size=BATCH_SIZE,
        run_dir=run_dir,
        lr=LEARNING_RATE,
        max_epochs=max_epochs,
        patience=PATIENCE,
        min_delta=MIN_DELTA,
        seed=SEED,
        model_config={**MODEL_CONFIG, 'emb_type': EMB_TYPE},
    )

    print("\n" + "=" * 70)
    print(f"DKT {'smoke test' if args.smoke else 'run'} complete.")
    print(f"  best epoch     : {results['best_epoch']}")
    print(f"  test macro AUC : {results['test_macro_auc']:.4f}")
    print(f"  test pooled AUC: {results['test_pooled_auc']:.4f}")
    print("=" * 70)


if __name__ == '__main__':
    main()