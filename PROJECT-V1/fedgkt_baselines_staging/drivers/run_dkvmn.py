"""
drivers/run_dkvmn.py

Trains the DKVMN baseline and scores it on the test set. Same structure
as run_dkt.py -- only the model and its settings differ. Meant to be run
in Colab with a GPU runtime.

All shared behaviour -- early stopping on validation macro AUC, resume
after disconnect, student matching, macro + pooled test scoring -- lives
in common/train_utils.py.

SETTINGS (verified against pyKT 0.0.38, not recalled)
-----------------------------------------------------
examples/wandb_dkvmn_train.py (argparse defaults) and
configs/kt_config.json -> "dkvmn" agree on every value:

      dim_s          200   (key/value memory embedding size)
      size_m          50   (number of memory slots)
      dropout        0.2
      learning_rate  1e-3

  batch_size          64   wandb_train.py lowers the kt_config default
                           of 256 to 64 for dkvmn ("because of OOM")
  emb_type         'qid'   pyKT's default

Constructor, pykt/models/dkvmn.py:
    DKVMN(num_c, dim_s, size_m, dropout=0.2, emb_type='qid', emb_path="")
so MODEL_CONFIG below maps onto it exactly.

These are pyKT's own defaults, used untuned -- same policy as every
baseline. Justify them in the paper as "pyKT's defaults".

Shared parity settings (patience 5 on validation macro AUC, max 100
epochs, no shuffling, seed 42) are identical to every other baseline.

USAGE (in Colab, from the staging folder)
-----------------------------------------
    python drivers/run_dkvmn.py --smoke    # 2 epochs, into dkvmn_smoke/
    python drivers/run_dkvmn.py            # the real run; resumes if disconnected

Results go to  <run-root>/dkvmn/  (or dkvmn_smoke/ for --smoke).
"""

import argparse
import os
import sys

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))          # drivers/
STAGING_ROOT = os.path.dirname(_THIS_DIR)                        # staging root
COMMON_DIR = os.path.join(STAGING_ROOT, 'common')
if COMMON_DIR not in sys.path:
    sys.path.insert(0, COMMON_DIR)

import data_config as dc          # noqa: E402
import train_utils as tu          # noqa: E402
from pykt.models import init_model  # noqa: E402


MODEL_NAME = 'dkvmn'

# pyKT 0.0.38 defaults for DKVMN -- see module docstring for sources
MODEL_CONFIG = {
    'dim_s': 200,
    'size_m': 50,
    'dropout': 0.2,
}
BATCH_SIZE = 64
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
    parser = argparse.ArgumentParser(description="Train the DKVMN baseline.")
    parser.add_argument('--smoke', action='store_true',
                        help="Quick 2-epoch check into a separate dkvmn_smoke folder.")
    parser.add_argument('--run-root', default=DEFAULT_RUN_ROOT,
                        help="Parent folder for run outputs (Google Drive in Colab).")
    parser.add_argument('--sequences-dir', default=DEFAULT_SEQUENCES_DIR,
                        help="Folder holding the three pyKT sequence CSVs.")
    parser.add_argument('--skip-verify', action='store_true',
                        help="Skip the structural check of the CSVs (not recommended).")
    args = parser.parse_args()

    # Refuse to write to an unmounted Drive path -- Colab would otherwise
    # create it on its own disk and every result would vanish at disconnect.
    if args.run_root == DEFAULT_RUN_ROOT:
        assert os.path.isdir('/content/drive/MyDrive'), (
            "Google Drive is not mounted, so results would be lost when "
            "Colab disconnects. Mount it first:\n"
            "    from google.colab import drive\n"
            "    drive.mount('/content/drive')"
        )

    # Smoke run gets its own folder, so the real run can never "resume"
    # from the 2-epoch test.
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
    print(f"DKVMN {'smoke test' if args.smoke else 'run'} complete.")
    print(f"  best epoch     : {results['best_epoch']}")
    print(f"  test macro AUC : {results['test_macro_auc']:.4f}")
    print(f"  test pooled AUC: {results['test_pooled_auc']:.4f}")
    print("=" * 70)


if __name__ == '__main__':
    main()