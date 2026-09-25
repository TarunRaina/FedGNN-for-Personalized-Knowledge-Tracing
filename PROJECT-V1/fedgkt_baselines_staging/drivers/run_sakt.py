"""
drivers/run_sakt.py

Trains the SAKT baseline and scores it on the test set. Same structure
as run_dkt.py / run_dkvmn.py -- only the model and its settings differ.
Meant to be run in Colab with a GPU runtime.

All shared behaviour -- early stopping on validation macro AUC, resume
after disconnect, student matching, macro + pooled test scoring -- lives
in common/train_utils.py.

SETTINGS (verified against pyKT 0.0.38, not recalled)
-----------------------------------------------------
examples/wandb_sakt_train.py (argparse defaults) and
configs/kt_config.json -> "sakt" agree on every value:

      emb_size         256
      num_attn_heads     8
      dropout          0.2
      num_en             1   (encoder layers)
      learning_rate   1e-3

  batch_size            64   wandb_train.py lowers kt_config's 256 to 64
                             for sakt ("because of OOM")
  emb_type           'qid'   pyKT's default

TWO THINGS THAT WOULD HAVE BEEN EASY TO GET WRONG
-------------------------------------------------
1. seq_len is REQUIRED by SAKT's constructor but appears in neither the
   argparse defaults nor kt_config.json. pyKT's wandb_train.py injects it:

       if model_name in ["saint","saint++", "sakt"]:
           model_config["seq_len"] = seq_len

   where seq_len comes from data_config[dataset]['maxlen'] ("prefer to use
   the maxlen in data config"). Ours is 200, so seq_len = 200 -- the size of
   SAKT's positional-embedding table.

2. num_en defaults to 2 in the constructor itself:

       SAKT(num_c, seq_len, emb_size, num_attn_heads, dropout, num_en=2, ...)

   but pyKT's own script and config both pass 1. Leaving it to the
   constructor default would silently train a deeper model than pyKT does,
   so it is set explicitly here.

These are pyKT's own defaults, used untuned -- same policy as every
baseline. Justify them in the paper as "pyKT's defaults".

Shared parity settings (patience 5 on validation macro AUC, max 100
epochs, no shuffling, seed 42) are identical to every other baseline.

USAGE (in Colab, from the staging folder)
-----------------------------------------
    python drivers/run_sakt.py --smoke    # 2 epochs, into sakt_smoke/
    python drivers/run_sakt.py            # the real run; resumes if disconnected

Results go to  <run-root>/sakt/  (or sakt_smoke/ for --smoke).
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


MODEL_NAME = 'sakt'

# pyKT 0.0.38 defaults for SAKT -- see module docstring for sources.
# seq_len is filled in from data_config at runtime (= maxlen, 200), exactly
# as pyKT's wandb_train.py does.
MODEL_CONFIG = {
    'emb_size': 256,
    'num_attn_heads': 8,
    'dropout': 0.2,
    'num_en': 1,
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
    parser = argparse.ArgumentParser(description="Train the SAKT baseline.")
    parser.add_argument('--smoke', action='store_true',
                        help="Quick 2-epoch check into a separate sakt_smoke folder.")
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

    # seq_len, exactly as pyKT's wandb_train.py supplies it for sakt
    model_config = {**MODEL_CONFIG, 'seq_len': cfg['maxlen']}
    assert model_config['seq_len'] == 200, (
        f"Expected seq_len 200 (our maxlen), got {model_config['seq_len']}"
    )

    tu.set_seed(SEED)   # seed BEFORE building the model: weight init is random
    model = init_model(MODEL_NAME, model_config, cfg, EMB_TYPE)

    results = tu.fit_and_evaluate(
        model, MODEL_NAME, cfg,
        batch_size=BATCH_SIZE,
        run_dir=run_dir,
        lr=LEARNING_RATE,
        max_epochs=max_epochs,
        patience=PATIENCE,
        min_delta=MIN_DELTA,
        seed=SEED,
        model_config={**model_config, 'emb_type': EMB_TYPE},
    )

    print("\n" + "=" * 70)
    print(f"SAKT {'smoke test' if args.smoke else 'run'} complete.")
    print(f"  best epoch     : {results['best_epoch']}")
    print(f"  test macro AUC : {results['test_macro_auc']:.4f}")
    print(f"  test pooled AUC: {results['test_pooled_auc']:.4f}")
    print("=" * 70)


if __name__ == '__main__':
    main()