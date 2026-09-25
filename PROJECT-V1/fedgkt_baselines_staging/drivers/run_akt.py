"""
drivers/run_akt.py

Trains the AKT baseline and scores it on the test set. Same structure
as the other drivers -- only the model and its settings differ. Meant to
be run in Colab with a GPU runtime.

All shared behaviour -- early stopping on validation macro AUC, resume
after disconnect, student matching, macro + pooled test scoring -- lives
in common/train_utils.py.

SETTINGS (verified against pyKT 0.0.38, not recalled)
-----------------------------------------------------
For AKT, pyKT's two sources DISAGREE substantially:

                       wandb_akt_train.py    configs/kt_config.json
                       (argparse defaults)   -> "akt"
      n_blocks                 4                     1
      d_ff                   512                   256
      dropout                0.2                  0.05
      learning_rate         1e-4                  1e-5
      d_model                256                   256
      num_attn_heads           8               (not listed)

The ARGPARSE column is what actually reaches the model. pyKT's
wandb_train.py builds the model's settings from the command-line
params:

      model_config = copy.deepcopy(params)

and only ever reads kt_config.json for train_config (batch size, epochs,
optimiser). The model entries in kt_config.json are never used by the
training script. This is the same resolution as DKT, where kt_config
lists dropout 0.1 but the script's 0.2 is what reaches the model.

  batch_size            64   wandb_train.py lowers kt_config's 256 to 64
                             for akt ("because of OOM")
  emb_type           'qid'   pyKT's default

NOTE -- learning rate 1e-4 differs from FedGKT's 1e-3 and from the other
baselines' 1e-3. Under the agreed "pyKT's defaults, untuned" policy it is
correct: 1e-4 is genuinely what pyKT uses for AKT. Worth stating
explicitly in the methodology rather than implying a single learning
rate across all models.

pyKT's AKT script also defaults to seed 3407. The seed is kept at 42 here
because it is one of the shared parity settings applied identically to
every model.

Constructor parameters not exposed by pyKT's script (kq_same, final_fc_dim,
separate_qa, l2) are left at the constructor's own defaults, exactly as
pyKT's script leaves them.

QUESTION IDS -- AKT WITHOUT ITS RASCH TERM
-------------------------------------------
AKT normally combines concept ids with question ids, using the question
id for a Rasch-style difficulty term. We register num_q = 0 because
Junyi's exercise id is the finest identifier in our locked data.

Verified in pykt/models/akt.py: every use of question ids is guarded by
`if self.n_pid > 0:`. With n_pid = 0 the difficulty embedding is never
created, never used, and the extra regularisation loss is exactly 0.:

      if self.n_pid > 0: # have problem id
          ...
      else:
          c_reg_loss = 0.

The empty question tensor pyKT passes in is never read. So AKT runs
cleanly as a concept-only model -- the disclosure sentence already
agreed for the paper covers this: AKT's item-difficulty component cannot
operate in our setup, since no finer-grained identifier exists.

Shared parity settings (patience 5 on validation macro AUC, max 100
epochs, no shuffling, seed 42) are identical to every other baseline.

USAGE (in Colab, from the staging folder)
-----------------------------------------
    python drivers/run_akt.py --smoke    # 2 epochs, into akt_smoke/
    python drivers/run_akt.py            # the real run; resumes if disconnected

Results go to  <run-root>/akt/  (or akt_smoke/ for --smoke).
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


MODEL_NAME = 'akt'

# pyKT 0.0.38 defaults for AKT, from wandb_akt_train.py's argparse --
# see module docstring for why these and not kt_config.json's.
MODEL_CONFIG = {
    'd_model': 256,
    'd_ff': 512,
    'num_attn_heads': 8,
    'n_blocks': 4,
    'dropout': 0.2,
}
BATCH_SIZE = 64
LEARNING_RATE = 1e-4      # pyKT's AKT default -- differs from the other baselines
EMB_TYPE = 'qid'

# shared parity settings (identical across all five baselines)
MAX_EPOCHS = 100
PATIENCE = 5
MIN_DELTA = 0.0
SEED = 42

DEFAULT_RUN_ROOT = '/content/drive/MyDrive/fedgkt_baselines_runs'
DEFAULT_SEQUENCES_DIR = os.path.join(STAGING_ROOT, 'pykt_sequences')


def main():
    parser = argparse.ArgumentParser(description="Train the AKT baseline.")
    parser.add_argument('--smoke', action='store_true',
                        help="Quick 2-epoch check into a separate akt_smoke folder.")
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

    # AKT's question-difficulty term only exists when num_q > 0. Ours is 0,
    # so the model runs concept-only. Assert it rather than assume it.
    assert cfg['num_q'] == 0, (
        f"Expected num_q = 0 (concept-only registration), got {cfg['num_q']}."
    )

    tu.set_seed(SEED)   # seed BEFORE building the model: weight init is random
    model = init_model(MODEL_NAME, MODEL_CONFIG, cfg, EMB_TYPE)

    assert not hasattr(model, 'difficult_param'), (
        "BUG: AKT built a question-difficulty embedding despite num_q = 0."
    )
    print("AKT running concept-only: question-difficulty (Rasch) term disabled, "
          "as expected with num_q = 0.")

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
    print(f"AKT {'smoke test' if args.smoke else 'run'} complete.")
    print(f"  best epoch     : {results['best_epoch']}")
    print(f"  test macro AUC : {results['test_macro_auc']:.4f}")
    print(f"  test pooled AUC: {results['test_pooled_auc']:.4f}")
    print("=" * 70)


if __name__ == '__main__':
    main()