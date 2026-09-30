"""
run_screen.py   (fedgkt_v2 root)

Runs one screening variant. Every v2 experiment goes through this, so
they are all trained and measured identically.

FIRST JOB: THE NOISE FLOOR
---------------------------
Before any change is evaluated, run the UNCHANGED model twice with
different seeds:

    python run_screen.py --variant baseline_seed42 --seed 42
    python run_screen.py --variant baseline_seed7  --seed 7

The difference between those two final validation scores is how much
this setup varies by chance alone. Any later "improvement" smaller than
that is noise, not a result. Nothing else should be interpreted until
this number exists.

THEN EACH VARIANT
-----------------
    python run_screen.py --variant lr3e-4 --lr 3e-4
    python run_screen.py --variant dropout0.1 --dropout 0.1

Each gets its own folder under --run-root, so nothing overwrites
anything, and each can resume independently after a disconnect.

SMOKE TEST FIRST
----------------
Before committing ~3 hours, check the pipeline runs end to end:

    python run_screen.py --variant smoke --epochs 2 --val-every 1

It writes to a 'smoke' folder, so it can never be confused with (or
resumed into) a real screen.

RESUME
------
If Colab disconnects, re-run the SAME command. It picks up from the next
epoch. Point --run-root at Google Drive so the state survives the
session ending.
"""

import argparse
import json
import os
import sys

import torch

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

from src.utils import config as cfg
from src.models.fedgkt import FedGKT
from src.models.fedgkt_v2 import FedGKTv2
from src.training.screen import run_screen, set_seed


DEFAULT_RUN_ROOT = '/content/drive/MyDrive/fedgkt_v2_screens'


def main():
    p = argparse.ArgumentParser(description="Run one v2 screening variant.")
    p.add_argument('--variant', required=True,
                   help="Name for this run. Its own folder under --run-root.")
    p.add_argument('--run-root', default=DEFAULT_RUN_ROOT)
    p.add_argument('--epochs', type=int, default=25,
                   help="Fixed budget, no early stopping (default 25).")
    p.add_argument('--val-every', type=int, default=2,
                   help="Validate every N epochs; the last epoch is always validated.")
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--lr', type=float, default=None,
                   help=f"Default: config.py's {cfg.LEARNING_RATE}")
    p.add_argument('--dropout', type=float, default=None,
                   help=f"Overrides config.py's {cfg.DROPOUT} BEFORE the model is built.")
    p.add_argument('--batch-size', type=int, default=8,
                   help="Training batch size. 8 matches the canonical run -- "
                        "changing it changes training behaviour, not just speed.")
    p.add_argument('--backward-chunk-size', type=int, default=50)
    p.add_argument('--eval-batch-size', type=int, default=16)
    p.add_argument('--model', default='v1', choices=['v1', 'v2', 'v2-prior', 'v2-skip'],
                   help="v1 = the original FedGKT (default). v2 = learned per-concept "
                        "cold-start prior + raw-feature skip. v2-prior / v2-skip run "
                        "one of the two alone, as an ablation.")
    p.add_argument('--patience', type=int, default=None,
                   help="None (default) = no early stopping, which is what SCREENING "
                        "wants: every variant gets the same budget. Set 5 to turn the "
                        "run into a FULL run matching the canonical protocol -- and "
                        "pass --val-every 1 with it, since patience counts VALIDATIONS.")
    p.add_argument('--cpu-eval', action='store_true',
                   help="Use the locked CPU evaluator instead of the batched GPU one "
                        "(~3x slower; for checking a final number).")
    args = p.parse_args()

    if args.run_root == DEFAULT_RUN_ROOT:
        assert os.path.isdir('/content/drive/MyDrive'), (
            "Google Drive is not mounted, so a disconnect would lose the run.\n"
            "    from google.colab import drive\n"
            "    drive.mount('/content/drive')")

    # dropout must be set before FedGKT is constructed -- gat.py and
    # fedgkt.py read cfg.DROPOUT at build time
    if args.dropout is not None:
        cfg.DROPOUT = args.dropout

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type != 'cuda':
        print("WARNING: no GPU. A screen on CPU will take many hours.")

    with open(cfg.SPLITS_PATH) as f:
        splits = json.load(f)
    train_ids, val_ids = splits['train'], splits['val']
    assert len(train_ids) == cfg.TRAIN_SIZE and len(val_ids) == cfg.VAL_SIZE, (
        f"Unexpected split sizes: train={len(train_ids)} val={len(val_ids)}")

    edge_index = torch.load(cfg.EDGE_INDEX_PATH, weights_only=False)
    run_dir = os.path.join(args.run_root, args.variant)

    print(f"variant   : {args.variant}")
    print(f"students  : train={len(train_ids)}  val={len(val_ids)}")
    print(f"dropout   : {cfg.DROPOUT}")
    print(f"run_dir   : {run_dir}")

    set_seed(args.seed)          # before building: weight init is random
    if args.model == 'v1':
        model = FedGKT()
    else:
        # FedGKTv2 builds the GAT and head FIRST, in v1's exact order, so with
        # the same seed every shared parameter starts IDENTICAL to v1's. That
        # makes this a paired comparison (~0.003 noise) rather than a
        # different-seed one (~0.008-0.012). Asserted in fedgkt_v2.py's self-test.
        model = FedGKTv2(use_prior=args.model in ('v2', 'v2-prior'),
                         use_skip=args.model in ('v2', 'v2-skip'))
    n_params = sum(p_.numel() for p_ in model.parameters())
    print(f"model     : {args.model}  ({n_params:,} parameters)")

    run_screen(
        model, train_ids, val_ids, run_dir,
        variant=args.variant,
        epochs=args.epochs,
        val_every=args.val_every,
        batch_size=args.batch_size,
        backward_chunk_size=args.backward_chunk_size,
        lr=args.lr,
        seed=args.seed,
        device=device,
        edge_index=edge_index,
        eval_batch_size=args.eval_batch_size,
        batched_eval=not args.cpu_eval,
        patience=args.patience,
    )


if __name__ == '__main__':
    main()