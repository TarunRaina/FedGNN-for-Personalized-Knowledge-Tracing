"""
train_baseline.py

Root-level entry point for Phase 1 baseline training. Run this to actually
train FedGKT on the real working subset (800 train / 100 val / 100 test
students, per student_splits.json).

SAFETY DESIGN (a real multi-hour run on a personal laptop has a real risk
of interruption -- sleep, accidental terminal close, crash):

1. On a FRESH start, this script trains exactly ONE real epoch first, times
   it, and prints a projected total based on that REAL measurement (not
   speculation) -- then asks for explicit confirmation before committing to
   the remaining epochs.

2. If a saved training state already exists (from a previous run that was
   deliberately stopped after step 1, OR that was interrupted mid-training),
   this script detects it automatically and resumes directly -- no
   re-timing, no re-asking. The timing/confirmation gate only needs to be
   passed once, ever, for a given training run.

3. Progress is checkpointed after every epoch (see src/training/centralised.py),
   so rerunning this script after ANY interruption -- whether you stopped it
   yourself or it crashed -- picks up from the last completed epoch, not
   from scratch.

Usage:
    python train_baseline.py            # asks for confirmation after epoch 1 timing
    python train_baseline.py --yes      # skips the confirmation prompt (for automation)
"""

import os
import sys
import json
import time

import torch

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))  # fedgkt/
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

from src.utils import config as cfg
from src.models.fedgkt import FedGKT
from src.training.centralised import train
from src.training.evaluator import evaluate


def load_splits():
    assert os.path.exists(cfg.SPLITS_PATH), f"student_splits.json not found at {cfg.SPLITS_PATH}"
    with open(cfg.SPLITS_PATH) as f:
        splits = json.load(f)
    return splits['train'], splits['val'], splits['test']


def main():
    auto_confirm = '--yes' in sys.argv

    print("=" * 70)
    print("FedGKT Phase 1 -- Baseline Training")
    print("=" * 70)

    train_ids, val_ids, test_ids = load_splits()
    print(f"\nLoaded splits: train={len(train_ids)}  val={len(val_ids)}  test={len(test_ids)}")

    assert os.path.exists(cfg.EDGE_INDEX_PATH), f"edge_index.pt not found at {cfg.EDGE_INDEX_PATH}"
    edge_index = torch.load(cfg.EDGE_INDEX_PATH, weights_only=False)

    resume_path = os.path.join(cfg.CHECKPOINT_DIR, 'latest_state.pt')

    if os.path.exists(resume_path):
        # ── a previous run already exists -- either it was deliberately
        # stopped after the epoch-1 timing step, or it was interrupted
        # mid-training. Either way, the timing/confirmation gate was already
        # passed (or is irrelevant, since real progress already exists) --
        # skip straight to continuing, no re-asking. ──────────────────────
        saved_preview = torch.load(resume_path, weights_only=False)
        saved_epoch = saved_preview['epoch']
        saved_stopped_early = saved_preview.get('stopped_early', False)

        print(f"\nFound existing training state: {saved_epoch} epoch(s) already "
              f"completed in a previous run (stopped_early={saved_stopped_early}).")
        print("Skipping the timing/confirmation step -- resuming directly.")

        model = FedGKT()  # init weights irrelevant -- overwritten by resume_from below
        result = train(
            model, train_ids=train_ids, val_ids=val_ids, edge_index=edge_index,
            num_epochs=cfg.NUM_EPOCHS, resume_from=resume_path, verbose=True,
        )
    else:
        # ── fresh start ────────────────────────────────────────────────────
        torch.manual_seed(cfg.RANDOM_SEED)
        model = FedGKT()
        n_params = sum(p.numel() for p in model.parameters())
        print(f"\nFresh start. Model initialised: {n_params:,} parameters")

        print(f"\n--- Timing one real epoch on the FULL train/val sets ---")
        print(f"Training 1 epoch over {len(train_ids)} students, validating on "
              f"{len(val_ids)} students. This is a REAL measurement, not an estimate.")

        epoch1_start = time.time()
        train(
            model, train_ids=train_ids, val_ids=val_ids, edge_index=edge_index,
            num_epochs=1, verbose=True,
        )
        epoch1_elapsed = time.time() - epoch1_start

        remaining_epochs = cfg.NUM_EPOCHS - 1
        projected_remaining_min = epoch1_elapsed * remaining_epochs / 60
        projected_total_min = epoch1_elapsed * cfg.NUM_EPOCHS / 60

        print(f"\nOne epoch took {epoch1_elapsed / 60:.1f} minutes ({epoch1_elapsed:.1f}s).")
        print(f"Projected time for remaining {remaining_epochs} epochs "
              f"(early stopping may end it sooner): ~{projected_remaining_min:.1f} minutes")
        print(f"Projected TOTAL time if all {cfg.NUM_EPOCHS} epochs run: "
              f"~{projected_total_min:.1f} minutes (~{projected_total_min / 60:.1f} hours)")

        if not auto_confirm:
            response = input(f"\nContinue training for the remaining {remaining_epochs} "
                              f"epochs? [y/n]: ").strip().lower()
            if response != 'y':
                print("\nStopping after 1 epoch, as requested. Progress is saved -- "
                      "just rerun this script anytime to resume automatically from here.")
                return
        else:
            print("\n--yes flag detected, continuing automatically without confirmation.")

        print(f"\n--- Continuing training for up to {remaining_epochs} more epochs ---")
        result = train(
            model, train_ids=train_ids, val_ids=val_ids, edge_index=edge_index,
            num_epochs=cfg.NUM_EPOCHS, resume_from=resume_path, verbose=True,
        )

    # ── final test evaluation -- the FIRST and ONLY use of test_ids in this
    # entire project. Never touched during training or per-epoch validation. ──
    print(f"\n--- Final evaluation on TEST set (first use in this entire project) ---")
    print(f"Evaluating the best checkpoint on {len(test_ids)} held-out test students.")
    test_result = evaluate(model, test_ids, edge_index=edge_index, verbose=False)

    print(f"\nTest macro AUC:        {test_result['macro_auc']:.4f} "
          f"({test_result['macro_n_valid']}/{test_result['macro_n_total']} valid)")
    print(f"Test overall BCE:      {test_result['overall_bce']:.4f}")
    print(f"Test concepts w/ AUC:  {test_result['n_concepts_with_auc']} "
          f"(skipped: {test_result['n_concepts_skipped']})")

    final_results = {
        'best_epoch': result['best_epoch'],
        'best_val_auc': result['best_val_auc'],
        'stopped_early': result['stopped_early'],
        'test_macro_auc': test_result['macro_auc'],
        'test_macro_n_valid': test_result['macro_n_valid'],
        'test_macro_n_total': test_result['macro_n_total'],
        'test_overall_bce': test_result['overall_bce'],
        'test_n_concepts_with_auc': test_result['n_concepts_with_auc'],
        'test_n_concepts_skipped': test_result['n_concepts_skipped'],
        'test_per_concept_auc': test_result['per_concept_auc'],
    }
    results_path = os.path.join(cfg.CHECKPOINT_DIR, 'final_results.json')
    with open(results_path, 'w') as f:
        json.dump(final_results, f, indent=2)

    print(f"\nSaved final results: {results_path}")
    print(f"Best model checkpoint: {result['checkpoint_path']}")
    print("\n" + "=" * 70)
    print("Baseline training complete.")
    print("=" * 70)


if __name__ == '__main__':
    main()