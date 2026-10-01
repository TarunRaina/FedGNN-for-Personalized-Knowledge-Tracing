"""
analyse_v2.py

Two analyses of the finished v2 model. No GPU, no training, no model
construction -- it reads the checkpoint's numbers directly and the result
files already written. Runs fine on the laptop.

    1. LENGTH BANDS
       v2's test macro AUC split by how many interactions a student has
       (30-100 / 100-500 / 500+), next to v1 and the five baselines.
       Answers: is the +0.045 spread evenly, or concentrated?

    2. LEARNED PRIORS
       The 835 per-concept cold-start values the model learned, checked
       against how hard each concept actually is in the training data.
       Answers: did the prior learn real difficulty, or just noise that
       happened to help?

WHY (2) MATTERS MORE THAN IT LOOKS
-----------------------------------
The prior replaced a hand-set constant (0.5 for every concept) with a
learned value. If those values correlate with the concepts students
actually get wrong, that is:

  - evidence the mechanism works for the reason claimed, not by accident
  - an interpretability result no baseline can produce -- DKT's 200-dim
    embeddings cannot be read this way
  - a figure for the paper

If they DON'T correlate, that is worth knowing too, and should be
reported rather than quietly dropped: it would mean the prior helped for
some other reason, and the interpretability claim would need softening.

The empirical difficulty is computed from the TRAINING students only.
Using validation or test students to judge the model would be circular.

INPUTS
------
  --test-results  <run_dir>/test_results_test.json   (from the test cell)
  --checkpoint    <run_dir>/best.pt                  (v2 weights)
  --vocab         data/processed/exercise_vocab.json
  --splits        data/processed/student_splits.json
  --pkg-dir       data/processed/pkgs/

Download best.pt and test_results_test.json from Drive first; everything
else is already in fedgkt_v2/.

USAGE
-----
    python analyse_v2.py --run-dir <folder holding best.pt and the json>

    # skip the slow part (reading 800 students) if you only want the priors listed
    python analyse_v2.py --run-dir <...> --no-empirical
"""

import argparse
import json
import os
import sys

import numpy as np
import torch

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

from src.utils import config as cfg

BANDS = ['30-100', '100-500', '500+']

# published test macro AUC by band, from baseline_step5_length_bands.py
REFERENCE_BANDS = {
    'FedGKT v1': [0.6663, 0.6996, 0.6892],
    'DKT':       [0.7411, 0.7605, 0.7634],
    'DKVMN':     [0.7600, 0.7551, 0.7557],
    'AKT':       [0.7225, 0.7550, 0.7566],
    'GKT':       [0.7422, 0.7427, 0.7265],
    'SAKT':      [0.7181, 0.7419, 0.7259],
}


def band_of(n):
    """[30,100), [100,500), [500,inf) -- the convention that reproduced
    FedGKT's published band scores."""
    if n < 100:
        return '30-100'
    if n < 500:
        return '100-500'
    return '500+'


# ── 1. length bands ──────────────────────────────────────────────────────
def length_bands(test_results_path):
    r = json.load(open(test_results_path))
    per_student = r['per_student']

    buckets = {b: [] for b in BANDS}
    counts = {b: 0 for b in BANDS}
    for s in per_student:
        b = band_of(s['n_interactions'])
        counts[b] += 1
        if s['auc_no_first'] is not None:      # single-class students excluded
            buckets[b].append(s['auc_no_first'])

    means = {b: (float(np.mean(v)) if v else None) for b, v in buckets.items()}

    print("=" * 72)
    print("1. TEST MACRO AUC BY SEQUENCE LENGTH")
    print("=" * 72)
    print(f"\n  {'':<14}" + "".join(f"{b:>12}" for b in BANDS))
    print(f"  {'students':<14}" + "".join(f"{counts[b]:>12}" for b in BANDS))
    print()
    print(f"  {'FedGKT v2':<14}" + "".join(
        f"{means[b]:>12.4f}" if means[b] is not None else f"{'-':>12}" for b in BANDS))
    for name, vals in REFERENCE_BANDS.items():
        print(f"  {name:<14}" + "".join(f"{v:>12.4f}" for v in vals))

    v1 = REFERENCE_BANDS['FedGKT v1']
    print(f"\n  {'v2 - v1':<14}" + "".join(
        f"{means[b] - v1[i]:>+12.4f}" for i, b in enumerate(BANDS)))
    print(f"\n  overall test macro AUC (no-first): {r['macro_auc_no_first']:.4f}")
    print("\n  If the v2 - v1 row is roughly flat, the gain is uniform across")
    print("  student lengths. If it is larger in one band, say which and why --")
    print("  the learned prior should in principle help SHORT sequences most,")
    print("  since that is where untouched concepts dominate.")
    return means, counts


# ── 2. learned priors ────────────────────────────────────────────────────
def learned_priors(checkpoint_path, vocab_path, splits_path, pkg_dir,
                   do_empirical=True, top_n=15, min_attempts=30):
    state = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    assert 'concept_prior_logit' in state, (
        f"No concept_prior_logit in {checkpoint_path} -- is this a v2 checkpoint "
        f"with the prior enabled? Keys: {sorted(state)[:8]}...")

    priors = torch.sigmoid(state['concept_prior_logit']).numpy()
    assert priors.shape == (cfg.NUM_NODES,), f"Unexpected shape {priors.shape}"

    print("\n" + "=" * 72)
    print("2. LEARNED PER-CONCEPT COLD-START PRIORS")
    print("=" * 72)
    print(f"\n  835 values, all initialised at 0.5 (sigmoid(0)) before training.")
    print(f"  range {priors.min():.4f} to {priors.max():.4f}   "
          f"mean {priors.mean():.4f}   std {priors.std():.4f}")
    moved = int((np.abs(priors - 0.5) > 0.01).sum())
    print(f"  moved more than 0.01 from 0.5: {moved} of {cfg.NUM_NODES} concepts")
    if moved < 50:
        print("  NOTE: few priors moved. The mechanism may be contributing less")
        print("  than the ablation suggests -- worth checking before claiming it.")

    names = {}
    if os.path.exists(vocab_path):
        vocab = json.load(open(vocab_path))
        idx_to_name = vocab.get('idx_to_name', {})
        names = {int(k): v for k, v in idx_to_name.items()}

    # ── empirical difficulty from the TRAINING students only ─────────
    emp_acc = np.full(cfg.NUM_NODES, np.nan)
    emp_n = np.zeros(cfg.NUM_NODES, dtype=int)
    if do_empirical:
        train_ids = json.load(open(splits_path))['train']
        print(f"\n  reading {len(train_ids)} training students for empirical difficulty...")
        correct_sum = np.zeros(cfg.NUM_NODES)
        for uid in train_ids:
            d = torch.load(os.path.join(pkg_dir, f'pkg_{uid}.pt'), weights_only=False)
            ex = d['exercise_idx'].numpy()
            co = d['correct'].numpy()
            np.add.at(correct_sum, ex, co)
            np.add.at(emp_n, ex, 1)
        seen = emp_n > 0
        emp_acc[seen] = correct_sum[seen] / emp_n[seen]
        print(f"  {int(seen.sum())} of {cfg.NUM_NODES} concepts appear in training data")

        # correlation, on concepts with enough attempts to be meaningful
        ok = emp_n >= min_attempts
        if ok.sum() >= 10:
            x, y = priors[ok], emp_acc[ok]
            pearson = float(np.corrcoef(x, y)[0, 1])
            rx = np.argsort(np.argsort(x)); ry = np.argsort(np.argsort(y))
            spearman = float(np.corrcoef(rx, ry)[0, 1])
            print(f"\n  --- Does the prior match real difficulty? ---")
            print(f"  concepts with >= {min_attempts} training attempts: {int(ok.sum())}")
            print(f"  Pearson  correlation (prior vs actual accuracy): {pearson:+.3f}")
            print(f"  Spearman correlation (rank agreement)          : {spearman:+.3f}")
            print()
            if spearman > 0.3:
                print("  POSITIVE and meaningful: the model learned which concepts are")
                print("  genuinely hard. This is a real interpretability result and a")
                print("  figure for the paper.")
            elif spearman > 0.1:
                print("  Weakly positive: some signal, but not a strong claim. Report")
                print("  the number rather than the interpretation.")
            else:
                print("  NOT positive. The prior helped the model (the ablation shows")
                print("  +0.019) but NOT by learning real concept difficulty. Report")
                print("  this honestly and drop the interpretability claim for it --")
                print("  a negative result stated plainly is better than an")
                print("  interpretation the data does not support.")
        else:
            print(f"\n  too few concepts with >= {min_attempts} attempts to correlate")

    # ── the table for the paper ──────────────────────────────────────
    order = np.argsort(priors)
    print(f"\n  --- {top_n} LOWEST priors (model considers hardest) ---")
    print(f"  {'idx':>5}{'prior':>9}{'actual acc':>12}{'attempts':>10}  concept")
    for i in order[:top_n]:
        acc = f"{emp_acc[i]:.3f}" if not np.isnan(emp_acc[i]) else "-"
        print(f"  {i:>5}{priors[i]:>9.4f}{acc:>12}{emp_n[i]:>10}  {names.get(int(i), '')[:44]}")

    print(f"\n  --- {top_n} HIGHEST priors (model considers easiest) ---")
    print(f"  {'idx':>5}{'prior':>9}{'actual acc':>12}{'attempts':>10}  concept")
    for i in order[-top_n:][::-1]:
        acc = f"{emp_acc[i]:.3f}" if not np.isnan(emp_acc[i]) else "-"
        print(f"  {i:>5}{priors[i]:>9.4f}{acc:>12}{emp_n[i]:>10}  {names.get(int(i), '')[:44]}")

    return priors, emp_acc, emp_n, names


def main():
    p = argparse.ArgumentParser(description="Length bands and learned priors for v2.")
    p.add_argument('--run-dir', required=True,
                   help="Folder holding best.pt and test_results_test.json.")
    p.add_argument('--checkpoint', default=None)
    p.add_argument('--test-results', default=None)
    p.add_argument('--no-empirical', action='store_true',
                   help="Skip reading the 800 training students (faster).")
    p.add_argument('--top-n', type=int, default=15)
    p.add_argument('--out-csv', default=None,
                   help="Write the per-concept table for plotting (default: "
                        "<run-dir>/learned_priors.csv)")
    args = p.parse_args()

    ckpt = args.checkpoint or os.path.join(args.run_dir, 'best.pt')
    testr = args.test_results or os.path.join(args.run_dir, 'test_results_test.json')
    vocab = os.path.join(cfg.PROCESSED_DIR, 'exercise_vocab.json')

    for path, what in [(ckpt, 'checkpoint'), (testr, 'test results')]:
        assert os.path.exists(path), f"{what} not found: {path}"

    length_bands(testr)
    priors, emp_acc, emp_n, names = learned_priors(
        ckpt, vocab, cfg.SPLITS_PATH, cfg.PKG_DIR,
        do_empirical=not args.no_empirical, top_n=args.top_n)

    out_csv = args.out_csv or os.path.join(args.run_dir, 'learned_priors.csv')
    with open(out_csv, 'w', encoding='utf-8') as f:
        f.write("concept_idx,learned_prior,empirical_accuracy,train_attempts,concept_name\n")
        for i in range(cfg.NUM_NODES):
            acc = '' if np.isnan(emp_acc[i]) else f"{emp_acc[i]:.6f}"
            nm = str(names.get(i, '')).replace(',', ' ')
            f.write(f"{i},{priors[i]:.6f},{acc},{emp_n[i]},{nm}\n")
    print(f"\nWritten: {out_csv}")
    print("(plot learned_prior against empirical_accuracy for the paper figure)")


if __name__ == '__main__':
    main()