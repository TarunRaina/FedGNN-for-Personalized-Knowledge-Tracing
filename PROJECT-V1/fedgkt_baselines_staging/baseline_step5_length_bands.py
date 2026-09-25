"""
baseline_step5_length_bands.py

Compares FedGKT against each baseline BY STUDENT SEQUENCE LENGTH, using
the per-student AUCs both already record. No GPU, no retraining -- it
only reads result files.

WHY
---
FedGKT's headline test macro AUC (0.6820) is one number averaged over 100
students of wildly different lengths -- 30 interactions to 2,440 in the
test split. FedGKT's own canonical result reported three length bands:

    30-100   0.6646
    100-500  0.6988
    500+     0.6894

One of the project's claimed contributions is handling realistic,
long-tailed data. That claim is about the BANDS, not the average, and it
has never been checked against the baselines. This file checks it.

INPUTS
------
  fedgkt_reeval/test_reeval.json     per-student AUC + n_interactions
                                     (from baseline_step3_fedgkt_reeval.py)
  <results-dir>/*.json               one results.json per baseline, each
                                     holding test_per_student_auc
                                     (download them from Drive:
                                      fedgkt_baselines_runs/<model>/results.json,
                                      renamed or kept in <model>/ subfolders)

Student ids are matched across files, so ordering does not matter.
Students whose AUC is null (single-class -- all right or all wrong) are
excluded from every model's average, exactly as the locked macro AUC
convention requires.

BAND BOUNDARIES
---------------
FedGKT's published bands are stated as 30-100 / 100-500 / 500+, which is
ambiguous about which side 100 and 500 fall on. Rather than guess, this
script recomputes FedGKT's OWN band scores under both conventions and
prints them next to the published values, so the matching one is
visible. Whichever matches is then the convention used for every model --
so the comparison is on FedGKT's own terms, not a convention invented
here.

USAGE (locally, from fedgkt_baselines_staging/)
-----------------------------------------------
    python baseline_step5_length_bands.py --results-dir baseline_results

where baseline_results/ holds the downloaded results.json files (any
name, or in per-model subfolders -- the model name is read from inside
each file).
"""

import argparse
import glob
import json
import os

import numpy as np

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_REEVAL = os.path.join(_THIS_DIR, 'fedgkt_reeval', 'test_reeval.json')
DEFAULT_RESULTS_DIR = os.path.join(_THIS_DIR, 'baseline_results')

# FedGKT's canonical published band scores, for the convention check
FEDGKT_PUBLISHED = {'30-100': 0.6646, '100-500': 0.6988, '500+': 0.6894}

BANDS = ['30-100', '100-500', '500+']


def band_of(n, upper_exclusive):
    """
    upper_exclusive=True  -> [30,100), [100,500), [500,inf)
    upper_exclusive=False -> [30,100], (100,500], (500,inf)
    """
    if upper_exclusive:
        if n < 100:
            return '30-100'
        if n < 500:
            return '100-500'
        return '500+'
    else:
        if n <= 100:
            return '30-100'
        if n <= 500:
            return '100-500'
        return '500+'


def macro_by_band(lengths, aucs, upper_exclusive):
    """Mean AUC per band, skipping students with no AUC (single-class)."""
    buckets = {b: [] for b in BANDS}
    for uid, n in lengths.items():
        a = aucs.get(uid)
        if a is None:
            continue
        buckets[band_of(n, upper_exclusive)].append(a)
    return ({b: (float(np.mean(v)) if v else None) for b, v in buckets.items()},
            {b: len(v) for b, v in buckets.items()})


def main():
    parser = argparse.ArgumentParser(
        description="Compare FedGKT and the baselines by student sequence length.")
    parser.add_argument('--reeval', default=DEFAULT_REEVAL)
    parser.add_argument('--results-dir', default=DEFAULT_RESULTS_DIR)
    parser.add_argument('--fedgkt-metric', default='auc_no_first',
                        choices=['auc_no_first', 'auc_full'],
                        help="Which FedGKT per-student AUC to compare against the "
                             "baselines (default: auc_no_first, the aligned one).")
    args = parser.parse_args()

    assert os.path.exists(args.reeval), f"Not found: {args.reeval}"
    reeval = json.load(open(args.reeval))

    lengths = {int(s['user_id']): int(s['n_interactions']) for s in reeval['per_student']}
    fedgkt_full = {int(s['user_id']): s['auc_full'] for s in reeval['per_student']}
    fedgkt_cmp = {int(s['user_id']): s[args.fedgkt_metric] for s in reeval['per_student']}

    print("=" * 74)
    print("Length-band comparison")
    print("=" * 74)
    print(f"\n{len(lengths)} test students, "
          f"{sum(1 for v in fedgkt_full.values() if v is None)} single-class (excluded)")
    lens = sorted(lengths.values())
    print(f"lengths: min {lens[0]}, median {lens[len(lens)//2]}, max {lens[-1]}")

    # ── which band convention reproduces FedGKT's published numbers? ──
    print("\n--- Band convention check (FedGKT's own full-sequence AUCs) ---")
    print(f"{'convention':<22}" + "".join(f"{b:>12}" for b in BANDS))
    print(f"{'published':<22}" + "".join(f"{FEDGKT_PUBLISHED[b]:>12.4f}" for b in BANDS))

    best, best_err = None, None
    for excl in (True, False):
        means, _ = macro_by_band(lengths, fedgkt_full, excl)
        label = '[30,100) [100,500)' if excl else '[30,100] (100,500]'
        print(f"{label:<22}" + "".join(
            f"{means[b]:>12.4f}" if means[b] is not None else f"{'-':>12}" for b in BANDS))
        err = max(abs(means[b] - FEDGKT_PUBLISHED[b]) for b in BANDS if means[b] is not None)
        if best_err is None or err < best_err:
            best, best_err = excl, err

    print(f"\nclosest convention: {'[lower, upper)' if best else '[lower, upper]'}"
          f"   max difference from published: {best_err:.4f}")
    if best_err > 0.005:
        print("  NOTE: neither convention reproduces the published band scores closely.")
        print("  The published numbers may have been computed on a different basis")
        print("  (e.g. before the first-interaction alignment). Treat the comparison")
        print("  below as internally consistent, but don't equate it to the published")
        print("  band figures without checking how those were produced.")

    # ── the comparison ────────────────────────────────────────────────
    models = {}
    paths = sorted(glob.glob(os.path.join(args.results_dir, '**', '*.json'), recursive=True))
    assert paths, (
        f"No results.json files found under {args.results_dir}. Download each "
        f"baseline's results.json from Drive (fedgkt_baselines_runs/<model>/) "
        f"into that folder."
    )
    for p in paths:
        r = json.load(open(p))
        if 'test_per_student_auc' not in r:
            continue
        models[r['model']] = {int(k): v for k, v in r['test_per_student_auc'].items()}

    fed_means, counts = macro_by_band(lengths, fedgkt_cmp, best)

    print(f"\n--- Macro AUC by band ({args.fedgkt_metric} for FedGKT) ---")
    print(f"{'students per band':<12}" + "".join(f"{counts[b]:>12}" for b in BANDS))
    print(f"{'':<12}" + "".join(f"{b:>12}" for b in BANDS))
    print(f"{'FedGKT':<12}" + "".join(
        f"{fed_means[b]:>12.4f}" if fed_means[b] is not None else f"{'-':>12}" for b in BANDS))

    for name in sorted(models):
        means, c = macro_by_band(lengths, models[name], best)
        missing = set(lengths) - set(models[name])
        assert not missing, f"{name}: {len(missing)} test students missing from its results"
        assert c == counts, (
            f"{name}: different students excluded as single-class than FedGKT "
            f"({c} vs {counts}) -- the comparison would not be like-for-like."
        )
        print(f"{name.upper():<12}" + "".join(
            f"{means[b]:>12.4f}" if means[b] is not None else f"{'-':>12}" for b in BANDS))

    # ── gaps, which is what the claim is actually about ───────────────
    print(f"\n--- Gap to FedGKT (positive = baseline ahead) ---")
    print(f"{'':<12}" + "".join(f"{b:>12}" for b in BANDS))
    for name in sorted(models):
        means, _ = macro_by_band(lengths, models[name], best)
        print(f"{name.upper():<12}" + "".join(
            f"{means[b] - fed_means[b]:>+12.4f}" for b in BANDS))

    print("\nA claim about handling long-tailed data lives in how the gap CHANGES")
    print("across bands, not in the overall average. A gap that shrinks toward the")
    print("short-sequence band is evidence for it; a flat or widening gap is not.")


if __name__ == '__main__':
    main()