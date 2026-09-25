"""
drivers/run_gkt.py

Trains the GKT baseline -- the only other graph-based model -- on FedGKT's
own prerequisite graph, and scores it on the test set. Same structure as
the other drivers. Meant to be run in Colab with a GPU runtime.

Requires pykt_sequences/gkt_graph_prerequisite.npz, built locally by
baseline_step4_build_gkt_graph.py and carried to Colab inside the zip.

SETTINGS (verified against pyKT 0.0.38, not recalled)
-----------------------------------------------------
From examples/wandb_gkt_train.py (argparse defaults -- what reaches the
model; kt_config.json's model entries are never used by the script):

      hidden_dim        64
      emb_size          64   not an argparse option: the script sets
                               args.emb_size = args.hidden_dim
      dropout          0.5
      learning_rate   1e-2
      graph_type   'transition'   <-- REPLACED here by FedGKT's graph

  batch_size        16 -> 8  wandb_train.py lowers kt_config's 256 to 16
                             for gkt; lowered further to 8 here -- see below
  emb_type           'qid'   pyKT's default

BATCH SIZE 8, NOT pyKT's 16  (a disclosed deviation)
-----------------------------------------------------
At batch size 16 the smoke test ran out of memory on a 16 GB T4 during
the first epoch (14.17 GB allocated, inside GKT's _agg_neighbors).

Why: at every one of the 200 steps, GKT holds a hidden state for ALL
concepts for every student in the batch, and backpropagation keeps all
200 steps in memory at once -- so memory grows with
batch x 200 x num_concepts. pyKT's 16 was itself already a reduction
"because of OOM", chosen on its benchmark datasets, which have far fewer
concepts (e.g. ~110 for ASSISTments 2009). Ours has 835.

Gradient accumulation (4 students at a time, update every 4 batches) was
considered and rejected. GKT's MLP layers use BatchNorm1d, which computes
its statistics from the batch it is given -- so accumulation would report
an effective batch of 16 while every BatchNorm actually saw 4 students.
Lowering the batch size honestly is the more defensible choice.

For the paper: "GKT's batch size was reduced from pyKT's default of 16 to
8 to fit a single 16 GB T4 GPU, given our 835-concept graph; pyKT itself
reduces GKT's batch size for memory reasons."

If 8 also runs out of memory, the same reasoning applies at 4. (1,716
training rows divide evenly by 4; at 8 the last batch has 4 rows. pyKT's
BatchNorm wrapper only special-cases batches of 0 or 1, so neither is an
issue.)

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True is also set below. It
only reduces memory fragmentation and does not change any computation.

(kt_config.json lists hidden_dim 32 / emb_size 32 / lr 1e-3, but as with
DKT and AKT, those entries never reach the model.)

NOTE -- learning rate 1e-2 is ten times FedGKT's 1e-3. Under the agreed
"pyKT's defaults, untuned" policy it is correct; state it explicitly in
the methodology.

THE ONE DELIBERATE DEPARTURE FROM pyKT's DEFAULTS
--------------------------------------------------
pyKT's GKT defaults to a 'transition' graph derived from student data.
This baseline uses FedGKT's expert prerequisite graph instead -- that is
the whole point of it: same edges as FedGKT, different architecture, so
the comparison isolates the model rather than the graph.

The graph is named 'prerequisite', not 'transition', on purpose. pyKT's
init_model() loads gkt_graph_{graph_type}.npz if present and otherwise
BUILDS one. Reusing the name 'transition' would mean a missing file
silently gets replaced by pyKT's data-derived graph. This driver also
checks the file exists and is correct BEFORE handing over to pyKT, then
confirms after building the model that GKT really holds that exact
matrix.

The graph's sha256 is recorded in the run's settings, so resuming a run
after the graph file has changed is refused rather than silently mixing
two graphs into one run.

Shared parity settings (patience 5 on validation macro AUC, max 100
epochs, no shuffling, seed 42) are identical to every other baseline.

USAGE (in Colab, from the staging folder)
-----------------------------------------
    python drivers/run_gkt.py --smoke    # 2 epochs, into gkt_smoke/
    python drivers/run_gkt.py            # the real run; resumes if disconnected

Results go to  <run-root>/gkt/  (or gkt_smoke/ for --smoke).
"""

import argparse
import hashlib
import os
import sys

# Must be set before the first CUDA allocation. Reduces fragmentation only;
# changes no computation. (Suggested by the OOM message itself.)
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')

import numpy as np   # noqa: E402
import torch         # noqa: E402

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))          # drivers/
STAGING_ROOT = os.path.dirname(_THIS_DIR)                        # staging root
COMMON_DIR = os.path.join(STAGING_ROOT, 'common')
if COMMON_DIR not in sys.path:
    sys.path.insert(0, COMMON_DIR)

import data_config as dc          # noqa: E402
import train_utils as tu          # noqa: E402
from pykt.models import init_model  # noqa: E402


MODEL_NAME = 'gkt'
GRAPH_TYPE = 'prerequisite'
EXPECTED_EDGES = 978

# pyKT 0.0.38 defaults for GKT -- see module docstring for sources
MODEL_CONFIG = {
    'hidden_dim': 64,
    'emb_size': 64,          # = hidden_dim, as pyKT's script sets it
    'dropout': 0.5,
    'graph_type': GRAPH_TYPE,
}
BATCH_SIZE = 8            # pyKT's default is 16 -- lowered to fit a T4; see docstring
LEARNING_RATE = 1e-2      # pyKT's GKT default -- differs from FedGKT's 1e-3
EMB_TYPE = 'qid'

# shared parity settings (identical across all five baselines)
MAX_EPOCHS = 100
PATIENCE = 5
MIN_DELTA = 0.0
SEED = 42

DEFAULT_RUN_ROOT = '/content/drive/MyDrive/fedgkt_baselines_runs'
DEFAULT_SEQUENCES_DIR = os.path.join(STAGING_ROOT, 'pykt_sequences')


def load_and_check_graph(dpath, num_c):
    """
    Loads the graph exactly as pyKT's init_model() will, and verifies it is
    FedGKT's prerequisite graph packaged pyKT's way. Returns (tensor, sha256).
    """
    path = os.path.join(dpath, f'gkt_graph_{GRAPH_TYPE}.npz')
    assert os.path.exists(path), (
        f"Graph file missing: {path}\n"
        f"Run baseline_step4_build_gkt_graph.py locally, then re-zip. Without "
        f"this file pyKT would try to build a graph of its own."
    )

    matrix = np.load(path, allow_pickle=True)['matrix']
    graph = torch.tensor(matrix).float()          # identical to init_model's load

    row_sums = graph.sum(dim=1)
    nonempty = row_sums > 0
    assert tuple(graph.shape) == (num_c, num_c), f"Graph shape {tuple(graph.shape)}, expected ({num_c}, {num_c})"
    assert torch.all(torch.isfinite(graph)), "Graph contains NaN or Inf"
    assert torch.all(torch.diagonal(graph) == 0), "Graph diagonal is not zero"
    assert int((graph > 0).sum()) == EXPECTED_EDGES, (
        f"Graph has {int((graph > 0).sum())} edges, expected {EXPECTED_EDGES}"
    )
    assert torch.allclose(row_sums[nonempty], torch.ones_like(row_sums[nonempty]), atol=1e-5), (
        "A non-empty row does not sum to 1 -- graph is not row-normalised"
    )

    sha = hashlib.sha256(np.ascontiguousarray(matrix, dtype=np.float32).tobytes()).hexdigest()
    print(f"Graph: {path}")
    print(f"  {EXPECTED_EDGES} edges, diagonal zero, {int(nonempty.sum())} rows summing to 1, "
          f"{int((~nonempty).sum())} empty rows")
    print(f"  sha256 {sha}")
    return graph, sha


def main():
    parser = argparse.ArgumentParser(description="Train the GKT baseline.")
    parser.add_argument('--smoke', action='store_true',
                        help="Quick 2-epoch check into a separate gkt_smoke folder.")
    parser.add_argument('--run-root', default=DEFAULT_RUN_ROOT,
                        help="Parent folder for run outputs (Google Drive in Colab).")
    parser.add_argument('--sequences-dir', default=DEFAULT_SEQUENCES_DIR,
                        help="Folder holding the three pyKT sequence CSVs and the graph file.")
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

    print("\n--- Checking the GKT graph ---")
    expected_graph, graph_sha = load_and_check_graph(cfg['dpath'], cfg['num_c'])

    tu.set_seed(SEED)   # seed BEFORE building the model: weight init is random
    model = init_model(MODEL_NAME, MODEL_CONFIG, cfg, EMB_TYPE)

    # Confirm GKT really holds FedGKT's graph, unchanged and frozen.
    held = model.graph.detach().cpu()
    assert torch.equal(held, expected_graph), (
        "BUG: the graph inside GKT is not the prerequisite graph that was checked."
    )
    assert not model.graph.requires_grad, "BUG: GKT's graph is trainable; it should be fixed."
    print("GKT holds FedGKT's prerequisite graph exactly, frozen -- OK")

    results = tu.fit_and_evaluate(
        model, MODEL_NAME, cfg,
        batch_size=BATCH_SIZE,
        run_dir=run_dir,
        lr=LEARNING_RATE,
        max_epochs=max_epochs,
        patience=PATIENCE,
        min_delta=MIN_DELTA,
        seed=SEED,
        # graph_sha256 is part of the run's settings: resuming with a
        # different graph file is refused rather than mixing two graphs
        model_config={**MODEL_CONFIG, 'emb_type': EMB_TYPE, 'graph_sha256': graph_sha},
    )

    print("\n" + "=" * 70)
    print(f"GKT {'smoke test' if args.smoke else 'run'} complete.")
    print(f"  best epoch     : {results['best_epoch']}")
    print(f"  test macro AUC : {results['test_macro_auc']:.4f}")
    print(f"  test pooled AUC: {results['test_pooled_auc']:.4f}")
    print("=" * 70)


if __name__ == '__main__':
    main()