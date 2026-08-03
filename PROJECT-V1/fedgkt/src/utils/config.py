import os

# ── project root ──────────────────────────────────────────────────────────
# Computed from this file's own location, not the current working directory.
# This avoids the exact cwd bug hit in preprocessing Step 4 (FileNotFoundError
# because the shell wasn't cd'd into fedgkt/). Since this config gets imported
# from multiple entry points later, anchoring to this file's real path is
# safer than relying on relative paths + always running from the right folder.
THIS_FILE = os.path.abspath(__file__)
SRC_UTILS_DIR = os.path.dirname(THIS_FILE)        # src/utils
SRC_DIR = os.path.dirname(SRC_UTILS_DIR)          # src
PROJECT_ROOT = os.path.dirname(SRC_DIR)           # fedgkt/

# ── data paths ───────────────────────────────────────────────────────────────
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
RAW_DIR = os.path.join(DATA_DIR, 'raw')
PROCESSED_DIR = os.path.join(DATA_DIR, 'processed')

VOCAB_PATH = os.path.join(PROCESSED_DIR, 'exercise_vocab.json')
EDGE_INDEX_PATH = os.path.join(PROCESSED_DIR, 'edge_index.pt')
PKG_DIR = os.path.join(PROCESSED_DIR, 'pkgs')
SPLITS_PATH = os.path.join(PROCESSED_DIR, 'student_splits.json')
STUDENT_STATS_PATH = os.path.join(PROCESSED_DIR, 'student_stats.csv')

# ── graph structure (locked, from preprocessing) ─────────────────────────────
NUM_NODES = 835          # exercises / concepts, indices 0-834
NUM_EDGES = 978           # prerequisite edges, valid DAG
NUM_FEATURES = 7          # node feature dimension

# ── node feature order (for readability everywhere else in the code) ────────
FEATURE_NAMES = [
    'mastery_score',
    'attempt_count',
    'recent_accuracy_5',
    'recent_accuracy_10',
    'time_decay',
    'streak',
    'first_attempt',
]
FEATURE_IDX = {name: i for i, name in enumerate(FEATURE_NAMES)}

# ── cold-start feature defaults (locked) ─────────────────────────────────────
COLD_START_MASTERY = 0.5   # unknown-state prior
COLD_START_DEFAULT = 0.0   # everything else starts at 0

# ── feature update constants (locked, confirmed across both Claude instances) ─
MASTERY_EMA_ALPHA = 0.3            # EMA weight on new observation
TIME_DECAY_RATE = 0.1              # exp(-rate * days_since_last_attempt_on_concept)
STREAK_CAP = 10                    # raw streak count caps at 10 before normalising
RECENT_ACCURACY_WINDOWS = [5, 10]  # recent_accuracy_5, recent_accuracy_10
# recent_accuracy_k = mean over whatever attempts exist so far on this concept
# if k not yet reached. Zero attempts -> defaults to 0.0 (first_attempt=0 signals
# to the model that this 0 is a cold-start placeholder, not a real low score).

# ── data quality handling (NEW — flagged during Step 6 manual verification) ──
# time_taken has extreme outliers in the raw log (idle sessions left open,
# observed up to ~28,892 seconds / ~8hr on pkg_21419). Clip time_taken to this
# cap before using it in any feature computation, so a handful of idle-session
# artifacts don't distort EMA-based or averaged features.
TIME_TAKEN_CAP_SECONDS = 1800       # 30 minutes

MICROSECONDS_PER_DAY = 86_400_000_000  # for converting time_done deltas to days

# ── GAT architecture (locked, from original design doc) ──────────────────────
GAT_LAYER_1_HEADS = 4
GAT_LAYER_1_DIM = 32     # per head -> 128-dim concatenated output
GAT_LAYER_2_HEADS = 4
GAT_LAYER_2_DIM = 32     # per head -> 128-dim concatenated output
GAT_LAYER_3_HEADS = 1
GAT_LAYER_3_DIM = 64     # final node embedding dimension

# ── personal MLP head (locked) ────────────────────────────────────────────────
HEAD_HIDDEN_DIM = 32     # Linear(64->32) -> ReLU -> Dropout -> Linear(32->1) -> Sigmoid
DROPOUT = 0.2

# ── training hyperparameters ──────────────────────────────────────────────────
# NOTE: these are NOT locked design decisions like the constants above — they
# are standard starting-point defaults. The project plan (Month 3: "hyperparameter
# sweep") explicitly expects these to be tuned later. Flagging clearly so this
# isn't mistaken for a settled decision the way the constants above are.
LEARNING_RATE = 1e-3
OPTIMIZER = 'Adam'
STUDENTS_PER_GRADIENT_STEP = 8    # accumulate loss over N students before backward+step
NUM_EPOCHS = 20                    # placeholder, will likely change after first real run
RANDOM_SEED = 42                  # same seed used throughout preprocessing

# ── train/val/test split sizes (locked, from Step 4) ─────────────────────────
TRAIN_SIZE = 800
VAL_SIZE = 100
TEST_SIZE = 100

# ── device (confirmed decision, not a silent omission) ───────────────────────
# 835 nodes / 978 edges, processed one interaction at a time, sequentially --
# GPU transfer overhead per call would likely exceed any compute benefit at
# this graph size. CPU is the deliberate default. Change to 'cuda' here to
# benchmark GPU later if ever needed -- no model code changes required, since
# nothing in pkg.py/gat.py/fedgkt.py/evaluator.py hardcodes a device.
DEVICE = 'cpu'

# ── early stopping (NOT locked -- standard default, like the other training
# hyperparameters in this file. Tune later if needed.) ───────────────────────
EARLY_STOPPING_PATIENCE = 5   # epochs without val macro_auc improvement before stopping

# ── checkpoints ────────────────────────────────────────────────────────────
CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, 'checkpoints')

if __name__ == '__main__':
    print("=" * 70)
    print("FedGKT Phase 1 Config — sanity check")
    print("=" * 70)

    print(f"\nPROJECT_ROOT: {PROJECT_ROOT}")
    print(f"  exists: {os.path.isdir(PROJECT_ROOT)}")

    print(f"\nData paths:")
    for name, path in [
        ('VOCAB_PATH', VOCAB_PATH),
        ('EDGE_INDEX_PATH', EDGE_INDEX_PATH),
        ('PKG_DIR', PKG_DIR),
        ('SPLITS_PATH', SPLITS_PATH),
        ('STUDENT_STATS_PATH', STUDENT_STATS_PATH),
    ]:
        exists = os.path.exists(path)
        status = 'OK' if exists else 'MISSING'
        print(f"  {name:22s} {path}   [{status}]")

    print(f"\nGraph structure: {NUM_NODES} nodes, {NUM_EDGES} edges, {NUM_FEATURES} features/node")
    print(f"Feature order: {FEATURE_NAMES}")

    print(f"\nGAT architecture:")
    print(f"  Layer 1: {GAT_LAYER_1_HEADS} heads x {GAT_LAYER_1_DIM} dim -> {GAT_LAYER_1_HEADS*GAT_LAYER_1_DIM}-dim output")
    print(f"  Layer 2: {GAT_LAYER_2_HEADS} heads x {GAT_LAYER_2_DIM} dim -> {GAT_LAYER_2_HEADS*GAT_LAYER_2_DIM}-dim output")
    print(f"  Layer 3: {GAT_LAYER_3_HEADS} head  x {GAT_LAYER_3_DIM} dim -> {GAT_LAYER_3_DIM}-dim output")
    print(f"  Head:    Linear({GAT_LAYER_3_DIM}->{HEAD_HIDDEN_DIM}) -> ReLU -> Dropout({DROPOUT}) -> Linear({HEAD_HIDDEN_DIM}->1) -> Sigmoid")

    print(f"\nTraining hyperparameters (placeholder defaults, not locked):")
    print(f"  Learning rate: {LEARNING_RATE}")
    print(f"  Optimizer: {OPTIMIZER}")
    print(f"  Students per gradient step: {STUDENTS_PER_GRADIENT_STEP}")
    print(f"  Epochs: {NUM_EPOCHS}")
    print(f"  Random seed: {RANDOM_SEED}")

    print(f"\nSplit sizes: train={TRAIN_SIZE} val={VAL_SIZE} test={TEST_SIZE}")

    print("\nConfig sanity check complete.")