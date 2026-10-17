"""
tests/test_step4a_fast_vs_slow.py -- Phase 2 step 4a.

The fast round (students grouped by length, side by side) must give the
same equal-vote average as step 3's proven one-at-a-time round.
Dropout OFF (the two paths draw dropout differently by design).

Tolerances, declared before running:
  CPU: averaged update <= 1e-6; weights after one server Adam step <= 1e-5.
  GPU: averaged update <= 1e-5. Weights after Adam are NOT compared on GPU:
       Adam's first step can blow GPU float noise in near-zero gradients up
       to a full step. Instead the fast round is run TWICE and its difference
       from itself is reported as the measured GPU noise floor.
Negative test N1: averaging the group means 50/50 (ignoring group sizes)
  must FAIL -- only meaningful if the groups are uneven, which is checked.

Usage:
  python tests/test_step4a_fast_vs_slow.py --ids 233536 45224 ... --group-size 3 --device cpu
  python tests/test_step4a_fast_vs_slow.py --n-train 100 --group-size 8 --device cuda
"""
import argparse, json, os, sys, time
import torch
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from src.utils import config as cfg
cfg.DROPOUT = 0.0
from src.models.fedgkt_v2 import FedGKTv2
from src.federated.server import Server, average_equal
from src.federated.batched_round import train_round_batched, make_groups
from run_federated import train_round

ap = argparse.ArgumentParser()
ap.add_argument('--ids', type=int, nargs='*')
ap.add_argument('--n-train', type=int, help="use the first N students of the train split")
ap.add_argument('--group-size', type=int, default=8)
ap.add_argument('--device', default='cpu')
a = ap.parse_args()
dev = torch.device(a.device)
if a.ids:
    ids = a.ids
else:
    ids = json.load(open(cfg.SPLITS_PATH))['train'][:a.n_train]
UPD_TOL = 1e-6 if dev.type == 'cpu' else 1e-5
W_TOL = 1e-5
ei = torch.load(cfg.EDGE_INDEX_PATH, weights_only=False)
torch.manual_seed(42); S0 = {k: v.clone() for k, v in FedGKTv2().state_dict().items()}
def fresh():
    m = FedGKTv2(); m.load_state_dict(S0, strict=True); return m.to(dev)
sizes = [len(g) for g in make_groups(ids, a.group_size)]
print(f"{len(ids)} students, device {dev}, group size {a.group_size} -> groups {sizes}", flush=True)
uneven = len(set(sizes)) > 1

t0 = time.time()
slow = average_equal([o['update'] for o in train_round(fresh(), S0, ids, ei, dev, 1, 0.0, 0, 42)])
t_slow = time.time() - t0
t0 = time.time()
fast_out = train_round_batched(fresh(), S0, ids, ei, dev, 1, 0.0, 0, 42, a.group_size)[0]
t_fast = time.time() - t0
fast = fast_out['update']
print(f"slow (one at a time) {t_slow:.1f} s   fast (grouped) {t_fast:.1f} s", flush=True)

mx = lambda x, y: max((x[n] - y[n]).abs().max().item() for n in x)
du = mx(fast, slow)
ok = du <= UPD_TOL
print(f"\naveraged update, fast vs slow: {du:.2e}  (tol {UPD_TOL:.0e})  {'PASS' if ok else 'FAIL'}")

if dev.type == 'cpu':
    s1, s2 = Server(fresh(), cfg.LEARNING_RATE), Server(fresh(), cfg.LEARNING_RATE)
    s1.apply_round([slow]); s2.apply_round([fast])
    dw = max((p1 - p2).abs().max().item() for p1, p2 in zip(s1.model.parameters(), s2.model.parameters()))
    print(f"weights after Adam, fast vs slow: {dw:.2e}  (tol {W_TOL:.0e})  {'PASS' if dw <= W_TOL else 'FAIL'}")
    ok &= dw <= W_TOL
else:
    fast2 = train_round_batched(fresh(), S0, ids, ei, dev, 1, 0.0, 0, 42, a.group_size)[0]['update']
    print(f"GPU noise floor (fast vs fast, same input): {mx(fast, fast2):.2e}")

if uneven:
    unweighted = {n: sum(g[n] for g in fast_out['group_means']) / len(sizes) for n in fast}
    d1 = mx(unweighted, slow)
    print(f"\nN1 group means averaged 50/50: {d1:.2e} -> {'caught (good)' if d1 > UPD_TOL else 'NOT caught'}")
    ok &= d1 > UPD_TOL
else:
    print("\nN1 skipped: groups are all the same size, so the weighting bug cannot show")
print(f"\nRESULT: {'ALL PASS' if ok else 'SOMETHING FAILED'}")