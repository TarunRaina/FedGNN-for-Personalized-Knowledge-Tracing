"""
tests/test_step3_equivalence.py -- Phase 2 step 3 anchor test.

ONE federated round (local_steps=1, equal vote, server Adam)
  ==  ONE centralised training step on the same students.
Conditions (see client.py): head shared, same students as one centralised
batch, dropout OFF. Pre-declared tolerances: update 1e-6, weights 1e-5.
Negative tests (must FAIL the 1e-6 check):
  N1  standard FedAvg weighting (by number of answers) instead of equal vote
  N2  summing updates instead of averaging (forgetting to divide by K)
Usage: python tests/test_step3_equivalence.py 233536 45224 21419
"""
import os, sys, time
import torch
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from src.utils import config as cfg
cfg.DROPOUT = 0.0                                   # condition 3, BEFORE any model is built
from src.models.fedgkt_v2 import FedGKTv2
from src.training.centralised_batched import train_one_epoch_batched
from src.federated.client import run_client
from src.federated.server import Server

UPD_TOL, W_TOL = 1e-6, 1e-5
ids = [int(a) for a in sys.argv[1:]]; B = len(ids); dev = torch.device('cpu')
ei = torch.load(cfg.EDGE_INDEX_PATH, weights_only=False)
torch.manual_seed(42); S0 = {k: v.clone() for k, v in FedGKTv2().state_dict().items()}
def fresh():
    m = FedGKTv2(); m.load_state_dict(S0, strict=True); return m
assert all(m.p == 0.0 for m in fresh().modules() if isinstance(m, torch.nn.Dropout)), "dropout not off"
print(f"students {ids}  (B = {B})", flush=True)

# ── centralised: one batch of all B students = exactly one Adam step ─────
t0 = time.time()
mc = fresh(); adam_c = torch.optim.Adam(mc.parameters(), lr=cfg.LEARNING_RATE)
train_one_epoch_batched(mc, adam_c, ids, ei, B, dev)
assert adam_c.state[next(iter(mc.parameters()))]['step'] == 1, "centralised took more than one step"
g_c = {n: p.grad.detach().double().clone() for n, p in mc.named_parameters()}
w_c = {n: p.detach().clone() for n, p in mc.named_parameters()}
print(f"centralised step done ({time.time()-t0:.0f} s)", flush=True)

# ── federated: each student alone, then server averages + Adam ──────────
t0 = time.time()
srv = Server(fresh(), lr=cfg.LEARNING_RATE); gstate = srv.global_state(); work = fresh()
outs = [run_client(work, gstate, u, ei, dev, local_steps=1, local_lr=0.0, round_idx=0, base_seed=42) for u in ids]
avg = srv.apply_round([o['update'] for o in outs])
w_f = {n: p.detach().clone() for n, p in srv.model.named_parameters()}
print(f"federated round done ({time.time()-t0:.0f} s)\n", flush=True)

mx = lambda a, b: max((a[n] - b[n]).abs().max().item() for n in a)
du, dw = mx(avg, g_c), mx(w_f, w_c)
print(f"averaged update vs centralised gradient : {du:.2e}  (tol {UPD_TOL:.0e})  {'PASS' if du <= UPD_TOL else 'FAIL'}")
print(f"weights after Adam vs centralised       : {dw:.2e}  (tol {W_TOL:.0e})  {'PASS' if dw <= W_TOL else 'FAIL'}")
gscale = max(g.abs().max().item() for g in g_c.values())
print(f"(largest gradient entry, for scale: {gscale:.2e})")

n = [o['n_answers'] for o in outs]
wavg = {k: sum(ni * o['update'][k] for ni, o in zip(n, outs)) / sum(n) for k in avg}
summ = {k: sum(o['update'][k] for o in outs) for k in avg}
d1, d2 = mx(wavg, g_c), mx(summ, g_c)
print(f"\nN1 answers-weighted average (lengths {n}): diff {d1:.2e} -> {'caught (good)' if d1 > UPD_TOL else 'NOT caught'}")
print(f"N2 sum instead of average             : diff {d2:.2e} -> {'caught (good)' if d2 > UPD_TOL else 'NOT caught'}")
ok = du <= UPD_TOL and dw <= W_TOL and d1 > UPD_TOL and d2 > UPD_TOL
print(f"\nRESULT: {'ALL PASS' if ok else 'SOMETHING FAILED'}")