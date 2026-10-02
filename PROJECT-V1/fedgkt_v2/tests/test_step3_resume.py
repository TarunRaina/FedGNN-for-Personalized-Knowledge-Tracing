"""
tests/test_step3_resume.py -- Phase 2 step 3: save/resume must be EXACT.

Setup: the 7 short real students, 3 students per round, 2 local steps,
dropout ON (so the per-round/per-student random seeding is really tested),
4 rounds, CPU.

  T1  straight run of 4 rounds  ==  run of 2 rounds, "disconnect", resume
      to 4. Weights and Adam memory must match with difference EXACTLY 0.
  T2  the students chosen each round are identical in both runs.
  N1  (negative) resuming with a changed setting (dropout) must be REFUSED.
  N2  (negative) a run with a different seed must NOT match -- proves the
      T1 comparison is able to see a difference at all.
"""
import os, sys, shutil, tempfile
import torch
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from run_federated import federated_training

IDS = [233536, 45224, 111852, 224825, 138435, 187058, 105754]
KW = dict(train_ids=IDS, rounds=4, clients_per_round=3, local_steps=2, local_lr=0.05,
          server_lr=1e-3, dropout=0.2, model_kind='v2', device=torch.device('cpu'), verbose=False)
tmp = tempfile.mkdtemp()

def max_diff(s1, s2):
    w = max((s1.model.state_dict()[k] - s2.model.state_dict()[k]).abs().max().item()
            for k in s1.model.state_dict())
    a1, a2 = s1.adam.state_dict()['state'], s2.adam.state_dict()['state']
    a = max((a1[i][key].float() - a2[i][key].float()).abs().max().item()
            for i in a1 for key in ('exp_avg', 'exp_avg_sq', 'step'))
    return w, a

print("straight run, 4 rounds ...", flush=True)
sA, hA = federated_training(os.path.join(tmp, 'A'), seed=42, **KW)
print("interrupted run: 2 rounds, then resume ...", flush=True)
federated_training(os.path.join(tmp, 'B'), seed=42, max_rounds_this_session=2, **KW)
sB, hB = federated_training(os.path.join(tmp, 'B'), seed=42, **KW)

w, a = max_diff(sA, sB)
t1 = w == 0.0 and a == 0.0 and sA.rounds_done == sB.rounds_done == 4
print(f"\nT1 straight vs resumed: weights diff {w:.1e}, Adam memory diff {a:.1e} -> {'PASS' if t1 else 'FAIL'}")
t2 = [h['students'] for h in hA] == [h['students'] for h in hB]
print(f"T2 same students every round: {t2}   {[h['students'] for h in hA]}")

try:
    federated_training(os.path.join(tmp, 'B'), seed=42, **{**KW, 'dropout': 0.1, 'rounds': 5})
    n1 = False
except SystemExit as e:
    n1 = True
    print("\nN1 resume with changed dropout -> refused (good):\n   " + str(e).splitlines()[0])
if not n1:
    print("\nN1 resume with changed dropout -> NOT refused (BAD)")

sC, _ = federated_training(os.path.join(tmp, 'C'), seed=7, **KW)
wc, _ = max_diff(sA, sC)
n2 = wc > 0
print(f"N2 different seed: weights diff {wc:.1e} -> {'differs (good)' if n2 else 'identical (BAD: comparison is blind)'}")

shutil.rmtree(tmp)
print(f"\nRESULT: {'ALL PASS' if (t1 and t2 and n1 and n2) else 'SOMETHING FAILED'}")