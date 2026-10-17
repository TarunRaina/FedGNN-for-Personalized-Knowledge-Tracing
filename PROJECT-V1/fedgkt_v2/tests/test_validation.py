"""
tests/test_validation.py -- validation inside run_federated.py.

Setup (sandbox-sized): 4 real short students train, 3 other real short
students stand in as the validation set. 2 students per round, fast
(batched) rounds, groups of 2, dropout ON, 4 rounds, validate every
4 student-passes (= every 2 rounds).

  T1  validation must not change training: final weights with validation
      ON == with validation OFF, difference EXACTLY 0.
  T2  the logged validation score of the last round == evaluate_batched()
      called directly on the saved final weights (exact).
  T3  best_model.pt holds the weights of the best logged validation:
      re-scoring it reproduces that number (exact).
  T4  validation happened exactly at the expected rounds (passes 4 and 8).
  N1  (negative) scoring the STARTING weights must give a different number
      -- proves T2/T3 can see a difference.
"""
import os, sys, shutil, tempfile
import torch
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from src.utils import config as cfg
from run_federated import federated_training, build_model, BEST_FILE
from src.training.evaluator_batched import evaluate_batched

TRAIN = [233536, 45224, 111852, 105754]
VAL = [224825, 138435, 187058]
dev = torch.device('cpu')
KW = dict(train_ids=TRAIN, rounds=4, clients_per_round=2, local_steps=1, local_lr=0.0,
          server_lr=1e-3, seed=42, dropout=0.2, model_kind='v2', device=dev, verbose=False,
          batched=True, group_size=2)
tmp = tempfile.mkdtemp()
ei = torch.load(cfg.EDGE_INDEX_PATH, weights_only=False)

s_on, h_on = federated_training(os.path.join(tmp, 'on'), val_ids=VAL, val_every_passes=4, **KW)
s_off, _ = federated_training(os.path.join(tmp, 'off'), val_ids=None, **KW)

w = max((a - b).abs().max().item() for a, b in zip(s_on.model.parameters(), s_off.model.parameters()))
t1 = w == 0.0
print(f"T1 weights, validation on vs off: diff {w:.1e} -> {'PASS' if t1 else 'FAIL'}")

def score(state):
    m = build_model('v2', 0.2, 42); m.load_state_dict(state, strict=True)
    return evaluate_batched(m, VAL, edge_index=ei, batch_size=16, device=dev)['macro_auc']

logged_last = h_on[-1]['val_macro_auc']
direct_last = score(s_on.global_state())
t2 = logged_last == direct_last
print(f"T2 last logged val {logged_last:.10f} vs re-scored {direct_last:.10f} -> {'PASS' if t2 else 'FAIL'}")

logged = [h['val_macro_auc'] for h in h_on if h['val_macro_auc'] is not None]
best_state = torch.load(os.path.join(tmp, 'on', BEST_FILE), weights_only=False)
t3 = score(best_state) == max(logged)
print(f"T3 best_model.pt re-scored {score(best_state):.10f} vs best logged {max(logged):.10f} -> {'PASS' if t3 else 'FAIL'}")

where = [(h['round'], h['passes']) for h in h_on if h['val_macro_auc'] is not None]
t4 = where == [(1, 4), (3, 8)]
print(f"T4 validated at (round, passes) {where} -> {'PASS' if t4 else 'FAIL'}")

start = build_model('v2', 0.2, 42).state_dict()
n1 = score(start) != direct_last
print(f"N1 starting weights score {score(start):.10f} -> {'differs (good)' if n1 else 'SAME (BAD)'}")

shutil.rmtree(tmp)
print(f"\nRESULT: {'ALL PASS' if (t1 and t2 and t3 and t4 and n1) else 'SOMETHING FAILED'}")