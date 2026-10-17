"""
run_federated.py   (fedgkt_v2 root, next to run_screen.py)

Phase 2, step 3 -- THE ROUND LOOP.

What it does, in plain words
----------------------------
Round after round:
  1. pick which training students take part this round;
  2. each one trains alone on its own answers (client.run_client) and
     returns its update -- never its answers;
  3. the server averages the updates with an equal vote and takes one
     Adam step (server.Server);
  4. everything is saved, so a disconnect loses at most the current round.

Step 3 scope (deliberately simple)
----------------------------------
- By default students train ONE AT A TIME (train_round below).
  --batched switches to the step 4a fast round (src/federated/batched_round.py):
  students grouped by length and computed side by side on the GPU, with the
  equal vote preserved. Only for --local-steps 1. Tested against this
  one-at-a-time version (tests/test_step4a_fast_vs_slow.py).
- Everything is shared, head included. Personal heads are step 6.
- Validation (added for the step 7 learning-speed tests): every
  --val-every-passes STUDENT-PASSES (default 800 = each student trained on
  once, on average), NOT every N rounds -- a round of 100 students costs
  ~12x a round of 8, so rounds are not comparable across settings.
  Scored exactly like the centralised screens: evaluate_batched(...)
  ['macro_auc'] on the 100 validation students (screen.py line 282).
  Training seconds and validation seconds are logged SEPARATELY, so
  learning curves can be compared against pure training GPU time.
  The best-validated shared weights are saved as best_model.pt.
  Validation runs in eval mode under no_grad and every client re-seeds
  itself, so validating does not change training (tested).

Resume -- exact, by construction
--------------------------------
Nothing in a round depends on hidden random state:
  - which students join round r is drawn from a generator seeded with
    (seed, r) alone;
  - each student's dropout is seeded with (seed, r, student) in client.py.
So the saved state only needs the server (weights + Adam memory), the
round count and the history. Re-running the SAME command continues from
the next round and gives bit-identical results to an uninterrupted run.

The saved run also records a "fingerprint": every setting that affects the
computation, the training student list, the graph file, and the code of
client.py, server.py and this file. Resuming with ANY of those changed is
refused, with the differences listed. (screen.py's resume check did not
record dropout; this one records everything.) The total number of rounds
is NOT in the fingerprint, so a finished run can be extended.

Usage (Colab):
    !python run_federated.py --run-dir /content/drive/MyDrive/fedgkt_phase2/<name> \
        --rounds 100 --clients-per-round 8 --local-steps 1
"""

import argparse
import hashlib
import json
import os
import random
import sys
import time

import torch

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

from src.utils import config as cfg
from src.federated.client import run_client
from src.federated.server import Server
from src.federated.batched_round import train_round_batched
from src.training.evaluator_batched import evaluate_batched

STATE_FILE = 'state.pt'
BEST_FILE = 'best_model.pt'
HISTORY_FILE = 'history.json'


# ── helpers ────────────────────────────────────────────────────────────────
def _sha(path):
    with open(path, 'rb') as f:
        return hashlib.sha256(f.read()).hexdigest()[:12]


def build_model(model_kind, dropout, seed):
    """Dropout is read at build time, so it is set first. Same seed -> same
    starting weights (FedGKTv2 builds in v1's order -- see run_screen.py)."""
    cfg.DROPOUT = dropout
    torch.manual_seed(seed)
    if model_kind == 'v1':
        from src.models.fedgkt import FedGKT
        return FedGKT()
    from src.models.fedgkt_v2 import FedGKTv2
    return FedGKTv2(use_prior=model_kind in ('v2', 'v2-prior'),
                    use_skip=model_kind in ('v2', 'v2-skip'))


def choose_students(train_ids, clients_per_round, seed, round_idx):
    """Which students join round `round_idx`. Depends only on (seed, round),
    so it is identical whether or not the run was interrupted."""
    pool = sorted(int(u) for u in train_ids)
    if clients_per_round >= len(pool):
        return pool
    rng = random.Random(int(seed) * 1_000_003 + int(round_idx))
    return sorted(rng.sample(pool, clients_per_round))


def make_fingerprint(settings, train_ids):
    code = {name: _sha(os.path.join(_THIS_DIR, rel)) for name, rel in [
        ('client.py', 'src/federated/client.py'),
        ('server.py', 'src/federated/server.py'),
        ('batched_round.py', 'src/federated/batched_round.py'),
        ('run_federated.py', 'run_federated.py')]}
    return {**settings,
            'train_ids_sha': hashlib.sha256(
                ','.join(str(u) for u in sorted(int(x) for x in train_ids)).encode()).hexdigest()[:12],
            'n_train_ids': len(train_ids),
            'edge_index_sha': _sha(cfg.EDGE_INDEX_PATH),
            'code': code}


def _atomic_save(obj, path, use_json=False):
    tmp = path + '.tmp'
    if use_json:
        with open(tmp, 'w') as f:
            json.dump(obj, f, indent=2)
    else:
        torch.save(obj, tmp)
    os.replace(tmp, path)        # never leaves a half-written file behind


# ── one round, one student at a time (step 4 replaces only this) ──────────
def train_round(work_model, global_state, student_ids, edge_index, device,
                local_steps, local_lr, round_idx, seed):
    return [run_client(work_model, global_state, u, edge_index, device,
                       local_steps, local_lr, round_idx, seed)
            for u in student_ids]


# ── the loop ───────────────────────────────────────────────────────────────
def federated_training(run_dir, train_ids, rounds, clients_per_round, local_steps,
                       local_lr, server_lr, seed, dropout, model_kind, device,
                       max_rounds_this_session=None, verbose=True,
                       batched=False, group_size=8,
                       val_ids=None, val_every_passes=800, eval_batch_size=16):
    """Runs (or resumes) until `rounds` rounds are done, or until
    `max_rounds_this_session` rounds have run in this call (used by the
    resume test to simulate a disconnect). Returns (server, history)."""
    os.makedirs(run_dir, exist_ok=True)
    settings = {'clients_per_round': int(clients_per_round), 'local_steps': int(local_steps),
                'local_lr': float(local_lr), 'server_lr': float(server_lr), 'seed': int(seed),
                'dropout': float(dropout), 'model_kind': model_kind,
                'batched': bool(batched), 'group_size': int(group_size)}
    fp = make_fingerprint(settings, train_ids)

    edge_index = torch.load(cfg.EDGE_INDEX_PATH, weights_only=False)
    server = Server(build_model(model_kind, dropout, seed).to(device), lr=server_lr)
    work_model = build_model(model_kind, dropout, seed).to(device)   # reused by every client
    history = []
    best = {'val_macro_auc': None, 'round': None}

    state_path = os.path.join(run_dir, STATE_FILE)
    if os.path.exists(state_path):
        saved = torch.load(state_path, map_location='cpu', weights_only=False)
        if saved['fingerprint'] != fp:
            diffs = [k for k in set(fp) | set(saved['fingerprint'])
                     if fp.get(k) != saved['fingerprint'].get(k)]
            raise SystemExit(
                f"REFUSING TO RESUME {run_dir}: settings/code differ from the saved run in {sorted(diffs)}.\n"
                f"  saved: { {k: saved['fingerprint'].get(k) for k in sorted(diffs)} }\n"
                f"  now:   { {k: fp.get(k) for k in sorted(diffs)} }\n"
                f"Use a new --run-dir for different settings.")
        server.load_state(saved['server'])
        history = saved['history']
        best = saved.get('best', best)
        if verbose:
            print(f"resumed {run_dir} after round {server.rounds_done}")

    ran = 0
    while server.rounds_done < rounds:
        if max_rounds_this_session is not None and ran >= max_rounds_this_session:
            break
        r = server.rounds_done
        passes_before = history[-1]['passes'] if history else 0
        t0 = time.time()
        chosen = choose_students(train_ids, clients_per_round, seed, r)
        if batched:
            outs = train_round_batched(work_model, server.global_state(), chosen, edge_index,
                                       device, local_steps, local_lr, r, seed, group_size)
        else:
            outs = train_round(work_model, server.global_state(), chosen, edge_index, device,
                               local_steps, local_lr, r, seed)
        server.apply_round([o['update'] for o in outs])
        losses = [o['losses'][0] for o in outs]           # loss before local training moved the weights
        train_s = time.time() - t0
        passes = passes_before + len(chosen) * local_steps
        entry = {'round': r, 'students': chosen,
                 'mean_loss': sum(losses) / len(losses),
                 'seconds': train_s,
                 'passes': passes,
                 'train_seconds_total': (history[-1]['train_seconds_total'] if history else 0.0) + train_s,
                 'val_macro_auc': None, 'val_seconds': None}

        last_round = server.rounds_done >= rounds
        crossed = val_every_passes and (passes // val_every_passes) > (passes_before // val_every_passes)
        if val_ids and (crossed or last_round):
            tv = time.time()
            work_model.load_state_dict(server.global_state(), strict=True)
            val = evaluate_batched(work_model, val_ids, edge_index=edge_index,
                                   batch_size=eval_batch_size, device=device)
            entry['val_macro_auc'] = float(val['macro_auc'])
            entry['val_seconds'] = time.time() - tv
            if best['val_macro_auc'] is None or entry['val_macro_auc'] > best['val_macro_auc']:
                best = {'val_macro_auc': entry['val_macro_auc'], 'round': r}
                _atomic_save(server.global_state(), os.path.join(run_dir, BEST_FILE))
        history.append(entry)
        _atomic_save({'server': server.state(), 'history': history, 'fingerprint': fp,
                      'best': best}, state_path)
        _atomic_save(history, os.path.join(run_dir, HISTORY_FILE), use_json=True)
        ran += 1
        if verbose:
            v = (f"  val macro AUC {entry['val_macro_auc']:.4f} ({entry['val_seconds']:.0f}s)"
                 if entry['val_macro_auc'] is not None else "")
            print(f"round {r:4d}  students {len(chosen):3d}  passes {passes:6d}  mean loss "
                  f"{entry['mean_loss']:.4f}  {train_s:.1f}s{v}", flush=True)
    return server, history


def main():
    p = argparse.ArgumentParser(description="Phase 2 step 3: federated training, one client at a time.")
    p.add_argument('--run-dir', required=True)
    p.add_argument('--rounds', type=int, required=True)
    p.add_argument('--clients-per-round', type=int, default=8)
    p.add_argument('--local-steps', type=int, default=1)
    p.add_argument('--local-lr', type=float, default=0.0,
                   help="SGD step size on the device; unused when --local-steps 1")
    p.add_argument('--server-lr', type=float, default=cfg.LEARNING_RATE)
    p.add_argument('--seed', type=int, default=cfg.RANDOM_SEED)
    p.add_argument('--dropout', type=float, default=cfg.DROPOUT)
    p.add_argument('--model', default='v2', choices=['v1', 'v2', 'v2-prior', 'v2-skip'])
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--batched', action='store_true',
                   help="step 4a fast round: students grouped by length, side by side (needs --local-steps 1)")
    p.add_argument('--group-size', type=int, default=8,
                   help="students per GPU group when --batched (recorded in the resume check)")
    p.add_argument('--val-every-passes', type=int, default=800,
                   help="validate every N student-passes (800 = one epoch-equivalent); 0 = only at the end")
    p.add_argument('--eval-batch-size', type=int, default=16)
    args = p.parse_args()
    if args.local_steps > 1 and args.local_lr <= 0:
        raise SystemExit("--local-steps > 1 needs a positive --local-lr")

    with open(cfg.SPLITS_PATH) as f:
        splits = json.load(f)
    train_ids, val_ids = splits['train'], splits['val']
    assert len(train_ids) == cfg.TRAIN_SIZE, f"expected {cfg.TRAIN_SIZE} training students"

    federated_training(args.run_dir, train_ids, args.rounds, args.clients_per_round,
                       args.local_steps, args.local_lr, args.server_lr, args.seed,
                       args.dropout, args.model, torch.device(args.device),
                       batched=args.batched, group_size=args.group_size,
                       val_ids=val_ids, val_every_passes=args.val_every_passes,
                       eval_batch_size=args.eval_batch_size)


if __name__ == '__main__':
    main()