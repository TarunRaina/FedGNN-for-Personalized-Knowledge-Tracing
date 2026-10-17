"""
src/federated/batched_round.py

Phase 2, step 4a -- THE FAST ROUND.

What it does, in plain words
----------------------------
Same job as run_federated.train_round (step 3, one student at a time), but
the GPU computes several students' updates side by side:
  1. sort the round's students by length;
  2. split them into groups of up to `group_size` similar-length students,
     so short students never wait for a long one;
  3. run each group through the verified centralised training function in
     one batch, starting from the SAME global weights (lr 0: nothing moves);
  4. combine the groups into one average where EVERY STUDENT COUNTS ONCE.

Why this is still federated
---------------------------
Each student's update depends only on that student's answers. The batch
computes those updates side by side and adds them -- which is the server's
job, done in the same pass. Step 3 proved one-at-a-time == batched.

Only local_steps == 1
---------------------
With one step per round every device starts from the identical global
model, so their work can share one batch. With more steps the devices'
models drift apart; that is step 4b, built only if step 7 shows it is needed.

THE EQUAL-VOTE TRAP
-------------------
A group's gradient from train_one_epoch_batched is the mean over ITS
students. Groups can have different sizes (10 students -> 8 + 2). Averaging
group means 50/50 would give the 2 students four times the say. So each
group's mean is weighted by (group size / students in round). Tested with
uneven groups, with a negative test for the 50/50 version.

Interface
---------
Returns a list with ONE entry: the round's equal-vote average is already
computed here. The (unchanged) server then averages a list of one, which is
that entry itself. client.py and server.py are untouched.
The entry also carries the per-group means and sizes (for the negative test
and for logging); the server ignores them.

Randomness: before each group, torch is seeded from (seed, round, group
number), so a resumed run reproduces the same dropout.
"""

import os
import sys

import torch

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))       # src/federated
_SRC_DIR = os.path.dirname(_THIS_DIR)                         # src
_PROJECT_ROOT = os.path.dirname(_SRC_DIR)                     # fedgkt_v2/
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.utils import config as cfg
from src.training.centralised_batched import train_one_epoch_batched

_LENGTHS = {}   # student -> sequence length, read once per process


def _length(uid):
    if uid not in _LENGTHS:
        _LENGTHS[uid] = int(torch.load(os.path.join(cfg.PKG_DIR, f'pkg_{uid}.pt'),
                                       weights_only=False)['exercise_idx'].shape[0])
    return _LENGTHS[uid]


def make_groups(student_ids, group_size):
    """Sort by length (ties by id, so it is deterministic), then cut into
    consecutive groups of up to group_size."""
    order = sorted((int(u) for u in student_ids), key=lambda u: (_length(u), u))
    return [order[i:i + group_size] for i in range(0, len(order), group_size)]


def group_seed(base_seed, round_idx, group_idx):
    return (int(base_seed) * 1_000_003 + int(round_idx) * 10_007 + 7_919 * (int(group_idx) + 1)) % (2**31 - 1)


def train_round_batched(work_model, global_state, student_ids, edge_index, device,
                        local_steps, local_lr, round_idx, seed, group_size):
    if local_steps != 1:
        raise SystemExit("batched rounds (step 4a) support only --local-steps 1; "
                         "more local steps need step 4b")
    assert group_size >= 1
    K = len(student_ids)
    assert K > 0, "BUG: empty round"

    work_model.load_state_dict(global_state, strict=True)
    work_model.to(device)
    opt = torch.optim.SGD(work_model.parameters(), lr=0.0)   # weights never move here
    names = [n for n, _ in work_model.named_parameters()]

    avg = {n: torch.zeros_like(p, dtype=torch.float64, device='cpu')
           for n, p in work_model.named_parameters()}
    group_means, group_sizes, weighted_loss = [], [], 0.0

    for gi, group in enumerate(make_groups(student_ids, group_size)):
        torch.manual_seed(group_seed(seed, round_idx, gi))
        if device.type == 'cuda':
            torch.cuda.manual_seed_all(group_seed(seed, round_idx, gi))
        loss = train_one_epoch_batched(work_model, opt, group, edge_index, len(group), device)
        g_mean = {}
        for n, p in work_model.named_parameters():
            assert p.grad is not None, f"BUG: no gradient for {n}"
            g_mean[n] = p.grad.detach().to('cpu', torch.float64).clone()
            avg[n] += (len(group) / K) * g_mean[n]          # every student counts once
        group_means.append(g_mean)
        group_sizes.append(len(group))
        weighted_loss += (len(group) / K) * loss

    assert sum(group_sizes) == K, "BUG: grouping lost or duplicated a student"
    return [{'update': avg, 'n_answers': sum(_length(u) for u in student_ids),
             'losses': [weighted_loss], 'group_means': group_means, 'group_sizes': group_sizes}]