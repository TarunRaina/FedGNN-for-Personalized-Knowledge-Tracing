"""
src/federated/client.py

Phase 2, step 3 -- ONE STUDENT'S DEVICE.

What it does, in plain words
----------------------------
The server sends the current shared model. The device trains it on its own
student's answers only, and sends back how the model wanted to change.
The student's answers never leave this function.

Design decisions this file implements (decided before code)
-----------------------------------------------------------
D3 / Option C: the device takes PLAIN steps (SGD). Adam lives on the server.
  - One local step = one full pass over the student's sequence, exactly as
    one centralised batch is one optimizer step.
  - The device returns its UPDATE = the SUM of the gradients of its local
    steps, kept in float64. The server averages updates and feeds the
    average to Adam as if it were a gradient.
  - With local_steps=1, a student's update is exactly the gradient the
    centralised code computes for that student alone. (We return the summed
    gradients rather than "new weights minus old weights": subtracting two
    nearly equal float32 weight tensors and dividing by a small learning
    rate would amplify rounding error ~1000x.)
D3: equal vote per student. Each student's loss is already divided by its
  own length inside train_one_epoch_batched (normaliser = length x B, with
  B = 1 here), so every student's update is on the same scale. The server
  must then average with EQUAL weights. This file does not weight anything.

How it trains
-------------
It reuses the verified centralised training function
train_one_epoch_batched() with a batch of ONE student and an SGD optimiser,
so the predict-before-update replay, the 1/length loss weighting and the
chunked backward are the exact code that produced the Phase 1 numbers.
That function calls zero_grad() once per batch, backward(), then step();
.grad after the call therefore holds that step's gradient. Checked in the
source (lines 198 and 253 of centralised_batched.py), not assumed.

Randomness (dropout)
--------------------
Before training, the device seeds torch from (base_seed, round, student).
So a student's dropout draws depend only on WHICH round and WHICH student,
not on the order clients run in or on what ran before. That makes resume
exact without saving any per-client random state.

Step 3 scope
------------
EVERY parameter is shared and returned, including the head. Keeping the
head on the device (FedPer) is step 6, deliberately not here.
Parameters are matched by NAME everywhere, never by position
(concept_prior_logit is listed first by PyTorch despite being created last).

Equivalence test conditions (stated BEFORE the test is written)
----------------------------------------------------------------
"One federated round with local_steps=1 == one centralised step on the
same students" is only expected to hold when:
  1. the head is shared and averaged like everything else (true in step 3);
  2. the round contains exactly the students of one centralised batch;
  3. dropout is OFF (cfg.DROPOUT = 0 at model build). Centralised training
     draws dropout for B students at once, a device draws for one; the draws
     can never line up, so with dropout on a mismatch would mean nothing.
Pre-declared tolerances:
  - averaged update vs centralised batch gradient: <= 1e-6 (max abs)
  - weights after the server's Adam step vs centralised weights after its
    Adam step: <= 1e-5 (max abs). Looser because Adam's first step divides
    each gradient by its own size: for near-zero gradients, harmless
    ~1e-8 rounding noise changes that ratio noticeably; with lr = 1e-3 the
    expected effect on weights is up to ~1e-5.
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


def client_seed(base_seed, round_idx, user_id):
    """Deterministic, order-independent seed for one student in one round.
    Plain integer arithmetic (not Python's hash(), which is randomised per
    process for strings and so would break resume)."""
    return (int(base_seed) * 1_000_003 + int(round_idx) * 10_007 + int(user_id)) % (2**31 - 1)


def run_client(model, global_state, user_id, edge_index, device,
               local_steps, local_lr, round_idx, base_seed):
    """
    model:        a model object to train in place (its weights are
                  overwritten from global_state first -- reused across
                  clients to avoid rebuilding).
    global_state: {name: tensor} -- the shared weights the server sent.
    user_id:      the one student this device holds.
    local_steps:  plain SGD steps, each one full pass over the sequence.
    local_lr:     SGD step size on the device. Irrelevant when
                  local_steps == 1 (the update is the gradient itself).

    Returns a dict:
        'update':    {name: float64 CPU tensor} -- sum of local gradients
        'n_answers': length of the student's sequence (reported for
                     logging and for the negative test only; the server
                     must NOT use it as a weight)
        'losses':    the training loss of each local step
    """
    assert local_steps >= 1, "local_steps must be at least 1"

    model.load_state_dict(global_state, strict=True)
    model.to(device)
    names = [n for n, _ in model.named_parameters()]
    assert set(names) == set(global_state.keys()), "BUG: model and global_state parameter names differ"

    torch.manual_seed(client_seed(base_seed, round_idx, user_id))
    if device.type == 'cuda':
        torch.cuda.manual_seed_all(client_seed(base_seed, round_idx, user_id))

    opt = torch.optim.SGD(model.parameters(), lr=local_lr)
    update = {n: torch.zeros_like(p, dtype=torch.float64, device='cpu')
              for n, p in model.named_parameters()}
    losses = []

    for _ in range(local_steps):
        loss = train_one_epoch_batched(model, opt, [user_id], edge_index, 1, device)
        losses.append(loss)
        for n, p in model.named_parameters():
            assert p.grad is not None, f"BUG: no gradient for {n}"
            update[n] += p.grad.detach().to('cpu', torch.float64)

    n_answers = int(torch.load(os.path.join(cfg.PKG_DIR, f'pkg_{user_id}.pt'),
                               weights_only=False)['exercise_idx'].shape[0])

    return {'update': update, 'n_answers': n_answers, 'losses': losses}