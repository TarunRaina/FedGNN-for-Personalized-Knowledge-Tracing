"""
src/federated/server.py

Phase 2, step 3 -- THE SERVER.

What it does, in plain words
----------------------------
It holds the shared model. Each round it receives one update per student
(from client.run_client), averages them with an EQUAL vote per student,
and uses Adam to move the shared model. It never sees any student's answers.

Design decisions this file implements (decided before code)
-----------------------------------------------------------
D3 equal vote: every student's update counts the same, regardless of how
  many answers they have. This matches the centralised objective, where each
  student's loss is divided by its own length. (Standard FedAvg weights by
  data size instead -- that would let student 21419, 9,548 answers, count
  ~300x more than student 233536, 30 answers. NOT done here.)
D3 Option C: Adam lives here. The averaged update is placed in .grad and
  one Adam step is taken -- the exact torch.optim.Adam the centralised runs
  use. Construction checked in screen.py line 208:
      torch.optim.Adam(model.parameters(), lr=lr)
  i.e. default betas (0.9, 0.999), eps 1e-8, no weight decay. Same here.

Why this reproduces centralised training at the simplest setting
----------------------------------------------------------------
Centralised: one Adam step on the batch gradient of B students, where each
student's loss is divided by (its length x B). That gradient equals the
plain average of the B single-student gradients (each divided by its own
length). With local_steps = 1 a client's update IS its single-student
gradient. So averaging B client updates equally and taking one Adam step is
the same operation -- up to float rounding. See client.py for the test
conditions and pre-declared tolerances.

Averaging is done in float64, then cast to float32 for Adam, so the
averaging itself adds no meaningful rounding.

Parameters are matched by NAME, never by position.

Resume
------
state() returns everything needed to continue: model weights, Adam's
memory, and the number of rounds done. load_state() puts it back.
Choosing which students join a round, and saving to disk, belong to
run_federated.py (next file), not here.
"""

import os
import sys

import torch

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))       # src/federated
_SRC_DIR = os.path.dirname(_THIS_DIR)                         # src
_PROJECT_ROOT = os.path.dirname(_SRC_DIR)                     # fedgkt_v2/
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)


def average_equal(updates):
    """
    updates: list of {name: float64 tensor}, one per student.
    Returns {name: float64 tensor} -- the plain mean, every student weight 1/K.
    Deliberately takes NO sizes or weights as input, so a data-size weighting
    cannot sneak in through an argument.
    """
    assert len(updates) > 0, "BUG: no updates to average"
    names = set(updates[0].keys())
    for u in updates[1:]:
        assert set(u.keys()) == names, "BUG: clients returned different parameter names"
    k = len(updates)
    return {n: sum(u[n] for u in updates) / k for n in names}


class Server:
    def __init__(self, model, lr):
        """model: the shared model object (weights live here). lr: Adam lr."""
        self.model = model
        self.adam = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.rounds_done = 0

    def global_state(self):
        """A detached CPU copy of the shared weights, to send to clients."""
        return {n: p.detach().to('cpu').clone() for n, p in self.model.named_parameters()}

    def apply_round(self, updates):
        """Average the round's client updates equally, take one Adam step.
        Returns the averaged update (float64), for logging and testing."""
        avg = average_equal(updates)
        params = dict(self.model.named_parameters())
        assert set(avg.keys()) == set(params.keys()), "BUG: update names != model parameter names"
        self.adam.zero_grad(set_to_none=False)
        for n, p in params.items():
            p.grad = avg[n].to(device=p.device, dtype=p.dtype).reshape(p.shape).clone()
        self.adam.step()
        self.rounds_done += 1
        return avg

    def state(self):
        return {'model': {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()},
                'adam': self.adam.state_dict(),
                'rounds_done': self.rounds_done}

    def load_state(self, st):
        self.model.load_state_dict(st['model'], strict=True)
        self.adam.load_state_dict(st['adam'])
        self.rounds_done = int(st['rounds_done'])