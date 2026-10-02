"""
timing_probe.py   (fedgkt_v2 root, next to run_screen.py)

Phase 2, step 2 -- the timing probe.

QUESTION: in federation, each student's device trains its own copy of the
model, alone. How long does that take, compared with the centralised trick
of training 8 students together in one batch?

HOW: it does not re-implement training. It calls the EXISTING
train_one_epoch_batched() from src/training/centralised_batched.py and
times it:
  - each of the three known students ALONE (batch of 1)
  - 8 copies of student 45224 in ONE batch, vs the same student alone,
    to measure what batching is worth on identical work

CORRECTNESS CHECKS (run first; timing is skipped if any fails):
  A. The loss train_one_epoch_batched reports for one student equals an
     independent step-by-step replay with the reference pkg.py
     (dropout off, same weights). Tolerance 1e-5.
  B. Negative test: the same comparison against a deliberately wrong
     replay (one answer flipped) MUST fail.
  C. 8 copies of a student in one batch give the same gradient as that
     student alone (the per-student 1/(length x B) weighting). Tolerance 1e-6.
Nothing in src/ is modified. No checkpoint is written.

RUN (Colab, T4). Copy the project to local disk first, so Drive file-reading
time is not mixed into the training time:
    !cp -r /content/drive/MyDrive/<your path>/fedgkt_v2 /content/fedgkt_v2
    %cd /content/fedgkt_v2
    !python timing_probe.py --device cuda --out /content/drive/MyDrive/timing_probe.json
CPU check (laptop or sandbox):
    python timing_probe.py --device cpu --repeats 1
"""

import argparse
import json
import os
import statistics
import sys
import time

import torch
import torch.nn.functional as F

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

from src.utils import config as cfg
from src.models.fedgkt_v2 import FedGKTv2
from src.data.pkg import PersonalKnowledgeGraph
from src.training.centralised_batched import train_one_epoch_batched

STUDENTS = [233536, 45224, 21419]
BATCH_STUDENT = 45224          # median-length student, used for the batch-of-8 comparison
LOSS_TOL = 1e-5
GRAD_TOL = 1e-6


def build_model(seed, dropout):
    """Same seed -> same starting weights. Dropout is read at build time."""
    saved = cfg.DROPOUT
    cfg.DROPOUT = dropout
    torch.manual_seed(seed)
    model = FedGKTv2()
    cfg.DROPOUT = saved
    return model


def load_seq(uid):
    d = torch.load(os.path.join(cfg.PKG_DIR, f'pkg_{uid}.pt'), weights_only=False)
    return d['exercise_idx'], d['correct'], d['time_done']


def reference_loss(model, uid, edge_index, flip_step=None):
    """Independent step-by-step replay with pkg.py: mean clamped BCE over the
    student's sequence, BEFORE any weight update. flip_step flips one answer
    (used only by the negative test)."""
    ex, co, td = load_seq(uid)
    pkg = PersonalKnowledgeGraph()
    model.train()                       # dropout is 0 in these models, so train == eval
    losses = []
    with torch.no_grad():
        for s in range(ex.shape[0]):
            t, c = int(ex[s]), float(co[s])
            if s == flip_step:
                c = 1.0 - c
            pkg.refresh_time_decay(int(td[s]))
            p = model(pkg.get_x(), edge_index, t).clamp(1e-7, 1 - 1e-7)
            losses.append(F.binary_cross_entropy(p.view(1), torch.tensor([c])).item())
            pkg.update(int(ex[s]), int(co[s]), int(td[s]))   # true answer always used for the state
    return sum(losses) / len(losses)


def batched_loss_and_grads(model, ids, batch_size, edge_index, device):
    """One call of the real training function. Returns its reported loss and
    the gradients it accumulated (left in .grad after its optimizer.step())."""
    model = model.to(device)
    opt = torch.optim.SGD(model.parameters(), lr=0.0)   # lr 0: weights do not move
    loss = train_one_epoch_batched(model, opt, ids, edge_index, batch_size, device)
    grads = {n: p.grad.detach().cpu().clone() for n, p in model.named_parameters()}
    return loss, grads


class PeakMemory:
    """Peak memory during a block. CPU: process RAM (RSS), sampled every 0.1 s
    from /proc (Linux, incl. Colab; None elsewhere). GPU: torch's own peak."""
    def __init__(self, device):
        self.device, self.peak_rss_mb, self.gpu_peak_mb = device, None, None
    def _rss(self):
        try:
            with open('/proc/self/status') as f:
                for line in f:
                    if line.startswith('VmRSS:'):
                        return int(line.split()[1]) / 1024
        except OSError:
            return None
    def _watch(self):
        while not self._stop:
            r = self._rss()
            if r is not None:
                self.peak_rss_mb = max(self.peak_rss_mb or 0, r)
            time.sleep(0.1)
    def __enter__(self):
        import threading
        self._stop = False
        self.start_rss_mb = self._rss()
        self._t = threading.Thread(target=self._watch, daemon=True); self._t.start()
        if self.device.type == 'cuda':
            torch.cuda.reset_peak_memory_stats()
        return self
    def __exit__(self, *exc):
        self._stop = True; self._t.join()
        if self.device.type == 'cuda':
            self.gpu_peak_mb = torch.cuda.max_memory_allocated() / 2**20
    def text(self):
        out = []
        if self.peak_rss_mb is not None:
            out.append(f"RAM peak {self.peak_rss_mb:,.0f} MB (start {self.start_rss_mb:,.0f})")
        if self.gpu_peak_mb is not None:
            out.append(f"GPU peak {self.gpu_peak_mb:,.0f} MB")
        return ", ".join(out) or "memory n/a"


def sync(device):
    if device.type == 'cuda':
        torch.cuda.synchronize()


def time_call(ids, batch_size, edge_index, device, repeats, seed):
    """Wall-clock time of train_one_epoch_batched on `ids`, real dropout,
    Adam as in the centralised runs. One untimed warm-up call first."""
    model = build_model(seed, cfg.DROPOUT).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.LEARNING_RATE)
    train_one_epoch_batched(model, opt, ids, edge_index, batch_size, device)   # warm-up
    times = []
    for _ in range(repeats):
        sync(device)
        t0 = time.perf_counter()
        train_one_epoch_batched(model, opt, ids, edge_index, batch_size, device)
        sync(device)
        times.append(time.perf_counter() - t0)
    return statistics.median(times), times


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    ap.add_argument('--repeats', type=int, default=3)
    ap.add_argument('--seed', type=int, default=cfg.RANDOM_SEED)
    ap.add_argument('--out', default=None, help="optional JSON path for the results")
    ap.add_argument('--skip-round-estimate', action='store_true')
    ap.add_argument('--students', type=int, nargs='+', default=STUDENTS,
                    help="which of the three students to use (sandbox: the two short ones)")
    args = ap.parse_args()

    device = torch.device(args.device)
    students = args.students
    assert BATCH_STUDENT in students and 233536 in students, "checks B/C need 233536 and 45224"
    edge_index = torch.load(cfg.EDGE_INDEX_PATH, weights_only=False)
    import torch_geometric
    print(f"torch {torch.__version__} | PyG {torch_geometric.__version__} | device {device}"
          + (f" ({torch.cuda.get_device_name(0)})" if device.type == 'cuda' else ""))

    # ── correctness checks (CPU, dropout off) ─────────────────────────────
    print("\n=== CHECK A: real training function's loss == independent replay ===")
    ok = True
    for uid in students:
        ref = reference_loss(build_model(args.seed, 0.0), uid, edge_index)
        with PeakMemory(torch.device('cpu')) as mem:
            got, _ = batched_loss_and_grads(build_model(args.seed, 0.0), [uid], 1, edge_index, torch.device('cpu'))
        d = abs(ref - got)
        print(f"  {uid:>6}: replay {ref:.7f}  training fn {got:.7f}  diff {d:.2e}  "
              f"{'PASS' if d <= LOSS_TOL else 'FAIL'}   [CPU training call: {mem.text()}]", flush=True)
        ok &= d <= LOSS_TOL

    print("\n=== CHECK B (negative): one answer flipped in the replay must be caught ===")
    bad = reference_loss(build_model(args.seed, 0.0), 233536, edge_index, flip_step=5)
    got, _ = batched_loss_and_grads(build_model(args.seed, 0.0), [233536], 1, edge_index, torch.device('cpu'))
    caught = abs(bad - got) > LOSS_TOL
    print(f"  diff {abs(bad - got):.2e} -> {'caught (good)' if caught else 'NOT caught (check is useless)'}")
    ok &= caught

    print(f"\n=== CHECK C: 8 copies of {BATCH_STUDENT} in one batch == the student alone ===")
    _, g1 = batched_loss_and_grads(build_model(args.seed, 0.0), [BATCH_STUDENT], 1, edge_index, torch.device('cpu'))
    _, g8 = batched_loss_and_grads(build_model(args.seed, 0.0), [BATCH_STUDENT] * 8, 8, edge_index, torch.device('cpu'))
    gd = max((g1[n] - g8[n]).abs().max().item() for n in g1)
    print(f"  max gradient difference {gd:.2e}  {'PASS' if gd <= GRAD_TOL else 'FAIL'}")
    ok &= gd <= GRAD_TOL

    if not ok:
        print("\nA correctness check failed -- NOT timing anything. Send this output back.")
        sys.exit(1)
    print("\nAll checks passed.")

    # ── timing ────────────────────────────────────────────────────────────
    print(f"\n=== TIMING on {device} (median of {args.repeats}, after one warm-up each) ===")
    results = {'torch': torch.__version__, 'pyg': torch_geometric.__version__,
               'device': str(device), 'repeats': args.repeats, 'alone': {}}
    lengths = {uid: load_seq(uid)[0].shape[0] for uid in students}
    for uid in students:
        with PeakMemory(device) as mem:
            med, all_t = time_call([uid], 1, edge_index, device, args.repeats, args.seed)
        results['alone'][uid] = {'length': lengths[uid], 'seconds': med, 'all': all_t,
                                 'ms_per_answer': 1000 * med / lengths[uid],
                                 'ram_peak_mb': mem.peak_rss_mb, 'gpu_peak_mb': mem.gpu_peak_mb}
        print(f"  {uid:>6} alone ({lengths[uid]:>5} answers): {med:8.2f} s   "
              f"{1000 * med / lengths[uid]:6.2f} ms per answer   [{mem.text()}]", flush=True)

    t8, all8 = time_call([BATCH_STUDENT] * 8, 8, edge_index, device, args.repeats, args.seed)
    t1 = results['alone'][BATCH_STUDENT]['seconds']
    results['batch8'] = {'seconds': t8, 'all': all8, 'alone_x8_seconds': 8 * t1, 'speedup': 8 * t1 / t8}
    print(f"\n  8 copies of {BATCH_STUDENT}, one batch: {t8:.2f} s   vs one-at-a-time x8: {8 * t1:.2f} s"
          f"   -> batching is {8 * t1 / t8:.1f}x faster")

    # straight-line fit: time = a + b * answers (over the timed students)
    xs = [lengths[u] for u in students]; ys = [results['alone'][u]['seconds'] for u in students]
    mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
    b = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sum((x - mx) ** 2 for x in xs)
    a = my - b * mx
    results['fit'] = {'a_seconds': a, 'b_seconds_per_answer': b}
    print(f"\n  fit: time alone ~= {a:.3f} s + {1000 * b:.2f} ms x answers")

    # ── one federated round where all 800 training students train once, alone ─
    if not args.skip_round_estimate and os.path.exists(cfg.SPLITS_PATH):
        with open(cfg.SPLITS_PATH) as f:
            train_ids = json.load(f)['train']
        total = sum(load_seq(u)[0].shape[0] for u in train_ids)
        est = len(train_ids) * a + b * total
        est8 = est / results['batch8']['speedup']
        results['round_estimate'] = {'n_students': len(train_ids), 'total_answers': total,
                                     'one_at_a_time_seconds': est, 'if_batched_8_seconds': est8}
        print(f"\n=== ONE ROUND, all {len(train_ids)} training students, {total:,} answers ===")
        print(f"  one device at a time: ~{est / 60:.0f} min per round")
        print(f"  if devices could be batched 8 at a time (step 4): ~{est8 / 60:.0f} min per round")
    else:
        print("\n(round estimate skipped: no student_splits.json here, or --skip-round-estimate)")

    if args.out:
        with open(args.out, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nSaved {args.out}")


if __name__ == '__main__':
    main()