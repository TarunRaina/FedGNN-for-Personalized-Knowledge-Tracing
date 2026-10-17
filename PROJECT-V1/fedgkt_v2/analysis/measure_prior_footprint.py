"""
Step 1 measurement -- which rows of concept_prior_logit receive gradient.

Prediction under test (stated before running):
  For a prediction on target concept t, the prior gradient is nonzero on
  EXACTLY  ({t} U ancestors of t within 3 hops)  intersected with  untouched concepts,
  where "ancestor" follows edge_index[0] -> edge_index[1] (PyG source_to_target flow).

Checked as exact set equality at every step, not as a subset, so that the
negative tests below have something to fail on.

Model: FedGKTv2 with model_training/best.pt, eval() mode (dropout off).
Replay: the reference pkg.py, in its predict-before-update order.
Nothing in fedgkt_v2/src is modified.
"""
import os, sys, time
import torch

ROOT = '/home/claude/fedgkt_v2'
sys.path.insert(0, ROOT)
from src.utils import config as cfg
from src.models.fedgkt_v2 import FedGKTv2
from src.data.pkg import PersonalKnowledgeGraph

STUDENTS = [233536, 45224, 21419]
FA = cfg.FEATURE_IDX['first_attempt']

ei = torch.load(cfg.EDGE_INDEX_PATH, weights_only=False)
src, dst = ei[0].tolist(), ei[1].tolist()
N = cfg.NUM_NODES

# ── graph facts ────────────────────────────────────────────────────────
parents = [set() for _ in range(N)]
children = [set() for _ in range(N)]
for s, d in zip(src, dst):
    parents[d].add(s); children[s].add(d)
indeg = [len(p) for p in parents]; outdeg = [len(c) for c in children]
print(f"edges {ei.shape[1]}, self-loops in file {sum(s == d for s, d in zip(src, dst))}, "
      f"duplicates {ei.shape[1] - len(set(zip(src, dst)))}")
print(f"in-degree 0: {sum(i == 0 for i in indeg)}   out-degree 0: {sum(o == 0 for o in outdeg)}   "
      f"isolated: {sum(i == 0 and o == 0 for i, o in zip(indeg, outdeg))}")

def within_k(t, k, nbrs):
    seen, frontier = set(), {t}
    for _ in range(k):
        frontier = {u for v in frontier for u in nbrs[v]} - seen - {t}
        seen |= frontier
    return seen

# ── model ──────────────────────────────────────────────────────────────
model = FedGKTv2()
sd = torch.load(os.path.join(ROOT, 'model_training/best.pt'), map_location='cpu', weights_only=False)
missing, unexpected = model.load_state_dict(sd, strict=True), None
model.eval()
pri = model.learned_priors()
print(f"\nbest.pt priors: min {pri.min():.3f} (idx {int(pri.argmin())})  max {pri.max():.3f} "
      f"(idx {int(pri.argmax())})  moved >0.01 from 0.5: {int(((pri - 0.5).abs() > 0.01).sum())}/{N}")

# ── replay + per-step check ────────────────────────────────────────────
def run(uid, predicted_fn, label, verbose=True, max_steps=None):
    d = torch.load(os.path.join(cfg.PKG_DIR, f'pkg_{uid}.pt'), weights_only=False)
    ex, co, td = d['exercise_idx'], d['correct'], d['time_done']
    L = ex.shape[0] if max_steps is None else min(max_steps, ex.shape[0])
    pkg = PersonalKnowledgeGraph()
    n_equal = 0; extra_actual = 0; missing_actual = 0; first_bad = None
    union = set(); sizes = []; min_abs = float('inf'); t0 = time.time()
    for step in range(L):
        t = int(ex[step]); pkg.refresh_time_decay(int(td[step]))
        x = pkg.get_x().clone()
        model.zero_grad(set_to_none=True)
        model(x, ei, t).backward()
        g = model.concept_prior_logit.grad
        nz = (g != 0).nonzero().flatten()
        actual = set(nz.tolist())
        if len(nz): min_abs = min(min_abs, g[nz].abs().min().item())
        untouched = set((x[:, FA] == 0).nonzero().flatten().tolist())
        predicted = predicted_fn(t) & untouched
        if actual == predicted: n_equal += 1
        else:
            extra_actual += len(actual - predicted); missing_actual += len(predicted - actual)
            if first_bad is None: first_bad = (step, t, sorted(actual - predicted)[:5], sorted(predicted - actual)[:5])
        union |= actual; sizes.append(len(actual))
        pkg.update(t, int(co[step]), int(td[step]))
    attempted = set(ex[:L].tolist())
    if verbose:
        print(f"\n[{label}] student {uid}: {L} steps in {time.time() - t0:.1f}s")
        print(f"  steps where actual == predicted: {n_equal}/{L}")
        print(f"  rows nonzero but NOT predicted (total over steps): {extra_actual}")
        print(f"  rows predicted but zero        (total over steps): {missing_actual}")
        if first_bad: print(f"  first mismatch: step {first_bad[0]}, target {first_bad[1]}, "
                            f"unexpected {first_bad[2]}, missing {first_bad[3]}")
        print(f"  rows per step: min {min(sizes)}, mean {sum(sizes)/len(sizes):.2f}, max {max(sizes)}")
        print(f"  smallest nonzero |grad|: {min_abs:.3e}")
        print(f"  distinct concepts attempted: {len(attempted)}   union of rows touched by gradient: {len(union)}")
        print(f"  attempted subset of union: {attempted <= union}   extra (not attempted): {len(union - attempted)}")
    return n_equal == L

pred3 = lambda t: {t} | within_k(t, 3, parents)
print("\n=========== MAIN CHECK: {t} + ancestors within 3 hops, untouched only ===========")
ok = [run(uid, pred3, 'main') for uid in STUDENTS]
print(f"\nALL STUDENTS EXACT AT EVERY STEP: {all(ok)}")

print("\n=========== NEGATIVE TESTS (each must FAIL) on student 45224 ===========")
negs = {
    'descendants instead of ancestors': lambda t: {t} | within_k(t, 3, children),
    'only 2 hops':                       lambda t: {t} | within_k(t, 2, parents),
    '4 hops (too many)':                 lambda t: {t} | within_k(t, 4, parents),
    'target excluded':                   lambda t: within_k(t, 3, parents),
}
for name, fn in negs.items():
    passed = run(45224, fn, name, verbose=False)
    print(f"  {name:36s} -> check {'PASSED (BAD: test did not catch it)' if passed else 'FAILED (good: caught)'}")

# touched-concepts negative test needs the untouched filter removed
def run_no_untouched_filter(uid):
    d = torch.load(os.path.join(cfg.PKG_DIR, f'pkg_{uid}.pt'), weights_only=False)
    ex, co, td = d['exercise_idx'], d['correct'], d['time_done']
    pkg = PersonalKnowledgeGraph(); bad = 0
    for step in range(ex.shape[0]):
        t = int(ex[step]); pkg.refresh_time_decay(int(td[step])); x = pkg.get_x().clone()
        model.zero_grad(set_to_none=True); model(x, ei, t).backward()
        actual = set((model.concept_prior_logit.grad != 0).nonzero().flatten().tolist())
        bad += actual != pred3(t)
        pkg.update(t, int(co[step]), int(td[step]))
    return bad == 0
print(f"  {'touched concepts not excluded':36s} -> check "
      f"{'PASSED (BAD)' if run_no_untouched_filter(45224) else 'FAILED (good: caught)'}")