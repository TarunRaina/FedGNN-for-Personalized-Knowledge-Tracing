"""How much of the attempted set can a server recover from the union of moved prior rows,
knowing only the public graph? Rule: a row u in the union that is NOT a <=3-hop ancestor of
any other row in the union can only be there because u itself was attempted."""
import sys, torch
sys.path.insert(0, '/home/claude/fedgkt_v2')
from src.utils import config as cfg
from src.models.fedgkt_v2 import FedGKTv2
from src.data.pkg import PersonalKnowledgeGraph
ei = torch.load(cfg.EDGE_INDEX_PATH, weights_only=False); N = cfg.NUM_NODES
parents = [set() for _ in range(N)]
for s, d in zip(ei[0].tolist(), ei[1].tolist()): parents[d].add(s)
def anc3(t):
    seen, fr = set(), {t}
    for _ in range(3):
        fr = {u for v in fr for u in parents[v]} - seen - {t}; seen |= fr
    return seen
m = FedGKTv2(); m.load_state_dict(torch.load('/home/claude/fedgkt_v2/model_training/best.pt', weights_only=False)); m.eval()
print(f"lowest prior {m.learned_priors().min().item():.5f}")
for uid in (233536, 45224, 21419):
    d = torch.load(f'{cfg.PKG_DIR}/pkg_{uid}.pt', weights_only=False)
    ex, co, td = d['exercise_idx'], d['correct'], d['time_done']
    pkg = PersonalKnowledgeGraph(); U = set()
    for i in range(ex.shape[0]):
        t = int(ex[i]); pkg.refresh_time_decay(int(td[i])); x = pkg.get_x().clone()
        m.zero_grad(set_to_none=True); m(x, ei, t).backward()
        U |= set((m.concept_prior_logit.grad != 0).nonzero().flatten().tolist())
        pkg.update(t, int(co[i]), int(td[i]))
    A = set(ex.tolist())
    covered = set().union(*(anc3(v) for v in U))
    sure = {u for u in U if u not in covered}
    print(f"{uid}: union {len(U)}, attempted {len(A)}, server can say 'certainly attempted' for {len(sure)} "
          f"(all correct: {sure <= A}) = {len(sure & A)}/{len(A)} of attempted")