import sys, re, csv, torch
sys.path.insert(0, '/home/claude/fedgkt_v2')
from src.utils import config as cfg
import glob
rows = list(csv.DictReader(open(glob.glob('/mnt/user-data/uploads/*learned_priors.csv')[0], newline='')))
idx = [int(r['concept_idx']) for r in rows]; names = {int(r['concept_idx']): r['concept_name'].strip() for r in rows}
csv_p = torch.tensor([float(r['learned_prior']) for r in sorted(rows, key=lambda r: int(r['concept_idx']))])
sd = torch.load('/home/claude/fedgkt_v2/model_training/best.pt', weights_only=False)
ck_p = torch.sigmoid(sd['concept_prior_logit'])

print("=== CHECK 1: CSV priors vs best.pt ===")
print(f"rows {len(rows)}, indices exactly 0..834: {sorted(idx) == list(range(835))}")
d = (csv_p - ck_p).abs(); TOL = 5e-7   # CSV stores 6 decimals -> rounding error at most 5e-7
print(f"max |CSV - checkpoint| = {d.max():.2e} (tolerance {TOL:.0e}) -> {'PASS' if d.max() <= TOL else 'FAIL'}")
bad = csv_p.clone(); bad[300] += 1e-3
print(f"negative test (one value nudged by 0.001): max diff {(bad - ck_p).abs().max():.2e} -> "
      f"{'caught' if (bad - ck_p).abs().max() > TOL else 'NOT caught'}")

print("\n=== CHECK 2: edge direction, using names ===")
ei = torch.load(cfg.EDGE_INDEX_PATH, weights_only=False)
def split(n):
    m = re.match(r'^(.*)_(\d+(?:\.\d+)?)$', n); return (m.group(1), float(m.group(2))) if m else (n, None)
def tally(src, dst):
    up = down = 0
    for s, t in zip(src, dst):
        (a, x), (b, y) = split(names[s]), split(names[t])
        if a == b and x is not None and y is not None and x != y:
            up += x < y; down += x > y
    return up, down
up, down = tally(ei[0].tolist(), ei[1].tolist())
print(f"edges between same-topic numbered concepts (e.g. addition_1 / addition_2): {up + down}")
print(f"  lower number -> higher number: {up}    higher -> lower: {down}")
print(f"  reading edge_index[0] as the PREREQUISITE: {'consistent' if up > 10 * max(down, 1) else 'NOT clearly consistent'}")
u2, d2 = tally(ei[1].tolist(), ei[0].tolist())
print(f"negative test (edges flipped): low->high {u2}, high->low {d2} -> "
      f"{'caught' if not (u2 > 10 * max(d2, 1)) else 'NOT caught'}")
print("\nexamples (edge_index[0] -> edge_index[1]):")
for s, t in list(zip(ei[0].tolist(), ei[1].tolist()))[:8]: print(f"  {names[s]:40s} -> {names[t]}")