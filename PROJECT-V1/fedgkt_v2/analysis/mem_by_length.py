"""Peak RAM of ONE training call on 21419's first N answers, in a fresh process.
Reads a truncated COPY of the data in /tmp/trunc; project files untouched."""
import sys, os, time, threading, torch
sys.path.insert(0, '/home/claude/fedgkt_v2')
from src.utils import config as cfg
N = int(sys.argv[1])
d = torch.load(f'{cfg.PKG_DIR}/pkg_21419.pt', weights_only=False)
t = {k: (v[:N] if torch.is_tensor(v) else v) for k, v in d.items()}
torch.save(t, f'/tmp/trunc/pkg_21419.pt'); cfg.PKG_DIR = '/tmp/trunc'
from src.models.fedgkt_v2 import FedGKTv2
from src.training.centralised_batched import train_one_epoch_batched
def rss():
    for l in open('/proc/self/status'):
        if l.startswith('VmRSS:'): return int(l.split()[1]) / 1024
peak = [0]; stop = [False]
def watch():
    while not stop[0]: peak[0] = max(peak[0], rss()); time.sleep(0.05)
ei = torch.load(cfg.EDGE_INDEX_PATH, weights_only=False)
torch.manual_seed(42); m = FedGKTv2(); opt = torch.optim.Adam(m.parameters(), lr=1e-3)
start = rss(); th = threading.Thread(target=watch, daemon=True); th.start(); t0 = time.time()
train_one_epoch_batched(m, opt, [21419], ei, 1, torch.device('cpu'))
stop[0] = True; th.join()
print(f"N={N:>5} answers: start {start:6.0f} MB  peak {peak[0]:6.0f} MB  extra {peak[0]-start:6.0f} MB  ({time.time()-t0:.0f} s)", flush=True)