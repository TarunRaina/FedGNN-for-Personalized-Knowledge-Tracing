"""
make_report_figures.py -- Phase 1 report figures for FedGKT (v2).

Run from the fedgkt_v2 folder:
    pip install matplotlib pandas
    python make_report_figures.py --history <path to history.json> --priors <path to learned_priors.csv>

Writes PNGs (300 dpi) into report_figures/:
  1_architecture.png      model diagram (paper style)
  2_results_table.png     comparison table (macro + pooled AUC, test set)
  3_auc_comparison.png    bar chart, macro vs pooled AUC
  4_training_curve.png    validation macro AUC + training loss per epoch
  5_learned_priors.png    learned concept difficulty vs real accuracy
"""
import argparse, json, os
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Circle, Rectangle, FancyArrowPatch
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument('--history', default='history.json')
ap.add_argument('--priors', default='learned_priors.csv')
ap.add_argument('--out', default='report_figures')
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
plt.rcParams.update({'font.size': 11, 'font.family': 'DejaVu Sans'})
OURS = '#C0392B'; OTHER = '#7F8C8D'; OTHER2 = '#BDC3C7'
SHARED = '#1E8449'; PERSONAL = '#922B21'; INK = '#2C3E50'

# Test-set results: 100 students, 27,871 predictions (each student's first answer excluded,
# the same rule pyKT uses for the baselines). FedGKT counting all answers: macro 0.7281.
RESULTS = [  # model, macro AUC, pooled AUC
    ('DKVMN', 0.7575, 0.7897), ('DKT', 0.7519, 0.7886), ('AKT', 0.7400, 0.7977),
    ('GKT', 0.7395, 0.7434), ('SAKT', 0.7278, 0.7560), ('FedGKT (ours)', 0.7270, 0.7854)]


def save(fig, name):
    fig.savefig(os.path.join(a.out, name), dpi=300, bbox_inches='tight')
    plt.close(fig)
    print('saved', name)


# 1 ── architecture (clean, paper style) ─────────────────────────────────
with plt.rc_context({'font.family': 'STIXGeneral', 'mathtext.fontset': 'stix', 'font.size': 10,
                     'hatch.color': '#B0B0B0', 'hatch.linewidth': 0.6}):
    fig, ax = plt.subplots(figsize=(13.6, 5.6))
    ax.set_xlim(0, 13.6); ax.set_ylim(0, 5.6); ax.set_aspect('equal'); ax.axis('off')
    K = '#222222'; G = '#8A8A8A'; ACC = '#B03A2E'; BLUE = '#2E5C8A'

    def line_arrow(x1, y1, x2, y2, color=K, lw=1.1, ls='-'):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle='-|>', mutation_scale=10,
                                     color=color, lw=lw, ls=ls, shrinkA=0, shrinkB=0))

    def vector(x, y0, n, h, label, label_side='left'):
        for k in range(n):
            ax.add_patch(Rectangle((x, y0 + k * h), 0.2, h, fc='white', ec=K, lw=0.8))
        ytop = y0 + n * h
        if label_side == 'left':
            ax.text(x - 0.08, y0 + n * h / 2, label, ha='right', va='center', fontsize=11)
        return ytop

    # ── knowledge graph panel ──
    ax.add_patch(Rectangle((0.3, 0.9), 3.8, 3.6, fc='none', ec=K, lw=0.9))
    ax.text(0.42, 4.3, r'Student knowledge graph $\mathcal{G}_s$', fontsize=10.5, va='center')
    R = 0.27
    MAST, PART, NONE = '#5A5A5A', '#B5B5B5', 'white'
    nodes = {1: (0.95, 3.5, MAST), 2: (2.2, 3.62, MAST), 3: (3.45, 3.4, PART),
             4: (1.3, 2.4, PART), 5: (2.75, 2.4, NONE), 6: (2.0, 1.4, NONE)}
    for s, d in [(1, 2), (1, 4), (2, 3), (2, 5), (4, 6), (5, 6), (3, 5)]:
        (x1, y1, _), (x2, y2, _) = nodes[s], nodes[d]
        dx, dy = x2 - x1, y2 - y1; L = (dx * dx + dy * dy) ** 0.5
        line_arrow(x1 + dx / L * R, y1 + dy / L * R, x2 - dx / L * (R + 0.02), y2 - dy / L * (R + 0.02),
                   color=G, lw=1.0)
    for i, (x, y, c) in nodes.items():
        tgt = i == 6
        ax.add_patch(Circle((x, y), R, fc=c, ec=ACC if tgt else K, lw=2.0 if tgt else 0.9,
                            hatch='////' if c == NONE else None, zorder=3))
        lab = r'$c_{q_t}$' if tgt else rf'$c_{i}$'
        ax.text(x, y, lab, ha='center', va='center', zorder=4, fontsize=11,
                color='white' if c == MAST else K,
                bbox=dict(boxstyle='circle,pad=0.12', fc='white', ec='none') if c == NONE else None)
    # feature vector of a node
    vector(3.55, 1.05, 7, 0.12, '')
    ax.text(3.65, 2.0, r'$x_i\in\mathbb{R}^7$', ha='center', va='bottom', fontsize=10)

    # ── learned prior (shared) ──
    for k in range(14):
        ax.add_patch(Rectangle((0.8 + k * 0.2, 0.32), 0.2, 0.2, fc='#E6EEF6', ec=BLUE, lw=0.7))
    ax.text(2.2, 0.12, r'learned concept prior $\pi\in\mathbb{R}^{835}$ (fills unattempted concepts)',
            ha='center', va='center', fontsize=9.5, color=BLUE)
    line_arrow(2.2, 0.52, 2.2, 0.9, color=BLUE)

    # ── GAT (shared) ──
    for off in (0.36, 0.18, 0.0):
        ax.add_patch(Rectangle((4.9 + off, 1.6 + off), 1.7, 1.6, fc='#F2F2F2' if off else 'white',
                               ec=K, lw=0.9))
    ax.text(5.75, 2.4, 'GAT\n3 layers', ha='center', va='center', fontsize=11)
    ax.text(5.93, 1.38, r'heads 4, 4, 1;  dims $128\rightarrow128\rightarrow64$', ha='center', fontsize=9)
    ax.text(5.93, 3.95, r'$h_i^{(l+1)}=\sigma\!\left(\sum_{j\in\mathcal{N}(i)\cup\{i\}}\alpha_{ij}\,W^{(l)}h_j^{(l)}\right)$',
            ha='center', va='center', fontsize=10)
    line_arrow(4.1, 2.4, 4.9, 2.4)
    ax.add_patch(Rectangle((4.65, 1.12), 2.6, 3.15, fc='none', ec=BLUE, lw=1.0, ls=(0, (4, 3))))
    ax.text(7.2, 4.37, 'shared', ha='right', fontsize=9.5, color=BLUE, style='italic')

    # ── read-out + skip ──
    vector(7.75, 2.1, 8, 0.17, '')
    ax.text(7.85, 3.5, r'$h_{q_t}$', ha='center', va='bottom', fontsize=11)
    vector(7.75, 0.75, 7, 0.12, '')
    ax.text(7.85, 1.62, r'$x_{q_t}$', ha='center', va='bottom', fontsize=11)
    line_arrow(7.25, 2.78, 7.75, 2.78)
    ax.add_patch(Circle((8.65, 2.1), 0.17, fc='white', ec=K, lw=0.9))
    ax.text(8.65, 2.1, '+', ha='center', va='center', fontsize=12)
    line_arrow(7.95, 2.78, 8.52, 2.2)
    line_arrow(7.95, 1.17, 8.52, 2.0)

    # ── personal head ──
    ax.add_patch(Rectangle((9.15, 1.7), 1.25, 0.8, fc='white', ec=K, lw=0.9))
    ax.text(9.775, 2.1, 'MLP\n' + r'$64\rightarrow32\rightarrow1$', ha='center', va='center', fontsize=10)
    line_arrow(8.82, 2.1, 9.15, 2.1)
    ax.add_patch(Rectangle((8.35, 1.5), 2.25, 1.25, fc='none', ec=ACC, lw=1.0, ls=(0, (4, 3))))
    ax.text(10.55, 2.85, 'personal', ha='right', fontsize=9.5, color=ACC, style='italic')

    # ── output ──
    ax.add_patch(Circle((11.3, 2.1), 0.33, fc='white', ec=K, lw=0.9))
    ax.text(11.3, 2.1, r'$\hat{p}_t$', ha='center', va='center', fontsize=12)
    line_arrow(10.4, 2.1, 10.97, 2.1)
    ax.text(11.75, 2.1, r'$=P(r_t=1\mid\mathcal{G}_s)$', va='center', fontsize=10.5)

    # ── predict-then-update loop ──
    ax.plot([11.3, 11.3, 2.2], [2.43, 4.85, 4.85], color=ACC, lw=1.0, ls=(0, (4, 3)))
    line_arrow(2.2, 4.85, 2.2, 4.5, color=ACC, ls=(0, (4, 3)))
    ax.text(6.75, 5.02, r'observe $r_t$, then update $x_{q_t}$ (prediction is made before the update)',
            ha='center', va='bottom', fontsize=9.5, color=ACC)

    # ── node legend ──
    lx, ly = 8.4, 0.55
    for k, (lab, c) in enumerate([('mastered', MAST), ('partial', PART), ('not attempted', NONE)]):
        ax.add_patch(Circle((lx + k * 1.6, ly), 0.1, fc=c, ec=K, lw=0.8, hatch='////' if c == NONE else None))
        ax.text(lx + k * 1.6 + 0.17, ly, lab, va='center', fontsize=9.5)
    ax.add_patch(Circle((lx, ly - 0.38), 0.1, fc='white', ec=ACC, lw=1.8))
    ax.text(lx + 0.17, ly - 0.38, r'queried concept $c_{q_t}$', va='center', fontsize=9.5)
    save(fig, '1_architecture.png')

# 2 ── results table ─────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7.5, 3.3)); ax.axis('off')
rows = [[m, f'{mac:.4f}', f'{poo:.4f}'] for m, mac, poo in RESULTS]
t = ax.table(cellText=rows, colLabels=['Model', 'Macro AUC\n(per student)', 'Pooled AUC'],
             loc='center', cellLoc='center')
t.auto_set_font_size(False); t.set_fontsize(10.5); t.scale(1, 1.6)
for j in range(3):
    t[0, j].set_facecolor('#2C3E50'); t[0, j].set_text_props(color='white', weight='bold')
    t[0, j].set_height(t[0, j].get_height() * 1.5)
    t[len(rows), j].set_facecolor('#FADBD8'); t[len(rows), j].set_text_props(weight='bold')
ax.set_title('Test-set results (100 students, 27,871 predictions)', pad=6)
ax.text(0.5, -0.04, "Each student's first answer is excluded for every model (pyKT's rule).\n"
        "Counting all answers, FedGKT's macro AUC is 0.7281.",
        ha='center', va='top', fontsize=8.5, style='italic', transform=ax.transAxes)
save(fig, '2_results_table.png')

# 3 ── AUC bar chart ─────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 4.5))
names = [r[0] for r in RESULTS]; x = range(len(names)); w = 0.38
c1 = [OURS if 'ours' in n else OTHER for n in names]; c2 = ['#E6B0AA' if 'ours' in n else OTHER2 for n in names]
b1 = ax.bar([i - w / 2 for i in x], [r[1] for r in RESULTS], w, color=c1, label='Macro AUC (per student)')
b2 = ax.bar([i + w / 2 for i in x], [r[2] for r in RESULTS], w, color=c2, label='Pooled AUC')
for bars in (b1, b2):
    for b in bars:
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.003, f'{b.get_height():.3f}',
                ha='center', fontsize=8.5)
ax.set_xticks(list(x)); ax.set_xticklabels(names); ax.set_ylim(0.70, 0.81); ax.set_ylabel('AUC')
ax.set_title('Model comparison on the test set'); ax.legend(loc='upper right', frameon=False)
ax.spines[['top', 'right']].set_visible(False)
save(fig, '3_auc_comparison.png')

# 4 ── training curve ────────────────────────────────────────────────────
if os.path.exists(a.history):
    h = json.load(open(a.history))
    ep = [e['epoch'] for e in h]; loss = [e['train_loss'] for e in h]
    vep = [e['epoch'] for e in h if e['val_macro_auc'] is not None]
    vauc = [e['val_macro_auc'] for e in h if e['val_macro_auc'] is not None]
    best = max(range(len(vauc)), key=lambda i: vauc[i])
    fig, ax1 = plt.subplots(figsize=(9, 4.5))
    ax1.plot(vep, vauc, color=OURS, lw=2, label='Validation macro AUC')
    ax1.scatter([vep[best]], [vauc[best]], color=OURS, zorder=5)
    ax1.annotate(f'best: epoch {vep[best]}, {vauc[best]:.4f}', (vep[best], vauc[best]),
                 xytext=(-150, -35), textcoords='offset points', arrowprops=dict(arrowstyle='->'))
    ax1.set_xlabel('Epoch'); ax1.set_ylabel('Validation macro AUC', color=OURS)
    ax2 = ax1.twinx(); ax2.plot(ep, loss, color='#2E86C1', lw=1.5, ls='--', label='Training loss')
    ax2.set_ylabel('Training loss (BCE)', color='#2E86C1')
    ax1.set_title('FedGKT training (early stopping, patience 5)')
    l1, n1 = ax1.get_legend_handles_labels(); l2, n2 = ax2.get_legend_handles_labels()
    ax1.legend(l1 + l2, n1 + n2, loc='center right', frameon=False)
    save(fig, '4_training_curve.png')
else:
    print('SKIPPED training curve: no history.json at', os.path.abspath(a.history))

# 5 ── learned priors vs real accuracy ──────────────────────────────────
if os.path.exists(a.priors):
    p = pd.read_csv(a.priors); p = p[p['train_attempts'] >= 30]
    rho = p['learned_prior'].corr(p['empirical_accuracy'], method='spearman')
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(p['empirical_accuracy'], p['learned_prior'], s=12, alpha=0.6, color=OURS)
    ax.set_xlabel('Real accuracy of students on the concept'); ax.set_ylabel('Learned prior (model)')
    ax.set_title(f'Learned concept difficulty vs reality\n{len(p)} concepts with 30+ attempts, Spearman = {rho:.3f}')
    ax.spines[['top', 'right']].set_visible(False)
    save(fig, '5_learned_priors.png')
else:
    print('SKIPPED priors plot: no learned_priors.csv at', os.path.abspath(a.priors))
