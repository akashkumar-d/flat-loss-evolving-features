#!/usr/bin/env python3
import json, sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
OUT = Path(sys.argv[1]); OUT.mkdir(parents=True, exist_ok=True)
GRID = '#E6E6E6'
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 8, 'axes.titlesize': 8.5, 'axes.labelsize': 8,
                     'xtick.labelsize': 7.5, 'ytick.labelsize': 7.5, 'legend.fontsize': 7, 'axes.spines.top': False,
                     'axes.spines.right': False, 'axes.edgecolor': '#555555', 'axes.linewidth': 0.7})
RUNS = [('Y_G2_s11', 'Transformer, one block (Fig. 5 setting)', '#D55E00', '-'),
        ('Z_B_s11', 'Transformer, two blocks (Fig. 15 setting)', '#8C2D04', '-'),
        ('W_P1_s8', 'Transformer, plain head', '#E69F00', '--'),
        ('F_H_p31_s10', 'MLP, p = 31', '#9ECAE1', '-'),
        ('H_p61_s1', 'MLP, p = 61', '#4292C6', '-'),
        ('H_p101_s7', 'MLP, p = 101', '#08519C', '-')]
summ = {Path(r['file']).stem: r for r in json.load(open(HERE / 'kcurve' / 'summary.json'))}

fig = plt.figure(figsize=(7.4, 3.5), layout='constrained')
gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.0, 1.05])

ax = fig.add_subplot(gs[0, 0])
for rid, lab, col, ls in RUNS:
    fn = HERE / 'kcurve' / f'{rid}.json'
    if not fn.exists() or rid not in summ: continue
    d = json.load(open(fn)); K = len(d['rows'][0]['acc_K']); t0 = summ[rid]['net'][0.5] if 0.5 in summ[rid]['net'] else summ[rid]['net']['0.5']
    ts = np.array([r['step'] for r in d['rows']]); c = np.array([r['carried'] for r in d['rows']]) / K
    m = (ts - t0 >= -4500) & (ts - t0 <= 2500)
    ax.plot((ts[m] - t0) / 1e3, c[m], color=col, ls=ls, lw=1.5, label=lab)
ax.axvline(0, color='#999999', lw=0.6, ls=':')
ax.set_xlabel("Updates from the network's 0.5 crossing (k)")
ax.set_ylabel('Share of the (p - 1)/2 frequencies')
ax.set_ylim(-0.02, 1.04); ax.grid(True, axis='y', color=GRID, lw=0.6); ax.set_axisbelow(True)
ax.set_title('(a) Frequencies with held-out signal\n(held-out $R^2$ > 0.05), replays', loc='left')

ax = fig.add_subplot(gs[0, 1])
for rid, lab, col, ls in RUNS:
    if rid not in summ: continue
    L = summ[rid]['lead0.5']; K = np.arange(1, len(L) + 1)
    v = np.array([np.nan if x is None else x / 1e3 for x in L])
    ax.plot(K, v, color=col, ls=ls, lw=1.5)
    if not np.isnan(v[-1]): ax.plot(K[-1], v[-1], 'o', color=col, ms=4.5, mec='white', mew=0.7, zorder=5)
ax.set_xscale('log', base=2); ax.set_xticks([1, 2, 4, 8, 16, 32, 64]); ax.set_xticklabels(['1', '2', '4', '8', '16', '32', '64'])
ax.axhline(0, color='#999999', lw=0.6); ax.axvline(8, color='#999999', lw=0.6, ls=':')
ax.set_xlabel('K (top by training energy)')
ax.set_ylabel('Lead over the network (k updates)')
ax.grid(True, axis='y', color=GRID, lw=0.6); ax.set_axisbelow(True)
ax.set_title("(b) Lead of the top-K refit at held-out\n0.5 (dot: K = all, the paper's refit)", loc='left')

ax = fig.add_subplot(gs[0, 2])
kr = json.load(open(HERE / 'k_rules.json'))
groups = [((61, 1), '61'), ((89, 1), '89'), ((113, 1), '113'), ((113, 2), '113\n2 blocks'), ((149, 1), '149')]
qs = [('0.1', '#FDD0A2', 'at 0.1'), ('0.5', '#F16913', 'at 0.5'), ('0.9', '#8C2D04', 'at 0.9')]
stats_c = {}
for i, (key, lab) in enumerate(groups):
    rs = [r for r in kr if (r['p'], r['layers']) == key]
    for j, (q, col, ql) in enumerate(qs):
        v = [r['at'][q]['carried'] for r in rs if q in r['at']]
        if not v: continue
        md, lo, hi = np.median(v), np.percentile(v, 25), np.percentile(v, 75)
        x = i + (j - 1) * 0.22
        ax.errorbar(x, md, yerr=[[md - lo], [hi - md]], fmt='o', color=col, ms=4.5, mec='#555555', mew=0.4, elinewidth=1.0,
                    capsize=1.5, label=ql if i == 0 else None, zorder=4)
        stats_c[f'{lab} {q}'] = dict(n=len(v), median=float(md), q25=float(lo), q75=float(hi))
    ax.text(i, -3.2, f'n = {len(rs)}', ha='center', fontsize=6.5, color='#555555')
ax.axhline(8, color='#999999', lw=0.6, ls=':')
ax.set_xticks(range(len(groups))); ax.set_xticklabels([g[1] for g in groups], fontsize=7.5); ax.set_xlabel('p')
ax.set_ylim(-5, 28); ax.set_ylabel('Frequencies with held-out signal')
ax.grid(True, axis='y', color=GRID, lw=0.6); ax.set_axisbelow(True)
ax.legend(loc='upper center', ncol=3, frameon=False, handletextpad=0.1, columnspacing=0.8)
ax.set_title(f"(c) All {len(kr)} transformer runs:\ncount at the network's crossings", loc='left')

h, l = fig.axes[0].get_legend_handles_labels()
fig.legend(h, l, loc='outside lower center', ncol=3, frameon=False, fontsize=7)
fig.savefig(OUT / 'refit_k_choice.png', dpi=200); fig.savefig(OUT / 'refit_k_choice.pdf'); plt.close(fig)
json.dump(stats_c, open(OUT / 'plot_k_stats.json', 'w'), indent=1)
print(json.dumps(stats_c, indent=1))
