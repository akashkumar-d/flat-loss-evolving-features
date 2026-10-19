#!/usr/bin/env python3
import json, sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT = Path(sys.argv[1]); OUT.mkdir(parents=True, exist_ok=True)
ORANGE, PURPLE, LBLUE, GRAY, GRID = '#D55E00', '#7B3294', '#56B4E9', '#7A7A7A', '#E6E6E6'
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 8, 'axes.titlesize': 8.5, 'axes.labelsize': 8,
                     'xtick.labelsize': 7.5, 'ytick.labelsize': 7.5, 'legend.fontsize': 7.5, 'axes.spines.top': False,
                     'axes.spines.right': False, 'axes.edgecolor': '#555555', 'axes.linewidth': 0.7})
CASES = [('lag_Y_G2_s11.json', 'One block, p = 113', 'Figure 5 setting'),
         ('lag_Z_B_s11.json', 'Two blocks, p = 113', 'Figure 15 setting'),
         ('lag_W_P1_s8.json', 'Plain head (no RMS step)', 'training CE near 1e-7'),
         ('lag_V_E2_s1.json', 'Two blocks, head decay 0.5', 'no gain')]


def cross(ts, vs, q):
    for i, v in enumerate(vs):
        if v is not None and v >= q:
            if i == 0: return float(ts[0])
            t0, v0 = ts[i - 1], vs[i - 1]
            return float(t0 + (q - v0) / (v - v0) * (ts[i] - t0)) if v > v0 else float(ts[i])
    return None


cases = [c for c in CASES if Path(c[0]).exists()]
n = len(cases)
fig = plt.figure(figsize=(2.6 * n, 5.4), layout='constrained')
gs = fig.add_gridspec(2, n, height_ratios=[1.2, 1.0])
stats = {}
bars = []
for j, (fn, t1, t2) in enumerate(cases):
    d = json.load(open(fn)); rows = d['rows']; ts = np.array([r['step'] for r in rows])
    ob = d['obs']; k = 'test_acc_nm' if any('test_acc_nm' in o for o in ob) else 'test_acc'
    to = np.array([o['step'] for o in ob if k in o]); vo = np.array([o[k] for o in ob if k in o])
    n50 = cross(to, vo, 0.5)
    ax = fig.add_subplot(gs[0, j])
    ax.plot(to / 1e3, vo, color=ORANGE, lw=1.8, label='Network', zorder=5)
    leads = {}
    for key, col, lab in (('ridge_t0.001', PURPLE, "Paper's refit"), ('filt8', LBLUE, 'Fourier refit (top 8)'), ('probe', GRAY, 'Logistic probe')):
        v = [r['resid'].get(key) for r in rows]
        ax.plot(ts / 1e3, v, color=col, lw=1.4, label=lab, zorder=4)
        c = cross(ts, v, 0.5); leads[key] = None if (c is None or n50 is None) else n50 - c
    ax.axhline(0.5, color='#999999', lw=0.6, ls=':')
    ax.set_xlim(max(0.0, (n50 - 4500) / 1e3), ts[-1] / 1e3); ax.set_ylim(-0.02, 1.02)
    ax.grid(True, axis='y', color=GRID, lw=0.6); ax.set_axisbelow(True)
    ax.set_title(f'{t1}\n({t2})', loc='left'); ax.set_xlabel('Updates (thousands)')
    if j == 0: ax.set_ylabel('Held-out accuracy (no-mirror pairs)'); ax.legend(loc='upper left', frameon=False, fontsize=7)
    lt = lambda x: 'never' if x is None else f'{x / 1e3:+.2f}k'
    ax.text(0.97, 0.04, f"lead at 0.5\nrefit {lt(leads['ridge_t0.001'])}\nFourier {lt(leads['filt8'])}\nprobe {lt(leads['probe'])}",
            transform=ax.transAxes, ha='right', va='bottom', fontsize=7, color='#333333')
    i = int(np.argmin(np.abs(ts - n50))); R = rows[i]['resid']; fq = R['freq']
    r2 = np.array(fq['r2_ev']); r2tr = np.array(fq['r2_tr']); E = np.array(fq['E']); rest = np.argsort(-E)[8:]
    sh = lambda v: v['rms_signal'] ** 2 / (v['rms_signal'] ** 2 + v['rms_rest'] ** 2)
    bars.append(dict(label=f"{t1}\n{int((r2 > 0.05).sum())} of {len(r2)} frequencies carried\nother {len(rest)}: held-out $R^2$ {r2[rest].mean():.2f}",
                     net=sh(R['dec']['net']), ref=sh(R['dec']['ridge']), net_acc=R['dec']['net']['acc_full'],
                     ref_acc=R['dec']['ridge']['acc_full'], net_top=R['dec']['net']['acc_signal'], ref_top=R['dec']['ridge']['acc_signal']))
    stats[fn] = dict(net50=n50, leads=leads, carried=int((r2 > 0.05).sum()), K=len(r2), r2_rest_tr=float(r2tr[rest].mean()),
                     r2_rest_ho=float(r2[rest].mean()), step=int(rows[i]['step']), bars=bars[-1])
ax = fig.add_subplot(gs[1, :])
x = np.arange(n); w = 0.32
for off, key, col, lab in ((-w / 2, 'net', ORANGE, 'Network'), (w / 2, 'ref', PURPLE, "Paper's refit")):
    vals = [b[key] for b in bars]
    ax.bar(x + off, vals, width=w * 0.92, color=col, alpha=0.85, label=lab, zorder=3)
    for xi, b in zip(x + off, bars):
        ax.text(xi, b[key] + 0.02, f"accuracy {b[key + '_acc']:.2f}\n(top-8 part {b[key + '_top']:.2f})", ha='center', va='bottom', fontsize=6.8)
ax.set_xticks(x); ax.set_xticklabels([b['label'] for b in bars], fontsize=7.2)
ax.set_ylim(0, 0.9); ax.set_ylabel('Share of held-out logit variance\non the top-8 frequencies')
ax.axhline(8 / 56, color='#999999', lw=0.6, ls=':'); ax.text(0.5, 8 / 56 + 0.012, 'one-hot target: 8/56', fontsize=6.8, color='#555555', ha='center')
ax.grid(True, axis='y', color=GRID, lw=0.6); ax.set_axisbelow(True)
ax.legend(loc='upper right', frameon=False)
ax.set_title("At the network's 0.5 crossing: how much of each readout's held-out logit variance sits on the top-8 frequencies", loc='left')
fig.savefig(OUT / 'refit_lag_cases.png', dpi=200); fig.savefig(OUT / 'refit_lag_cases.pdf'); plt.close(fig)
json.dump(stats, open(OUT / 'plot_cases_stats.json', 'w'), indent=1)
print(json.dumps({k: dict(leads=v['leads'], carried=v['carried'], r2_rest_ho=v['r2_rest_ho'], r2_rest_tr=v['r2_rest_tr']) for k, v in stats.items()}, indent=1))
