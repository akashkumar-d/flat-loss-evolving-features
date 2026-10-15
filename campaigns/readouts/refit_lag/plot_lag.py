#!/usr/bin/env python3
import json, sys, re
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT = Path(sys.argv[1]); OUT.mkdir(parents=True, exist_ok=True)
ORANGE, PURPLE, LBLUE, GRAY, INK, GRID = '#D55E00', '#7B3294', '#56B4E9', '#7A7A7A', '#222222', '#E6E6E6'
CAT = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4']
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 8, 'axes.titlesize': 8.5, 'axes.labelsize': 8,
                     'xtick.labelsize': 7.5, 'ytick.labelsize': 7.5, 'legend.fontsize': 7.5, 'axes.spines.top': False,
                     'axes.spines.right': False, 'axes.edgecolor': '#555555', 'axes.linewidth': 0.7})


def cross(ts, vs, q):
    for i, v in enumerate(vs):
        if v is not None and v >= q:
            if i == 0: return float(ts[0])
            t0, v0 = ts[i - 1], vs[i - 1]
            return float(t0 + (q - v0) / (v - v0) * (ts[i] - t0)) if v > v0 else float(ts[i])
    return None


d = json.load(open('lag_Y_G2_s11.json')); rows = d['rows']; ts = np.array([r['step'] for r in rows])
obs = d['obs']; to = np.array([o['step'] for o in obs]); vo = np.array([o['test_acc_nm'] for o in obs])
net50 = cross(to, vo, 0.5)
fig, axs = plt.subplots(1, 3, figsize=(10.2, 3.1), layout='constrained', gridspec_kw=dict(width_ratios=[1.25, 1.15, 1.0]))
ax = axs[0]
ax.plot(to / 1e3, vo, color=ORANGE, lw=1.8, label='Network', zorder=5)
series = [('ridge_t0.001', PURPLE, '-', "Paper's refit ($\\tau=10^{-3}n$)"), ('ridge_t10', PURPLE, (0, (3, 2)), 'Same refit, $\\tau=10\\,n$'),
          ('filt8', LBLUE, '-', 'Fourier refit (top 8)'), ('probe', GRAY, '-', 'Logistic probe')]
for key, col, ls, lab in series:
    v = [r['resid'][key] for r in rows]
    ax.plot(ts / 1e3, v, color=col, ls=ls, lw=1.4, label=lab, zorder=4)
ax.axhline(0.5, color='#999999', lw=0.6, ls=':', zorder=1)
ax.set_xlim(1, 8); ax.set_ylim(-0.02, 1.02); ax.set_xlabel('Updates (thousands)'); ax.set_ylabel('Held-out accuracy (no-mirror pairs)')
ax.grid(True, axis='y', color=GRID, lw=0.6); ax.set_axisbelow(True)
ax.set_title('(a) Readouts of the same features\n(Figure 5 setting, seed 11)', loc='left')
ax.legend(loc='upper left', frameon=False, handlelength=2.2)

ax = axs[1]
i = int(np.argmin(np.abs(ts - net50))); R = rows[i]['resid']; fq = R['freq']
E = np.array(fq['E']); r2 = np.array(fq['r2_ev']); order = np.argsort(-E)
x = np.arange(1, len(order) + 1); vals = r2[order]
cols = [LBLUE if j < 8 else '#BDBDBD' for j in range(len(order))]
ax.bar(x, vals, width=0.8, color=cols, edgecolor='none', zorder=3)
ax.axhline(0, color='#555555', lw=0.7, zorder=4)
ax.set_xlim(0, len(order) + 1); ax.set_ylim(-0.7, 0.8)
ax.set_xlabel('Frequency, ranked by training energy'); ax.set_ylabel('Held-out $R^2$ of the refit\'s clock prediction')
ax.grid(True, axis='y', color=GRID, lw=0.6); ax.set_axisbelow(True)
ax.set_title(f'(b) Held-out $R^2$ per frequency at the\nnetwork\'s 0.5 crossing (update {rows[i]["step"]})', loc='left')
ax.text(4.5, 0.70, 'top 8', color='#1f6fa8', ha='center', fontsize=7.5)
ax.text(32, 0.30, f'other {len(order) - 8}: mean $R^2$ {np.mean(vals[8:]):.2f}\n(fitted on training rows,\nnoise on held-out rows)',
        color='#555555', ha='center', fontsize=7.5)

ax = axs[2]
dec = R['dec']; labs = [('net', 'Network'), ('probe', 'Logistic probe'), ('centroid', 'Refit, $\\tau\\to\\infty$'), ('ridge', "Paper's refit")]
yy = np.arange(len(labs))[::-1]
for (k, lab), y in zip(labs, yy):
    v = dec[k]; s = v['rms_signal'] ** 2 / (v['rms_signal'] ** 2 + v['rms_rest'] ** 2)
    ax.barh(y, s, color=LBLUE, height=0.55, zorder=3); ax.barh(y, 1 - s, left=s, color='#BDBDBD', height=0.55, zorder=3)
    ax.text(1.03, y, f"{v['acc_signal']:.2f}   {v['acc_full']:.2f}", va='center', fontsize=7.5)
ax.text(1.03, len(labs) - 0.42, 'top 8   all', fontsize=7, color='#555555')
ax.text(1.03, len(labs) - 0.12, 'accuracy of', fontsize=7, color='#555555')
ax.set_ylim(-0.5, len(labs) + 0.1)
ax.set_yticks(yy); ax.set_yticklabels([l for _, l in labs]); ax.set_xlim(0, 1.0)
ax.set_xlabel('Share of held-out logit variance'); ax.axvline(0.5, color='#999999', lw=0.6, ls=':')
ax.set_title('(c) Held-out logit variance at the same update:\ntop 8 frequencies (blue), other 48 (gray)', loc='left')
fig.savefig(OUT / 'refit_lag_mechanism.png', dpi=200); fig.savefig(OUT / 'refit_lag_mechanism.pdf'); plt.close(fig)

S = json.load(open('scan_runs.json'))
cells = [('One block, p = 61 (mirror-free)', lambda r: r['id'].startswith('ZM_f0.55') or r['id'].startswith('Q_M_p61'), 30),
         ('One block, p = 89', lambda r: r['id'].startswith('ZP_P_p89_f0.25') or r['id'].startswith('Q_P_p89'), 44),
         ('One block, p = 113 (Fig. 5)', lambda r: r['id'].startswith('Y_G2_'), 56),
         ('One block, p = 149', lambda r: r['id'].startswith('ZP_P_p149_f0.16') or r['id'].startswith('Q_P_p149'), 74),
         ('Two blocks, p = 113 (Fig. 15)', lambda r: re.match(r'(Z|Q)_B_f0\.2_lam_v0\.275_tau10k_s\d+$', r['id']) is not None, 56)]
KS = [('fref_top4', '4'), ('fref_top8', '8'), ('fref_top16', '16'), ('refit', 'all')]
fig, axs = plt.subplots(1, 2, figsize=(9.6, 3.2), layout='constrained')
ax = axs[0]; xs = np.arange(len(KS))
stats = {}
for j, (lab, sel, kf) in enumerate(cells):
    rs = [r for r in S if sel(r)]; med, lo, hi = [], [], []
    for key, _ in KS:
        L = np.array([r['readouts'][key]['lead']['0.5'] for r in rs if key in r['readouts'] and r['readouts'][key]['lead']['0.5'] is not None])
        med.append(np.median(L) / 1e3); lo.append(np.percentile(L, 25) / 1e3); hi.append(np.percentile(L, 75) / 1e3)
    off = (j - 2) * 0.06
    ax.errorbar(xs + off, med, yerr=[np.array(med) - lo, np.array(hi) - med], color=CAT[j], lw=1.6, marker='o', ms=4,
                capsize=0, elinewidth=0.9, label=f'{lab} ({len(rs)} runs)')
    stats[lab] = dict(median_k=med)
ax.axhline(0, color='#555555', lw=0.7)
ax.set_xticks(xs); ax.set_xticklabels(['top 4', 'top 8', 'top 16', 'all (paper refit)'])
ax.set_xlabel('Frequencies kept in the refit (ranked by training energy)')
ax.set_ylabel('Lead over the network at held-out 0.5\n(thousands of updates; > 0: readout first)')
ax.grid(True, axis='y', color=GRID, lw=0.6); ax.set_axisbelow(True)
ax.set_title('(a) Adding frequencies the features do not carry delays the refit', loc='left')
ax.legend(loc='lower left', frameon=False, fontsize=7)

ax = axs[1]
A = {}
for fn, lab, col in (('lag_Y_G2_s11.json', 'Transformer, one block (Fig. 5 setting)', CAT[0]),
                     ('lag_Z_B_s11.json', 'Transformer, two blocks (Fig. 15 setting)', CAT[2]),
                     ('lag_R5_s1.json', 'Transformer, one block, no gain (R5)', CAT[3]),
                     ('lag_F_H_p31_s10.json', 'One-hidden-layer MLP (p = 31)', CAT[1])):
    dd = json.load(open(fn)); rr = dd['rows']; tt = [r['step'] for r in rr]; tab = 'resid' if dd['kind'] == 'tf' else 'last'
    ob = dd['obs']; k = 'test_acc_nm' if any('test_acc_nm' in o for o in ob) else 'test_acc'
    n50 = cross([o['step'] for o in ob if k in o], [o[k] for o in ob if k in o], 0.5)
    taus = dd['taus']; leads = []
    for x in taus:
        c = cross(tt, [r[tab][f'ridge_t{x:g}'] for r in rr], 0.5); leads.append(None if c is None else (n50 - c) / 1e3)
    A[lab] = leads
    xv = np.log10(taus); yv = [np.nan if v is None else v for v in leads]
    ax.plot(xv, yv, color=col, lw=1.6, marker='o', ms=4, label=lab)
    nv = [xx for xx, v in zip(xv, leads) if v is None]
    if nv: ax.scatter(nv, [-3.6] * len(nv), marker='v', color=col, s=22, zorder=5)
ax.axhline(0, color='#555555', lw=0.7); ax.axvline(-3, color='#999999', lw=0.6, ls=':')
ax.text(-2.95, 3.0, "paper's $\\tau$", fontsize=7, color='#555555')
ax.text(1.2, -3.35, 'never reaches 0.5', fontsize=7, color='#555555', ha='center')
ax.set_xticks(np.log10(taus)); ax.set_xticklabels([f'$10^{{{int(np.log10(x))}}}$' for x in taus])
ax.set_ylim(-3.9, 3.4); ax.set_xlabel('Ridge penalty $\\tau / n$ (all frequencies kept)')
ax.set_ylabel('Lead at held-out 0.5 (thousands)')
ax.grid(True, axis='y', color=GRID, lw=0.6); ax.set_axisbelow(True)
ax.set_title('(b) A larger penalty helps the transformer, breaks the MLP', loc='left')
ax.legend(loc='lower left', frameon=False, fontsize=7)
fig.savefig(OUT / 'refit_lag_dose.png', dpi=200); fig.savefig(OUT / 'refit_lag_dose.pdf'); plt.close(fig)
json.dump(dict(dose=stats, penalty=A), open(OUT / 'plot_lag_stats.json', 'w'), indent=1)
print('ok')
