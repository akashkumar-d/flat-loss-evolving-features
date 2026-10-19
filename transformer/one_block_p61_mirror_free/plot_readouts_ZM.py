#!/usr/bin/env python3
import json, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import plot_ZM as Z
C, PX, AX = Z.C, Z.PX, Z.AX
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator

C_K16 = '#E69F00'
CURVES = (('net', 'Network (held-out)', 'test_acc_nm', C.C_NET, '-', 1.7), ('refit', 'Refit (all frequencies)', 'refit_test_acc_nm', C.C_REFIT, '-', 1.4),
          ('k8', 'Fourier refit, top 8', 'fref_top8_test_acc_nm', C.C_FOUR, '-', 1.4), ('k16', 'Fourier refit, top 16', 'fref_top16_test_acc_nm', C_K16, C.DASH, 1.4))
LEADS = (('refit', 'paper refit (all 30 or 56 freq.)', 'o'), ('k8', 'Fourier refit top 8', 's'), ('k16', 'Fourier refit top 16', 'D'))
CELLS = ('p61 mirror-free f0.55 (ZM)', 'p61 random f0.33 (ZP)', 'p89 random f0.25 (ZP)', 'p113 random f0.2 (Y G2)', 'p149 random f0.16 (ZP)')
XLAB = ('61\nmirror-free', '61\nrandom', '89', '113', '149')


def main(ana, rjson, outdir):
    A = json.load(open(ana))['cells']; RO = json.load(open(rjson))
    cols = [('ZP_random_f0.33', 'p = 61, random split, fraction 0.33'), ('ZM_f0.55', 'p = 61, mirror-free split, fraction 0.55')]
    data = [(A[k], Z.load(A[k]), t) for k, t in cols]
    T = min(min(len(d['loss']) for d in runs) for _, runs, _ in data) - 1; g = np.arange(0, T + 1, 20)
    fig = plt.figure(figsize=(6.75, 6.0), layout='constrained')
    gs = fig.add_gridspec(3, 2, height_ratios=[1.45, 0.8, 1.25])
    for j, (cell, runs, ttl) in enumerate(data):
        ax = fig.add_subplot(gs[0, j]); axr = fig.add_subplot(gs[1, j], sharex=ax)
        W = [r['rec']['all']['window']['W'] for r in runs if r['rec']['all'].get('window', {}).get('W')]
        a, b = np.median([w[0] for w in W]), np.median([w[1] for w in W])
        for x_ in (ax, axr): x_.axvspan(a, min(b, T), color=C.C_BAND, lw=0, zorder=0)
        for i, (key, lab, obs_key, col, ls, lw) in enumerate(CURVES):
            Y = PX.on_grid(runs, obs_key, g); Z.shadowed(ax, g, np.nanmedian(Y, 0), col, ls=ls, lw=lw, z=6 - i)
            X = []
            for d in runs:
                s, v = PX.series(d['obs'], obs_key); X.append([AX.interp_cross(s, v, q) for q in (0.1, 0.5, 0.9)])
            X = np.array(X, float); m = np.nanmedian(X, 0)
            axr.plot([m[0], m[2]], [i, i], color=col, lw=5.5, alpha=0.35, solid_capstyle='butt', zorder=2)
            for t in X[:, 1]: axr.plot([t, t], [i - 0.28, i + 0.28], color=col, lw=0.9, alpha=0.85, zorder=3)
            axr.plot([m[1]], [i], 'o', ms=5, color=col, mec='white', mew=0.8, zorder=4)
            axr.text(max(m[2], np.nanmax(X[:, 1])) + 0.015 * T, i, ('network', 'refit', 'top 8', 'top 16')[i], va='center', fontsize=7.5, color='#333333')
        ax.set_ylim(-0.02, 1.04); ax.set_xlim(0, T); ax.grid(True, axis='y', color=C.C_GRID, lw=0.6)
        ax.set_title(ttl); ax.tick_params(labelbottom=False)
        axr.set_ylim(len(CURVES) - 0.4, -0.6); axr.set_yticks([]); axr.grid(True, axis='x', color=C.C_GRID, lw=0.6)
        axr.xaxis.set_major_formatter(C.KFMT); axr.xaxis.set_major_locator(MaxNLocator(4, integer=True)); axr.set_xlabel('Updates (thousands)')
        if j == 0: ax.set_ylabel('Held-out accuracy\n(no-mirror pairs)'); axr.set_ylabel('Rise: 0.1 to 0.9,\ndot at 0.5')
        else: ax.tick_params(labelleft=False)
    YL = (-3000, 2000)
    for j, (qi, qname) in enumerate(((1, 0.5), (2, 0.9))):
        ax = fig.add_subplot(gs[2, j])
        ax.axhline(0, color='#555555', lw=0.8, zorder=1)
        for k, (key, name, mk) in enumerate(LEADS):
            col = dict(refit=C.C_REFIT, k8=C.C_FOUR, k16=C_K16)[key]
            for ci, cname in enumerate(CELLS):
                L = [r['lead'][qi] for r in RO[cname][name]]
                assert all(x is not None for x in L), (cname, name, qi)
                xx = ci + (k - 1) * 0.22; lo, hi = min(L), max(L)
                ax.plot([xx, xx], [max(lo, YL[0] + 80), min(hi, YL[1] - 80)], color=col, lw=1.0, zorder=2)
                for v_, lim, m_, va in ((lo, YL[0] + 80, 'v', 'top'), (hi, YL[1] - 80, '^', 'bottom')):
                    if (m_ == 'v' and v_ < lim) or (m_ == '^' and v_ > lim):
                        ax.plot([xx], [lim], marker=m_, ms=3.5, color=col, ls='', zorder=3)
                        ax.text(xx + 0.06, lim, f'{v_ / 1000:+.1f}k', fontsize=6.5, color='#444444', ha='left', va='center')
                ax.plot([xx], [np.median(L)], marker=mk, ms=5.5, color=col, mec='white', mew=0.7, ls='', zorder=4)
        ax.set_xticks(range(len(CELLS))); ax.set_xticklabels(XLAB); ax.set_xlim(-0.5, len(CELLS) - 0.5)
        ax.set_ylim(*YL); ax.grid(True, axis='y', color=C.C_GRID, lw=0.6); ax.set_axisbelow(True)
        ax.set_xlabel('p (split)'); ax.set_title(f"Lead at the network's {qname} crossing")
        ax.yaxis.set_major_formatter(C.KFMT)
        if j == 0: ax.set_ylabel('Readout lead\n(thousands of updates)')
        ax.text(len(CELLS) - 0.55, 70, 'readout first', ha='right', va='bottom', fontsize=7.5, color='#555555')
        ax.text(len(CELLS) - 0.55, -70, 'network first', ha='right', va='top', fontsize=7.5, color='#555555')
    h = []
    for key, lab, obs_key, col, ls, lw in CURVES:
        ln = Line2D([], [], color=col, ls=ls, lw=lw); ln.set_path_effects(Z.SHADOW); h.append(ln)
    l = [c[1] for c in CURVES]
    h += [Line2D([], [], ls='', marker=mk, ms=5.5, color=dict(refit=C.C_REFIT, k8=C.C_FOUR, k16=C_K16)[key]) for key, _, mk in LEADS] + [Patch(color=C.C_BAND, lw=0)]
    l += ['lead: refit (all freq.)', 'lead: Fourier top 8', 'lead: Fourier top 16', 'Window (all pairs)']
    fig.legend(h, l, loc='outside lower center', ncol=4, frameon=False, handlelength=1.8, columnspacing=1.2, fontsize=8)
    C.save(fig, outdir, 'ZM_readouts')


if __name__ == '__main__':
    main(*sys.argv[1:4])
