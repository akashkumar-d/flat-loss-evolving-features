#!/usr/bin/env python3
import json, math, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'campaign_ZP')); sys.path.insert(0, os.path.join(HERE, '..', 'campaign_Z'))
import paper_ZP as PP
import plot_Z as PZ
import plot_X as PX
import analyze_X as AX
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator

C = PP
SHADOW = [pe.SimpleLineShadow(offset=(0.9, -0.9), shadow_color='#000000', alpha=0.28), pe.Normal()]
TRAJ = (('mirror', 'Mirror pairs', C.C_NET, C.DOT, 1.7), ('four', 'Fourier refit (top 8)', C.C_FOUR, '-', 1.4),
        ('all', 'Held-out (all pairs)', C.C_NET, '-', 1.7), ('nm', 'Held-out (no-mirror)', C.C_NET, C.DASH, 1.2),
        ('refit', 'Refit', C.C_REFIT, '-', 1.4))
SHORT = dict(mirror='mirror pairs', four='Fourier refit', all='all pairs', nm='no-mirror', refit='refit')


def load(cell):
    return [dict(PX.load(r['file']), rec=r) for r in sorted(cell['runs'], key=lambda r: r['id'])]


def series(d, key):
    r = d['rec']
    if key == 'mirror':
        if r['n_mirror'] == 0: return None
        s, va = PX.series(d['obs'], 'test_acc'); _, vn = PX.series(d['obs'], 'test_acc_nm')
        return s, (r['n_heldout'] * va - r['n_nm'] * vn) / r['n_mirror']
    return PX.series(d['obs'], {'four': 'fref_top8_test_acc', 'all': 'test_acc', 'nm': 'test_acc_nm', 'refit': 'refit_test_acc'}[key])


def crossings(runs, key):
    out = []
    for d in runs:
        sv = series(d, key)
        if sv is None: return None
        out.append([AX.interp_cross(sv[0], sv[1], q) for q in (0.1, 0.5, 0.9)])
    return np.array(out, dtype=float)


def shadowed(ax, x, y, color, ls='-', lw=1.4, z=3):
    ln, = ax.plot(x, y, color=color, ls=ls, lw=lw, zorder=z, solid_capstyle='round'); ln.set_path_effects(SHADOW)


def main(ana, outdir, main_cell):
    A = json.load(open(ana))['cells']; os.makedirs(outdir, exist_ok=True)
    cols = [('ZP_random_f0.33', 'Random split'), (main_cell, 'Mirror-free split')]
    data = [(A[k], load(A[k]), t) for k, t in cols]
    T = min(min(len(d['loss']) for d in runs) for _, runs, _ in data) - 1; g = np.arange(0, T + 1, 20)
    fig = plt.figure(figsize=(6.75, 7.6), layout='constrained')
    gs = fig.add_gridspec(5, 2, height_ratios=[0.85, 1.55, 1.0, 0.7, 1.05])
    axs = [[fig.add_subplot(gs[i, j]) for j in range(2)] for i in range(4)]
    ce_top = 0; record = {}
    for j, (cell, runs, ttl) in enumerate(data):
        p = runs[0]['rec']['p']; r0 = runs[0]['rec']; S = cell['summary']
        W = [r['rec']['all']['window']['W'] for r in runs if r['rec']['all'].get('window', {}).get('W')]
        a, b = np.median([w[0] for w in W]), np.median([w[1] for w in W])
        for i in range(4): axs[i][j].axvspan(a, min(b, T), color=C.C_BAND, lw=0, zorder=0)
        Y = PZ.ce_grid(runs, g) / math.log(p); C.band(axs[0][j], g, Y, C.C_CE); ce_top = max(ce_top, np.nanpercentile(Y[:, g >= 400], 99))
        ax = axs[1][j]
        ax.plot(g, np.nanmedian(PX.on_grid(runs, 'train_acc', g), 0), color=C.C_TRAIN, ls='--', lw=0.9, zorder=2)
        Ya, Yn, Ym = C.mirror_acc(runs, g) if r0['n_mirror'] > 0 else (PX.on_grid(runs, 'test_acc', g), None, None)
        med = {'four': np.nanmedian(PX.on_grid(runs, 'fref_top8_test_acc', g), 0), 'refit': np.nanmedian(PX.on_grid(runs, 'refit_test_acc', g), 0),
               'all': np.nanmedian(Ya, 0), 'nm': None if Yn is None else np.nanmedian(Yn, 0), 'mirror': None if Ym is None else np.nanmedian(Ym, 0)}
        for z, key in enumerate(('refit', 'four', 'mirror', 'nm', 'all')):
            k = [t for t in TRAJ if t[0] == key][0]
            if med[key] is not None: shadowed(ax, g, med[key], k[2], ls=k[3], lw=k[4], z=3 + z)
        a0 = np.median([r['rec']['a0_all'] for r in runs if 'a0_all' in r['rec']])
        ax.plot([a], [a0], 'o', ms=4.5, color=C.C_NET, mec='white', mew=0.8, zorder=9)
        if a0 >= 0.1: ax.annotate(f'{a0:.2f}', (a, a0), xytext=(5, -11), textcoords='offset points', fontsize=8, color='#222222')
        ax.set_ylim(-0.02, 1.04)
        axr = axs[2][j]; record[cell['runs'][0]['id'].rsplit('_s', 1)[0]] = rec = {}
        for i, (key, lab, col, ls, lw) in enumerate(TRAJ):
            X = crossings(runs, key)
            if X is None:
                axr.text(0.02 * T, i, 'none (no mirror pairs held out)', va='center', fontsize=7.5, color='#777777'); continue
            m = np.nanmedian(X, 0); rec[key] = dict(t10=m[0], t50=m[1], t90=m[2], t50_seeds=X[:, 1].tolist())
            axr.plot([m[0], m[2]], [i, i], color=col, lw=5.5, alpha=0.35, solid_capstyle='butt', zorder=2)
            for t in X[:, 1]: axr.plot([t, t], [i - 0.28, i + 0.28], color=col, lw=0.9, alpha=0.8, zorder=3)
            axr.plot([m[1]], [i], 'o', ms=5, color=col, mec='white', mew=0.8, zorder=4)
            axr.text(max(m[2], X[:, 1].max()) + 0.015 * T, i, SHORT[key], va='center', fontsize=7.5, color='#333333')
        axr.set_ylim(len(TRAJ) - 0.4, -0.6); axr.set_yticks([])
        axr.grid(True, axis='x', color=C.C_GRID, lw=0.6)
        C.band(axs[3][j], g, PX.on_grid(runs, 'AH_full_resid', g), C.C_AH); axs[3][j].set_ylim(0, 1)
        q = [r['rec']['all']['window']['q_te'] for r in runs if r['rec']['all'].get('window', {}).get('q_te') is not None]
        qs = f'{min(q):.2f}' if f'{min(q):.2f}' == f'{max(q):.2f}' else f'{min(q):.2f}-{max(q):.2f}'
        axs[0][j].set_title(f"{ttl}, fraction {r0['frac']:g}\n{r0['n_train']:,} training pairs, {r0['n_mirror']:,} held-out mirror pairs\n"
                            f"{S['pass_strict']}/{S['n']} pass (share {qs})")
    for i in range(4):
        for j in range(2):
            ax = axs[i][j]; ax.set_xlim(0, T)
            if i != 2: ax.grid(True, axis='y', color=C.C_GRID, lw=0.6)
            ax.xaxis.set_major_formatter(C.KFMT); ax.xaxis.set_major_locator(MaxNLocator(4, integer=True))
            if i < 3: ax.tick_params(labelbottom=False)
            if j and i != 2: ax.tick_params(labelleft=False)
        if i != 2: axs[i][1].sharey(axs[i][0])
    axs[0][0].set_ylim(0, 1.15 * ce_top)
    for i, l in ((0, 'Training CE / log p\n(201-update avg.)'), (1, 'Accuracy'), (3, 'Fourier $A_H$\n(head input)')): axs[i][0].set_ylabel(l)
    axs[2][0].set_ylabel('Rise: 0.1 to 0.9,\ndot at 0.5', labelpad=2)
    for j in range(2): axs[3][j].set_xlabel('Updates (thousands)')
    axb = fig.add_subplot(gs[4, :])
    z = np.linspace(0, 0.42, 200); axb.plot(z, np.minimum(1, (0.9 - np.maximum(z, 0.1)) / 0.8), color=C.C_REF, lw=0.9, ls='--', zorder=1)
    axb.axhline(0.75, color=C.C_REF, lw=0.8, ls=C.DOT, zorder=1)
    mk = {0: ('o', '#1c5cab', 'Random split'), 1: ('D', '#D55E00', 'Mirror-free split')}
    for j, (cell, runs, ttl) in enumerate(data):
        m_, c_, l_ = mk[j]
        for r in runs:
            rec_ = r['rec']; w = rec_['all'].get('window', {})
            if 'a0_all' not in rec_: continue
            axb.scatter([rec_['a0_all']], [w['q_te']], marker=m_, s=26, lw=0.9, edgecolor=c_, facecolor=c_ if rec_['pass_strict'] else 'white', zorder=3)
    axb.set_xlim(-0.01, 0.42); axb.set_ylim(0.6, 1.03); axb.grid(True, color=C.C_GRID, lw=0.6); axb.set_axisbelow(True)
    axb.set_xlabel('All-pairs held-out accuracy when the window opens, $a_0$'); axb.set_ylabel('Share of the held-out\nrise inside the window')
    axb.text(0.415, 0.757, 'pass: 0.75', ha='right', va='bottom', fontsize=8, color='#555555')
    axb.text(0.22, 0.86, r'$(0.9 - a_0)\,/\,0.8$', fontsize=8, color='#555555', rotation=-14)
    hb = [Line2D([], [], ls='', marker=mk[j][0], markersize=5, color=mk[j][1]) for j in (0, 1)] + \
         [Line2D([], [], ls='', marker='o', markersize=5, markerfacecolor='white', markeredgecolor='#444444')]
    axb.legend(hb, [mk[0][2], mk[1][2], 'open: run fails'], loc='lower left', fontsize=8, ncol=3, columnspacing=1.2, handletextpad=0.3)
    h = [Line2D([], [], color=C.C_CE, lw=1.4), Line2D([], [], color=C.C_TRAIN, lw=0.9, ls='--')]
    l = ['Train CE', 'Train acc.']
    for key, lab, col, ls, lw in TRAJ:
        ln = Line2D([], [], color=col, ls=ls, lw=lw); ln.set_path_effects(SHADOW); h.append(ln); l.append(lab)
    h += [Line2D([], [], color=C.C_AH, lw=1.4), Patch(color=C.C_BAND, lw=0)]; l += ['Fourier $A_H$', 'Window (all pairs)']
    fig.legend(h, l, loc='outside lower center', ncol=3, frameon=False, handlelength=1.8, columnspacing=1.6, fontsize=8)
    C.save(fig, outdir, 'ZM_mirrorfree_paper')
    json.dump(dict(doc='Median crossing times (updates) of each accuracy trajectory in ZM_mirrorfree_paper: t10, t50, t90, and '
                       'every seed t50; crossings interpolated on each curve own log.', cells=record),
              open(os.path.join(outdir, 'ZM_rise_times.json'), 'w'), indent=1, default=float)


if __name__ == '__main__':
    main(*sys.argv[1:4])
