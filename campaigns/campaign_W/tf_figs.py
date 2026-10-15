#!/usr/bin/env python3
import math, os
os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig_tf")
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter
import paperdefs_tf as P

K = dict(train="#0072B2", train_acc="#000000", test="#D55E00", refit="#7B3294", AH="#009E73", window="#EBEBEB",
         grid="#E6E6E6", layer="#444444")
plt.rcParams.update(matplotlib.rcParamsDefault)
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 8, 'axes.titlesize': 8, 'axes.labelsize': 8,
                     'xtick.labelsize': 8, 'ytick.labelsize': 8, 'legend.fontsize': 8, 'axes.spines.top': False,
                     'axes.spines.right': False, 'axes.edgecolor': '#555555', 'axes.linewidth': 0.7})
KFMT = FuncFormatter(lambda v, _: f"{v / 1000:g}")
THIN = dict(lw=0.5, alpha=0.35, zorder=2)


def on_grid(x, y, g):
    x = np.asarray(x, float); y = np.asarray(y, float); ok = np.isfinite(y); x, y = x[ok], y[ok]
    if not len(x): return np.full(len(g), np.nan)
    return np.interp(g, x, y, left=y[0], right=y[-1])


def column(runs, title, xmax=None, every=20, band='posthoc'):
    c0 = P.cfg_of(runs[0]); p = c0['p']; lp = math.log(p); ntr = int(c0['frac'] * p * p)
    T = max(len(r['loss']) for r in runs) - 1
    xmax = xmax or T
    g = np.arange(0, xmax + 1, every, dtype=float)
    S = dict(ce=[], raw=[], err=[], test=[], test_all=[], train=[], refit=[], probe=[], ahr=[], ahm=[])
    wins = []; NM = False
    for r in runs:
        L = np.asarray(r['loss'], float) / lp; cm = (L[:-1] + L[1:]) / 2
        t = np.arange(len(cm)); S['ce'].append(on_grid(t, cm, g))
        S['raw'].append((np.arange(len(L)), L))
        D = P.cfg_of(r).get('n_layers', 1); K = P.te_key(r); nm = '_nm' if K == 'test_acc_nm' else ''; NM = NM or bool(nm)
        for key, name in ((K, 'test'), ('test_acc', 'test_all'), ('train_acc', 'train'), ('refit_test_acc' + nm, 'refit'),
                          ('probe_test_acc' + nm, 'probe'), ('AH_full_resid', 'ahr'), (f'AH_full_h{D}', 'ahm')):
            s, v = P.series(r['obs'], key)
            S[name].append(on_grid(s, v, g) if len(s) else np.full(len(g), np.nan))
        s, v = P.series(r['obs'], 'train_acc'); S['err'].append(on_grid(s, (1 - v) * ntr, g))
        try:
            if band == 'posthoc':
                ph = P.posthoc(r)
                if ph.get('W2'): wins.append(ph['W2'])
            elif band == 'raw10':
                I = P.raw10(r)['fit'][1]
                if I: wins.append(list(I))
        except Exception:
            pass
    band = (float(np.median([w[0] for w in wins])), float(np.median([w[1] for w in wins]))) if wins else None
    return dict(title=title, g=g, S=S, band=band, xmax=xmax, n=len(runs), nm=NM)


def med(a):
    a = np.array(a, float)
    with np.errstate(all='ignore'):
        return np.nanmedian(a, 0) if np.isfinite(a).any() else None


def draw(cols, out, rows=('ce', 'acc', 'ah'), height=None, ce_log=False, legend=True, suptitle=None):
    n = len(cols); h = height or 1.45 * len(rows) + 0.6
    fig, axs = plt.subplots(len(rows), n, figsize=(max(6.75, 2.25 * n), h), sharey='row', layout='constrained', squeeze=False)
    for j, c in enumerate(cols):
        g, S = c['g'], c['S']
        for i, row in enumerate(rows):
            ax = axs[i][j]
            if c['band'] and c['band'][1] - c['band'][0] > 0.004 * c['xmax']: ax.axvspan(*c['band'], color=K['window'], lw=0, zorder=0)
            ax.grid(True, axis='y', color=K['grid'], lw=0.6); ax.set_xlim(0, c['xmax']); ax.xaxis.set_major_formatter(KFMT)
            if row == 'ce':
                for y in S['ce']: ax.plot(g, y, color=K['train'], **THIN)
                ax.plot(g, med(S['ce']), color=K['train'], lw=1.4, zorder=3)
                if ce_log: ax.set_yscale('log')
            elif row == 'raw':
                for t, y in S['raw']: ax.plot(t, y, color=K['train'], lw=0.4, alpha=0.5, zorder=2)
                ax.set_yscale('log')
            elif row == 'err':
                for y in S['err']: ax.plot(g, y, color=K['train_acc'], **THIN)
                ax.plot(g, med(S['err']), color=K['train_acc'], lw=1.1, zorder=3)
                ax.set_yscale('symlog', linthresh=1.0); ax.set_ylim(-0.2, 3000)
            elif row == 'acc':
                for y in S['test']: ax.plot(g, y, color=K['test'], **THIN)
                ax.plot(g, med(S['train']), color=K['train_acc'], lw=1.0, ls='--', zorder=3)
                if med(S['refit']) is not None: ax.plot(g, med(S['refit']), color=K['refit'], lw=1.4, zorder=4)
                if med(S['probe']) is not None: ax.plot(g, med(S['probe']), color=K['refit'], lw=1.2, ls=':', zorder=4)
                ax.plot(g, med(S['test']), color=K['test'], lw=1.4, zorder=5); ax.set_ylim(-0.02, 1.04)
                if c['nm']: ax.plot(g, med(S['test_all']), color=K['test'], lw=0.9, ls='-.', alpha=0.8, zorder=4)
            elif row == 'ah':
                if med(S['ahr']) is not None: ax.plot(g, med(S['ahr']), color=K['AH'], lw=1.4, zorder=3)
                if med(S['ahm']) is not None: ax.plot(g, med(S['ahm']), color=K['AH'], lw=1.2, ls='-.', zorder=3)
                ax.set_ylim(-0.02, 1.02)
            if i == 0: ax.set_title(f"{c['title']} ({c['n']} run{'s' if c['n'] != 1 else ''})")
            if i == len(rows) - 1: ax.set_xlabel('update (thousands)')
            if j == 0:
                ax.set_ylabel({'ce': 'CE / log p', 'raw': 'raw CE / log p', 'err': 'misclassified\ntraining pairs',
                               'acc': 'accuracy', 'ah': '$A_H$'}[row])
    if legend:
        hd = [Line2D([], [], color=K['train'], lw=1.4), Line2D([], [], color=K['train_acc'], lw=1.0, ls='--'),
              Line2D([], [], color=K['test'], lw=1.4), Line2D([], [], color=K['refit'], lw=1.4),
              Line2D([], [], color=K['refit'], lw=1.2, ls=':'), Line2D([], [], color=K['AH'], lw=1.4),
              Line2D([], [], color=K['AH'], lw=1.2, ls='-.')]
        anynm = any(c['nm'] for c in cols)
        lb = ['Training CE', 'Training acc.', 'Held-out acc.' + (' (no mirror)' if anynm else ''), 'Ridge refit', 'Logistic probe',
              '$A_H$ head input', '$A_H$ MLP']
        if anynm:
            hd.insert(3, Line2D([], [], color=K['test'], lw=0.9, ls='-.')); lb.insert(3, 'Held-out acc. (all pairs)')
        fig.legend(hd, lb, loc='outside lower center', ncol=7 if n >= 4 else 4, frameon=False, handlelength=1.8,
                   columnspacing=0.9, fontsize=8)
    if suptitle: fig.suptitle(suptitle, fontsize=9)
    fig.savefig(out + '.png', dpi=200); fig.savefig(out + '.pdf'); plt.close(fig)
    return out
