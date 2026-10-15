#!/usr/bin/env python3
import json, os, sys
from pathlib import Path
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import numpy as np
import acrossp_figs as A
import plot_N as P
import plot_N_cut as C
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator

ROWS = ('ce', 'acc', 'err', 'ah')
LAB = dict(ce='Training CE / log p\n(cycle mean)', acc='Accuracy', err='Misclassified\ntraining pairs', ah='Hidden Fourier\n$A_H$')


def draw(cols, out, stem):
    K = A.K; n = len(cols)
    fig, axs = plt.subplots(len(ROWS), n, figsize=(6.75, 5.6), sharey='row', layout='constrained', squeeze=False)
    ah_top = max(float(np.nanmax(c['S']['ahL'])) for c in cols) * 1.08
    err_top = max(c['ntr'] for c in cols) * 1.5
    thin = dict(lw=0.45, alpha=0.28, zorder=1)
    for j, c in enumerate(cols):
        g, S = c['g'], c['S']
        for i, row in enumerate(ROWS):
            ax = axs[i, j]
            if c['band'][1] - c['band'][0] > 0.004 * c['xmax']: ax.axvspan(*c['band'], color=K['window'], lw=0, zorder=0)
            ax.grid(True, axis='y', color=K['grid'], lw=0.6); ax.set_xlim(0, c['xmax'])
            ax.xaxis.set_major_formatter(A.KFMT); ax.xaxis.set_major_locator(MaxNLocator(3, integer=True))
            if row == 'ce':
                for y in S['ce']: ax.plot(g, y, color=K['train'], **thin)
                ax.plot(g, np.median(S['ce'], 0), color=K['train'], lw=1.4, zorder=3); ax.set_ylim(0, 1.05)
            elif row == 'acc':
                for y in S['test']: ax.plot(g, y, color=K['test'], **thin)
                ax.plot(g, np.median(S['train'], 0), color=K['train_acc'], lw=1.0, ls='--', zorder=3)
                ax.plot(g, np.median(S['refit'], 0), color=K['refit'], lw=1.4, zorder=4)
                ax.plot(g, np.median(S['test'], 0), color=K['test'], lw=1.4, zorder=5); ax.set_ylim(-0.02, 1.04)
            elif row == 'err':
                for y in S['err']: ax.plot(g, y, color=K['train_acc'], **thin)
                ax.plot(g, np.median(S['err'], 0), color=K['train_acc'], lw=1.1, zorder=3)
                ax.set_yscale('symlog', linthresh=1); ax.set_ylim(-0.2, err_top)
                ticks = [t for t in (0, 1, 10, 100, 1000) if t <= err_top]
                ax.set_yticks(ticks); ax.set_yticklabels([{1000: '1k'}.get(t, str(t)) for t in ticks])
            elif row == 'ah':
                for y in S['ahL']: ax.plot(g, y, color=K['AH'], **thin)
                ax.plot(g, np.median(S['ahL'], 0), color=K['AH'], lw=1.4, ls=A.LAST, zorder=3)
                ax.plot(g, np.median(S['ah1'], 0), color=K['AH'], lw=1.2, ls=A.FIRST, zorder=3)
                ax.set_ylim(0, ah_top)
            if i < len(ROWS) - 1: ax.tick_params(labelbottom=False)
        axs[0, j].set_title(c['title'], loc='center')
    for i, row in enumerate(ROWS): axs[i, 0].set_ylabel(LAB[row])
    axs[-1, n // 2].set_xlabel('Updates (thousands)')
    h = [Line2D([], [], color=K['train'], lw=1.4), Line2D([], [], color=K['train_acc'], lw=1.0, ls='--'),
         Line2D([], [], color=K['test'], lw=1.4), Line2D([], [], color=K['refit'], lw=1.4),
         Line2D([], [], color=K['train_acc'], lw=1.1), Line2D([], [], color=K['layer'], lw=1.4, ls=A.LAST),
         Line2D([], [], color=K['layer'], lw=1.2, ls=A.FIRST), Patch(color=K['window'], lw=0)]
    l = ['Train CE', 'Train acc.', 'Held-out', 'Refit', 'Misclassified', 'Last hidden', 'First hidden', 'Window (median)']
    fig.legend(h, l, loc='outside lower center', ncol=4, frameon=False, handlelength=1.8, columnspacing=1.2, fontsize=8)
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / f'{stem}.pdf', metadata={'Title': stem, 'Author': '', 'Creator': 'paper_N.py', 'CreationDate': None})
    fig.savefig(out / f'{stem}.png', dpi=200); plt.close(fig); print('drew', stem, flush=True)


def main(nres, ana, outdir):
    out = Path(outdir); R = json.load(open(ana))
    A.column2 = C.column2_cut
    for D, p in P.GROUPS:
        cols = []
        for f in P.FRACS:
            c = R['cells'][f'D{D}_p{p}_f{f:g}']
            fin = [x for x in R['runs'].values() if x['depth'] == D and x['p'] == p and x['dial']['frac'] == f and x['state'] == 'finished']
            assert len(fin) == c['n'] == 10, (D, p, f)
            cols.append(P.column(nres, sorted(fin, key=lambda x: x['seed']),
                                 f"Fraction {f:g}, r = {c['r']:g}\n{c['pass_M']}/{c['n']} seeds pass"))
        draw(cols, out, f'N_ladder_D{D}_p{p}_paper')


if __name__ == '__main__':
    main(*sys.argv[1:4])
