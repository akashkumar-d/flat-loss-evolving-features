#!/usr/bin/env python3
import json, math, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import plot_Z as PZ
import plot_X as PX
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter, MultipleLocator

XMAX = 9000
C_REFIT = '#7B3294'
COLS = (('Y_G2', 'Polar (the paper\'s rule)'), ('C_NGD_G2', 'Normalized gradient (NGD)'), ('C_ADAMW_G2', 'AdamW'))
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 8, 'axes.spines.top': False, 'axes.spines.right': False,
                     'axes.titlesize': 8.5, 'legend.frameon': False})


def main(ana, outdir):
    A = json.load(open(ana))['cells']; os.makedirs(outdir, exist_ok=True)
    rows = ('ce', 'acc', 'ah', 'gain'); hr = dict(ce=1.0, acc=1.35, ah=0.8, gain=0.55)
    fig, axs = plt.subplots(len(rows), len(COLS), figsize=(6.75, 5.6), sharex=True, squeeze=False,
                            gridspec_kw=dict(height_ratios=[hr[r] for r in rows], hspace=0.16, wspace=0.28))
    grid = np.arange(0, XMAX + 1, 20)
    for j, (key, ttl) in enumerate(COLS):
        cell = A[key]; runs = PZ.load_runs(cell); S = cell['summary']; p = runs[0]['rec']['p']
        assert len(runs) == 10 and min(len(d['loss']) for d in runs) > XMAX, key
        W = [r['nm'].get('window', {}).get('W') for r in cell['runs']]; W = [w for w in W if w]
        band = (np.median([w[0] for w in W]), np.median([w[1] for w in W])) if W else None
        for i, rname in enumerate(rows):
            ax = axs[i, j]
            if band: ax.axvspan(band[0], min(band[1], XMAX), color=PX.C_BAND, lw=0, zorder=0)
            if rname == 'ce':
                Y = PZ.ce_grid(runs, grid) / math.log(p)
                PZ.draw(ax, grid, Y, PX.C_CE, lw=1.4)
                top = np.nanpercentile(Y[:, grid >= 400], 99)
                ax.set_ylim(0, 1.15 * top)
            elif rname == 'acc':
                PZ.draw(ax, grid, PX.on_grid(runs, 'train_acc', grid), PX.C_TRAIN, ls='--', lw=0.9)
                PZ.draw(ax, grid, PX.on_grid(runs, 'refit_test_acc', grid), C_REFIT, lw=1.3)
                PZ.draw(ax, grid, PX.on_grid(runs, 'fref_top8_test_acc', grid), PX.C_FOUR, lw=1.3)
                PZ.draw(ax, grid, PX.on_grid(runs, 'test_acc', grid), PX.C_NET, lw=1.7)
                PZ.draw(ax, grid, PX.on_grid(runs, 'test_acc_nm', grid), PX.C_NET, ls=PZ.DASH, lw=1.1)
                ax.set_ylim(-0.02, 1.03); ax.grid(alpha=0.25, lw=0.5)
            elif rname == 'ah':
                PZ.draw(ax, grid, PX.on_grid(runs, 'AH_full_resid', grid), PX.C_AH, lw=1.4)
                ax.set_ylim(0, 1)
            elif rname == 'gain':
                PZ.draw(ax, grid, PX.on_grid(runs, 'gain', grid), PZ.C_GAIN, lw=1.3)
                ax.set_ylim(0.8, 2.2)
            ax.set_xlim(0, XMAX)
        axs[0, j].set_title(f"{ttl}\n{S['pass_strict']}/{S['n']} seeds pass", fontsize=8.5)
    lab = dict(ce='Training CE / log p\n(201-update avg.)', acc='Accuracy', ah='Fourier $A_H$\n(head input)', gain='Gain $g$')
    for i, r in enumerate(rows): axs[i, 0].set_ylabel(lab[r])
    for ax in axs[-1]:
        ax.xaxis.set_major_locator(MultipleLocator(3000)); ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f'{v / 1000:g}'))
    axs[-1, 1].set_xlabel('Updates (thousands)')
    h = [Line2D([], [], color=PX.C_CE, lw=1.4), Line2D([], [], color=PX.C_TRAIN, lw=0.9, ls='--'),
         Line2D([], [], color=PX.C_NET, lw=1.7), Line2D([], [], color=PX.C_NET, lw=1.1, ls=PZ.DASH),
         Line2D([], [], color=PX.C_FOUR, lw=1.3), Line2D([], [], color=C_REFIT, lw=1.3),
         Line2D([], [], color=PX.C_AH, lw=1.4), Line2D([], [], color=PZ.C_GAIN, lw=1.3), Patch(color=PX.C_BAND, lw=0)]
    l = ['Train CE', 'Train acc.', 'Held-out (all pairs)', 'Held-out (no-mirror pairs)', 'Fourier refit (top 8)',
         'Refit', 'Fourier $A_H$', 'Gain', 'Window (median)']
    fig.legend(h, l, loc='lower center', ncol=5, fontsize=8, handlelength=1.8, columnspacing=1.0, bbox_to_anchor=(0.5, 0.0))
    fig.subplots_adjust(left=0.1, right=0.99, top=0.92, bottom=0.135)
    stem = os.path.join(outdir, 'Z_contrast_gain_paper')
    fig.savefig(stem + '.pdf', metadata={'Title': 'Z_contrast_gain_paper', 'Author': '', 'Creator': 'paper_contrast.py', 'CreationDate': None})
    fig.savefig(stem + '.png', dpi=200); plt.close(fig); print('drew', stem)


if __name__ == '__main__':
    main(*sys.argv[1:3])
