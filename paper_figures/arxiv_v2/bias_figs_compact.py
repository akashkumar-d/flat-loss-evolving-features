#!/usr/bin/env python3
import sys
sys.path.insert(0, sys.argv.pop(1))
import numpy as np
import bias_figs as BF
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator

A, K, KFMT, LAST, FIRST = BF.A, BF.K, BF.KFMT, BF.LAST, BF.FIRST
HEIGHT = 2.25


PANEL_W = {}
PAD = {}


def _grid(nr, n, pad):
    fig = plt.figure(figsize=(6.75, HEIGHT), layout='constrained')
    if pad > 0:
        sf = fig.subfigures(1, 3, width_ratios=[pad, 6.75 - 2 * pad, pad], wspace=0)
        axs = sf[1].subplots(nr, n, sharey='row')
    else:
        axs = fig.subplots(nr, n, sharey='row')
    return fig, np.atleast_2d(axs)


def _panel_w(fig, ax):
    fig.canvas.draw(); return ax.get_window_extent().width / fig.dpi


def draw(cols, out, stem, rows, height, legend=True, overlay=None, creator='bias_figs.py'):
    if len(cols) != 4:
        _draw(cols, out, stem, rows, height, legend, overlay, creator, keep=True); return
    PAD['pad'] = 0.0
    for _ in range(4):
        fig, w = _draw(cols, out, stem, rows, height, legend, overlay, creator, measure=True)
        target = PANEL_W['w']
        PAD['pad'] = max(0.0, PAD['pad'] + 4 * (w - target) / 2)
    _draw(cols, out, stem, rows, height, legend, overlay, creator)


def _draw(cols, out, stem, rows, height, legend=True, overlay=None, creator='bias_figs.py', keep=False, measure=False):
    overlay = overlay or {}
    n = len(cols); right = False
    fig, axs = _grid(len(rows), n, PAD.get('pad', 0.0) if n == 4 else 0.0)
    ah_top = max(float(np.nanmax(c['S']['ahL'])) for c in cols) * 1.08
    for j, c in enumerate(cols):
        g, S, B = c['g'], c['S'], overlay.get(j)
        for i, row in enumerate(rows):
            ax = axs[i, j]
            if c['band'][1] - c['band'][0] > 0.004 * c['xmax']: ax.axvspan(*c['band'], color=K['window'], lw=0, zorder=0)
            ax.grid(True, axis='y', color=K['grid'], lw=0.6); ax.set_xlim(0, c['xmax'])
            ax.xaxis.set_major_formatter(KFMT); ax.xaxis.set_major_locator(MaxNLocator(2 if n > 5 else 3, integer=True))
            thin = dict(lw=0.45, alpha=0.28, zorder=1)
            dot = dict(lw=BF.BIAS_LW, ls=BF.BIAS_LS, zorder=6)
            if row == 'ce':
                for y in S['ce']: ax.plot(g, y, color=K['train'], **thin)
                ax.plot(g, np.median(S['ce'], 0), color=K['train'], lw=1.4, zorder=3)
                if B is not None: ax.plot(g, B['ce'], color=K['train'], **dot)
                ax.set_ylim(0, 1.05)
            elif row == 'acc':
                for y in S['test']: ax.plot(g, y, color=K['test'], **thin)
                ax.plot(g, np.median(S['train'], 0), color=K['train_acc'], lw=1.0, ls='--', zorder=3)
                ax.plot(g, np.median(S['refit'], 0), color=K['refit'], lw=1.4, zorder=4)
                ax.plot(g, np.median(S['test'], 0), color=K['test'], lw=1.4, zorder=5); ax.set_ylim(-0.02, 1.04)
                if B is not None:
                    ax.plot(g, B['refit'], color=K['refit'], **dot); ax.plot(g, B['test'], color=K['test'], **dict(dot, zorder=7))
            elif row == 'ah':
                for y in S['ahL']: ax.plot(g, y, color=K['AH'], **thin)
                ax.plot(g, np.median(S['ahL'], 0), color=K['AH'], lw=1.4, ls=LAST, zorder=3)
                if S['ah1'] is not None: ax.plot(g, np.median(S['ah1'], 0), color=K['AH'], lw=1.2, ls=FIRST, zorder=3)
                if B is not None: ax.plot(g, B['ahL'], color=K['AH'], **dot)
                ax.set_ylim(0, ah_top)
            if i < len(rows) - 1: ax.tick_params(labelbottom=False)
        axs[0, j].set_title(c['title'], loc='center')
    lab = dict(ce='Train CE\n/ log p', acc='Accuracy', ah='Hidden\nFourier $A_H$')
    for i, row in enumerate(rows): axs[i, 0].set_ylabel(lab[row])
    xl = axs[-1, n // 2].set_xlabel('Updates (thousands)')
    h = [Line2D([], [], color=K['train'], lw=1.4), Line2D([], [], color=K['train_acc'], lw=1.0, ls='--'),
         Line2D([], [], color=K['test'], lw=1.4), Line2D([], [], color=K['refit'], lw=1.4)]
    l = ['Train CE', 'Train acc.', 'Held-out', 'Refit']
    if 'ah' in rows and any(c['S']['ah1'] is not None for c in cols):
        h += [Line2D([], [], color=K['layer'], lw=1.4, ls=LAST), Line2D([], [], color=K['layer'], lw=1.2, ls=FIRST)]
        l += ['Last hidden', 'First hidden']
    elif 'ah' in rows: h.append(Line2D([], [], color=K['AH'], lw=1.4)); l.append('$A_H$')
    if overlay: h.append(Line2D([], [], color=K['layer'], lw=BF.BIAS_LW, ls=BF.BIAS_LS)); l.append('With biases')
    if right:
        fig.legend(h, l, loc='outside right center', ncol=1, frameon=False, handlelength=1.8, fontsize=8)
    else:
        fig.legend(h, l, loc='outside lower center', ncol=len(h) if len(h) <= 7 else (len(h) + 1) // 2, frameon=False,
                   handlelength=1.8, columnspacing=0.9, fontsize=8)
    out.mkdir(parents=True, exist_ok=True)
    fig.canvas.draw()
    if keep: PANEL_W['w'] = _panel_w(fig, axs[-1, 0])
    if measure:
        w = _panel_w(fig, axs[-1, 0]); plt.close(fig); return fig, w
    if n % 2 == 0:
        r = fig.canvas.get_renderer(); bb = xl.get_window_extent(r).transformed(fig.transFigure.inverted())
        inv = fig.transFigure.inverted(); x0 = inv.transform((axs[-1, 0].get_window_extent(r).x0, 0))[0]
        x1 = inv.transform((axs[-1, -1].get_window_extent(r).x1, 0))[0]
        xl.set_color('none')
        fig.text((x0 + x1) / 2, (bb.y0 + bb.y1) / 2, 'Updates (thousands)', ha='center', va='center',
                 fontsize=xl.get_fontsize())
    fig.savefig(out / f'{stem}.pdf', metadata={'Title': stem, 'Author': '', 'Creator': 'bias_figs_compact.py', 'CreationDate': None})
    fig.savefig(out / f'{stem}.png', dpi=200); plt.close(fig)


BF.draw = draw
if __name__ == '__main__':
    BF.main()
