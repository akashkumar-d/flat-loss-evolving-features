#!/usr/bin/env python3
import json, math, os, sys
from pathlib import Path
CAMPAIGNS, DATA, AQ, OUT = sys.argv[1:5]
sys.argv = [sys.argv[0], CAMPAIGNS, DATA]
_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.environ.get('PAPER_UNIFORM_DIR', _here))
import paper_uniform as PU
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator

K, A, THIN, C_FOUR, DASH = PU.K, PU.A, PU.THIN, PU.C_FOUR, PU.DASH


def two_blocks_row(c, out, stem, refit=True):
    g, S = c['g'], c['S']
    fig, axs = plt.subplots(1, 4, figsize=(6.75, 1.9), layout='constrained')
    titles = ('Training CE / log p', 'Accuracy', 'Misclassified training pairs', 'Fourier $A_H$')
    err_top = c['ntr'] * 1.5
    for ax, ttl in zip(axs, titles):
        if c['band'] and c['band'][1] - c['band'][0] > 0.004 * c['xmax']:
            ax.axvspan(c['band'][0], min(c['band'][1], c['xmax']), color=K['window'], lw=0, zorder=0)
        ax.grid(True, axis='y', color=K['grid'], lw=0.6); ax.set_axisbelow(True); ax.set_xlim(0, c['xmax'])
        ax.xaxis.set_major_formatter(A.KFMT); ax.xaxis.set_major_locator(MaxNLocator(3, integer=True))
        ax.set_title(ttl, loc='center', fontsize=8); ax.set_xlabel(' ')
    ax = axs[0]
    for y in S['ce']: ax.plot(g, y, color=K['train'], **THIN)
    med = np.median(S['ce'], 0); ax.plot(g, med, color=K['train'], lw=1.4, zorder=3)
    ax.set_ylim(0, 1.18 * float(np.percentile(med[g >= 400], 99))); ax.yaxis.set_major_locator(MaxNLocator(3))
    ax = axs[1]
    for y in S['test']: ax.plot(g, y, color=K['test'], **THIN)
    ax.plot(g, np.median(S['train'], 0), color=K['train_acc'], lw=1.0, ls='--', zorder=3)
    if refit: ax.plot(g, np.median(S['refit'], 0), color=K['refit'], lw=1.3, zorder=4)
    ax.plot(g, np.median(S['four'], 0), color=C_FOUR, lw=1.3, zorder=4)
    ax.plot(g, np.median(S['test_nm'], 0), color=K['test'], lw=1.1, ls=DASH, zorder=5)
    ax.plot(g, np.median(S['test'], 0), color=K['test'], lw=1.4, zorder=6); ax.set_ylim(-0.02, 1.04)
    ax = axs[2]
    for y in S['err']: ax.plot(g, y, color=K['train_acc'], **THIN)
    ax.plot(g, np.median(S['err'], 0), color=K['train_acc'], lw=1.1, zorder=3)
    ax.set_yscale('symlog', linthresh=1); ax.set_ylim(-0.2, err_top)
    ticks = [t for t in (0, 1, 10, 100, 1000) if t <= err_top]
    ax.set_yticks(ticks); ax.set_yticklabels([{1000: '1k'}.get(t, str(t)) for t in ticks])
    ax = axs[3]
    for y in S['ahL']: ax.plot(g, y, color=K['AH'], **THIN)
    ax.plot(g, np.median(S['ahL'], 0), color=K['AH'], lw=1.4, zorder=3); ax.set_ylim(0, 1.0)
    h = [Line2D([], [], color=K['train'], lw=1.4), Line2D([], [], color=K['train_acc'], lw=1.0, ls='--'),
         Line2D([], [], color=K['test'], lw=1.4), Line2D([], [], color=K['test'], lw=1.1, ls=DASH),
         Line2D([], [], color=K['refit'], lw=1.3), Line2D([], [], color=C_FOUR, lw=1.3),
         Line2D([], [], color=K['train_acc'], lw=1.1), Line2D([], [], color=K['AH'], lw=1.4)]
    l = ['Train CE', 'Train acc.', 'Held-out', 'Held-out, no-mirror', 'Refit', 'Fourier refit', 'Misclassified', '$A_H$']
    if not refit: del h[4], l[4]
    fig.legend(h, l, loc='outside lower center', ncol=4, frameon=False, handlelength=1.6, columnspacing=0.9, fontsize=8)
    fig.canvas.draw(); fig.set_layout_engine('none')
    bb = axs[0].xaxis.label.get_window_extent(fig.canvas.get_renderer())
    y = fig.transFigure.inverted().transform((0, (bb.y0 + bb.y1) / 2))[1]
    fig.text((axs[0].get_position().x0 + axs[-1].get_position().x1) / 2, y, 'Updates (thousands)', ha='center', va='center', fontsize=8)
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / f'{stem}.pdf', metadata={'Title': stem, 'Author': '', 'Creator': 'paper_q_figs.py', 'CreationDate': None})
    fig.savefig(out / f'{stem}.png', dpi=200); plt.close(fig)
    st = dict(title=c['title'], n=c['n'], passes=c['passes'], xmax=int(c['xmax']), band=[round(x) for x in c['band']] if c['band'] else None)
    print(stem, json.dumps(st), flush=True)
    return st


def main():
    PU.style(); out = Path(OUT); Q = json.load(open(AQ))['cells']; rep = {}
    cols = [PU.tf_col('p = 61, mirror-free, fraction 0.55', Q['P_p61_mf'], '/'),
            PU.tf_col('p = 89, fraction 0.25', Q['P_p89'], '/'),
            PU.tf_col('p = 113, fraction 0.2', Q['P_p113'], '/'),
            PU.tf_col('p = 149, fraction 0.16', Q['P_p149'], '/')]
    rep['TF_acrossp_app'] = PU.draw(cols, out, 'TF_acrossp_app', ('ce', 'acc', 'err', 'ah'), 4.0)
    rep['TF_two_blocks_app'] = two_blocks_row(PU.tf_col('Two blocks, fraction 0.2', Q['B_base'], '/'), out, 'TF_two_blocks_app')
    json.dump(rep, open(out / 'paper_q_figs_stats.json', 'w'), indent=1)


if __name__ == '__main__':
    main()
