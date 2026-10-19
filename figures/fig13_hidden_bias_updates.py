#!/usr/bin/env python3
import argparse, json, os, sys
from pathlib import Path

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, 'lib'))
import numpy as np
import acrossp_figs as A
import matplotlib.pyplot as plt

K = A.K
COND = [('twin', 'No\nbias'), ('joint', 'Bias,\njoint\nstep'), ('K', 'Bias\nfrozen\nat knee'), ('KD', 'Bias\ndecays\nafter knee'),
        ('S', 'Bias,\nown\nstep')]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--analysis', required=True); ap.add_argument('--out', type=Path, required=True)
    a = ap.parse_args(); X = json.load(open(a.analysis)); seeds = list(range(1, 11))
    val = {}
    for key, _ in COND:
        for q in ('fall', 'share'):
            if key in ('twin', 'joint'): v = [X['references'][f'{key}_s{i}'][q] for i in seeds]
            else: v = [next(r for r in X['runs'].values() if r['arm'] == key and r['seed'] == i)[q] for i in seeds]
            val[(key, q)] = np.array(v, float)
    fig, axs = plt.subplots(1, 2, figsize=(6.75, 2.3), layout='constrained')
    rng = np.random.default_rng(0); jit = rng.uniform(-0.07, 0.07, len(seeds))
    for ax, (q, col, ylab, scale, ref) in zip(axs, (('fall', K['train'], 'Training-CE fall,\npeak to held-out $t_{90}$ (%)', 100, None),
                                                   ('share', K['test'], 'Captured share $q(I_C)$', 1, 0.75))):
        Y = np.array([val[(k, q)] * scale for k, _ in COND])
        for i in range(len(seeds)): ax.plot(range(len(COND)), Y[:, i], color=K['grid'], lw=0.8, zorder=1)
        for j in range(len(COND)):
            ax.scatter(j + jit, Y[j], s=14, color=col, edgecolor='white', linewidth=0.6, zorder=3)
            m = float(np.median(Y[j])); ax.plot([j - 0.24, j + 0.24], [m, m], color=K['train_acc'], lw=1.6, zorder=4)
        if ref is not None: ax.axhline(ref, color=K['null'], lw=0.8, ls='--', zorder=0)
        ax.set_xticks(range(len(COND))); ax.set_xticklabels([n for _, n in COND], fontsize=7)
        ax.set_ylabel(ylab); ax.grid(True, axis='y', color=K['grid'], lw=0.6); ax.set_axisbelow(True)
        ax.set_xlim(-0.5, len(COND) - 0.5)
    axs[0].set_ylim(bottom=0); axs[1].set_ylim(0, 1.05)
    a.out.mkdir(parents=True, exist_ok=True); fig.canvas.draw()
    fig.savefig(a.out / 'bias_ablation_1hl.pdf', metadata={'Title': 'bias_ablation_1hl', 'Author': '', 'Creator': 'bias_appendix_fig.py',
                                                             'CreationDate': None})
    fig.savefig(a.out / 'bias_ablation_1hl.png', dpi=200); plt.close(fig)
    stats = {f'{k}_{q}': dict(median=float(np.median(val[(k, q)])), per_seed=[float(x) for x in val[(k, q)]]) for k, _ in COND for q in ('fall', 'share')}
    (a.out / 'bias_ablation_1hl_stats.json').write_text(json.dumps(stats, indent=1) + '\n')
    print({k: v['median'] for k, v in stats.items()})


if __name__ == '__main__':
    main()
