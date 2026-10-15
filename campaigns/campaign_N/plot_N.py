#!/usr/bin/env python3
import json, math, os, sys
from pathlib import Path
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import numpy as np
import acrossp_figs as A
import endpoints_i as E
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb
from matplotlib.lines import Line2D

FRACS = [0.8, 0.7, 0.6, 0.5]
GROUPS = [(D, p) for D in (3, 4) for p in (47, 61)]
MARK = {47: 'o', 61: 's'}


def save(fig, out, stem):
    fig.savefig(out / f'{stem}.pdf', metadata={'Title': stem, 'Author': '', 'Creator': 'plot_N.py', 'CreationDate': None})
    fig.savefig(out / f'{stem}.png', dpi=200); plt.close(fig); print('drew', stem, flush=True)


def draw2(cols, out, stem):
    orig = A.plt.subplots
    A.plt.subplots = lambda *a, **k: orig(*a, **dict(k, squeeze=False))
    try: A.draw2(cols, out, stem, rows=('ce', 'acc', 'err', 'ah'), height=5.6)
    finally: A.plt.subplots = orig
    print('drew', stem, flush=True)


def column(nres, rows, title):
    runs = []
    for x in rows:
        d = E.load(os.path.join(nres, x['id'], 'run.json.gz')); w = x['M']['window']
        runs.append(A.deep_run(d, x['id'], (w['W'], x['M']['verdict']['pass_M'], 0.0, 0.0))); del d
    return A.column2(runs, title, [x['id'] for x in rows])


def ladders(nres, R, out):
    for D, p in GROUPS:
        cols = []
        for f in FRACS:
            c = R['cells'][f'D{D}_p{p}_f{f:g}']
            fin = [x for x in R['runs'].values() if x['depth'] == D and x['p'] == p and x['dial']['frac'] == f and x['state'] == 'finished']
            if not fin: continue
            cols.append(column(nres, sorted(fin, key=lambda x: x['seed']),
                               f"fraction {f:g}, r = {c['r']:g}\n{c['pass_M']}/{c['n']} pass_M"))
        if cols: draw2(cols, out / 'ladder', f'N_ladder_D{D}_p{p}')


def board(R, out):
    fig, ax = plt.subplots(figsize=(8.4, 4.2), layout='constrained')
    ax.set_xlim(0, len(FRACS)); ax.set_ylim(len(GROUPS), -0.3); ax.axis('off')
    green = np.array(to_rgb(A.K['AH'])); white = np.ones(3)
    for j, f in enumerate(FRACS): ax.text(j + 0.5, -0.08, f'fraction {f:g}', ha='center', va='bottom', fontsize=8)
    for i, (D, p) in enumerate(GROUPS):
        ax.text(-0.05, i + 0.5, f'depth {D}, p = {p}\n$\\lambda_V = 0.2\\cdot 31/p$', ha='right', va='center', fontsize=8)
        for j, f in enumerate(FRACS):
            c = R['cells'][f'D{D}_p{p}_f{f:g}']; n = c['n']; k = c['pass_M']
            frac = k / n if n else 0
            col = white * (1 - 0.85 * frac) + green * 0.85 * frac if frac > 0 else np.array([0.955] * 3)
            ax.add_patch(plt.Rectangle((j + 0.03, i + 0.05), 0.94, 0.9, color=col, lw=0))
            tc = 'white' if frac > 0.75 else 'black'; sc = '#555555' if tc == 'black' else tc
            md = c.get('M_dev'); q = c['q_te']['median']; ce = c['ce_change']['median']
            ax.text(j + 0.5, i + 0.2, f"r = {c['r']:g}", ha='center', va='center', fontsize=6.6, color=sc)
            ax.text(j + 0.5, i + 0.42, f'{k}/{n}' if n else 'not finished', ha='center', va='center', fontsize=9 if n else 6.4,
                    fontweight='bold' if n else 'normal', color=tc)
            ax.text(j + 0.5, i + 0.62, (f"M3 {c['pass_M3']}/{n}" + (f", dev {md['pass_M']}/{md['n']}" if md else '')) if n else
                    f"{c['of'] - n} to finish", ha='center', va='center', fontsize=5.8, color=sc)
            ax.text(j + 0.5, i + 0.8, (f'q_te {q:.2f}' + ('' if ce is None else f', CE {100 * ce:+.0f}%')) if q is not None else '',
                    ha='center', va='center', fontsize=5.6, color=sc)
        if i == 1: ax.plot([-1.3, len(FRACS)], [2, 2], color='#BBBBBB', lw=0.6, clip_on=False)
    fig.suptitle('pass_M on 10 fresh seeds per cell (refit criterion; at least 75% of the held-out and refit rises inside the '
                 'post-hoc window; A_H +0.02)\nsmall print: pass_M3; campaign M development count (seeds 1-3); median share of the '
                 'held-out rise inside the window; median CE change across the rise', fontsize=7.0)
    save(fig, out, 'N_scoreboard')


def runs_figure(R, out):
    fig, axs = plt.subplots(2, 2, figsize=(7.6, 4.8), sharex=True, layout='constrained')
    xs = {f: k for k, f in enumerate(FRACS)}
    for i, D in enumerate((3, 4)):
        for j, (key, line, ylab) in enumerate((('q', 0.75, 'Share of the held-out rise\ninside the window'),
                                                ('ce', 0.0, 'CE change, refit 0.2 to\nheld-out 0.9 (log scale)'))):
            ax = axs[i, j]
            for x in R['runs'].values():
                if x['depth'] != D or x['state'] != 'finished': continue
                m = x['M']; ok = m['verdict']['pass_M']
                y = m['window']['q_te'] if key == 'q' else m['timing'].get('ce_change_rf20_te90')
                if y is None: continue
                if key == 'ce': y = math.copysign(math.log10(1 + abs(y) * 100) , y)
                off = (-0.12 if x['p'] == 47 else 0.12) + 0.035 * (x['seed'] - 5.5) / 4.5
                ax.scatter(xs[x['dial']['frac']] + off, y, s=16, marker=MARK[x['p']], lw=0.8, zorder=3,
                           facecolor='#333333' if ok else 'white', edgecolor='#333333')
            ax.axhline(0 if key == 'ce' else line, color='#888888', lw=0.7, ls='--', zorder=1)
            ax.grid(True, axis='y', color=A.K['grid'], lw=0.6); ax.set_ylabel(ylab, fontsize=7.6)
            if key == 'ce':
                t = [-10, 0, 10, 100, 1000]; ax.set_yticks([math.copysign(math.log10(1 + abs(v)), v) for v in t])
                ax.set_yticklabels([f'{v:+d}%' if v else '0' for v in t])
            else: ax.set_ylim(-0.05, 1.05)
            ax.set_title(f'depth {D}', fontsize=8)
    for ax in axs[-1]: ax.set_xticks(range(len(FRACS))); ax.set_xticklabels([f'{f:g}' for f in FRACS]); ax.set_xlabel('Training fraction')
    h = [Line2D([], [], ls='', marker=MARK[p], color='#333333', markersize=5) for p in (47, 61)] + \
        [Line2D([], [], ls='', marker='o', markerfacecolor='white', markeredgecolor='#333333', markersize=5)]
    fig.legend(h, ['p = 47', 'p = 61', 'open: not pass_M'], loc='outside lower center', ncol=3, frameon=False, fontsize=7.6)
    save(fig, out, 'N_runs')


def main(nres, ana, outdir):
    out = Path(outdir); out.mkdir(parents=True, exist_ok=True); R = json.load(open(ana))
    board(R, out); runs_figure(R, out); ladders(nres, R, out)


if __name__ == '__main__':
    main(*sys.argv[1:4])
