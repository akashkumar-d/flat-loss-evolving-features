#!/usr/bin/env python3
import json, math, os, sys
from pathlib import Path
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import numpy as np
import acrossp_figs as A
import endpoints_i as E
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb, LogNorm
from matplotlib import cm as mcm

GROUPS = [(D, p, rule) for D in (3, 4) for p in (47, 61) for rule in ('paper', 'lin')]
FRACS = [0.8, 0.7, 0.6, 0.5]
R_GRID = {3: {0.8: [0.45], 0.7: [0.45, 0.30], 0.6: [0.45, 0.28, 0.18], 0.5: [0.20, 0.13, 0.08]},
          4: {0.8: [0.45], 0.7: [0.45, 0.28], 0.6: [0.45, 0.20, 0.13], 0.5: [0.14, 0.09, 0.06]}}
SLOTS = [(0.8, 0), (0.7, 0), (0.7, 1), (0.6, 0), (0.6, 1), (0.6, 2), (0.5, 0), (0.5, 1), (0.5, 2), ('L', 0)]
RULE_TXT = dict(paper='$\\lambda_V$ = 0.2', lin='$\\lambda_V = 0.2\\cdot 31/p$')
RNORM = LogNorm(0.05, 0.5)
RMAP = lambda x: plt.get_cmap('viridis')(0.85 * (1 - x))


def lam_v(p, rule): return 0.2 if rule == 'paper' else round(0.2 * 31 / p, 4)


def setting_id(D, p, rule, f, r): return f'M_D{D}_p{p}_{rule}_f{f:.1f}_r{r:.2f}'


def l_id(D, p, rule, i): return f'L_D{D}_p{p}_lv{lam_v(p, rule):.4f}_lw0.015_s{i}'


def slot_rows(R, D, p, rule, slot):
    f, k = slot
    if f == 'L':
        return '0.5 (L)', 0.45, [R['controls_L'][l_id(D, p, rule, i)] for i in (1, 2, 3) if l_id(D, p, rule, i) in R['controls_L']], [], 0
    r = R_GRID[D][f][k]; st = setting_id(D, p, rule, f, r)
    rows = [R['runs'][f'{st}_s{i}'] for i in (1, 2, 3)]
    fin = [x for x in rows if x['state'] == 'finished']
    return f'{f:g}', r, fin, [x for x in rows if x['state'] == 'running'], sum(x['state'] == 'not started' for x in rows)


def board(R, out, stem, kind):
    fig, ax = plt.subplots(figsize=(10.4, 5.6), layout='constrained')
    ax.set_xlim(0, len(SLOTS)); ax.set_ylim(len(GROUPS), -0.32); ax.axis('off')
    green = np.array(to_rgb(A.K['AH'])); white = np.ones(3)
    heads = ['0.8', '0.7', '0.7', '0.6', '0.6', '0.6', '0.5', '0.5', '0.5', '0.5\ncampaign L']
    for j, h in enumerate(heads): ax.text(j + 0.5, -0.1, f'fraction {h}', ha='center', va='bottom', fontsize=7.2)
    for i, (D, p, rule) in enumerate(GROUPS):
        ax.text(-0.06, i + 0.5, f'depth {D}, p = {p}\n{RULE_TXT[rule]}', ha='right', va='center', fontsize=7.6)
        for j, slot in enumerate(SLOTS):
            lab, r, fin, run, left = slot_rows(R, D, p, rule, slot)
            n = len(fin)
            if kind == 'M':
                k = sum(bool(x['M']['verdict']['pass_M']) for x in fin); k3 = sum(bool(x['M']['verdict']['pass_M3']) for x in fin)
                kr = sum(bool(x['M']['refit']['ok']) for x in fin)
                q = np.median([x['M']['window']['q_te'] for x in fin]) if fin else None
                sub = f'refit {kr}/{n}, M3 {k3}/{n}' if n else ''
                sub2 = f'q_te {q:.2f}' if q is not None else ''
            else:
                k = sum(bool(x['M']['refit']['ok']) for x in fin); k3 = sum(bool(x['M']['refit']['ok3']) for x in fin)
                fails = sorted({w for x in fin for w in x['M']['refit']['why']})
                sub = f'pass3 {k3}/{n}' if n else ''
                sub2 = ('fails: ' + ', '.join(fails)) if fails else ''
            frac = k / n if n else 0
            col = white * (1 - 0.85 * frac) + green * 0.85 * frac if frac > 0 else np.array([0.955] * 3)
            ax.add_patch(plt.Rectangle((j + 0.03, i + 0.05), 0.94, 0.90, color=col, lw=0))
            tc = 'white' if frac > 0.6 else 'black'; sc = '#555555' if tc == 'black' else tc
            ax.text(j + 0.5, i + 0.2, f'r = {r:g}', ha='center', va='center', fontsize=6.4, color=sc)
            ax.text(j + 0.5, i + 0.42, f'{k}/{n}' if n else 'not finished', ha='center', va='center', fontsize=8.5 if n else 6.2,
                    fontweight='bold' if n else 'normal', color=tc)
            pend = ', '.join(x for x in ((f'{len(run)} running' if run else ''), (f'{left} to start' if left else '')) if x)
            ax.text(j + 0.5, i + 0.62, sub or pend, ha='center', va='center', fontsize=5.6, color=sc)
            ax.text(j + 0.5, i + 0.8, sub2 if n else '', ha='center', va='center', fontsize=5.4, color=sc)
        if i == 3: ax.plot([-1.6, len(SLOTS)], [4, 4], color='#BBBBBB', lw=0.6, clip_on=False)
    title = ('pass_M: refit criterion, and >= 75% of the held-out and refit rises inside the post-hoc window, and A_H +0.02'
             if kind == 'M' else 'Refit criterion (campaigns K and L): refit +0.2 in the post-hoc window, held-out +0.2 after its start, final >= 0.99')
    fig.suptitle(title, fontsize=8.2)
    fig.savefig(out / f'{stem}.pdf', metadata={'Title': stem, 'Author': '', 'Creator': 'plot_M.py', 'CreationDate': None})
    fig.savefig(out / f'{stem}.png', dpi=200); plt.close(fig); print('drew', stem, flush=True)


def scatter(R, out, stem, key, line, ylab, ylim):
    fig, axs = plt.subplots(2, 4, figsize=(9.6, 4.8), sharex=True, sharey=True, layout='constrained')
    xs = {0.8: 0, 0.7: 1, 0.6: 2, 0.5: 3}
    for gi, (D, p, rule) in enumerate(GROUPS):
        ax = axs[gi // 4, gi % 4]
        pts = []
        for f in FRACS:
            for k, r in enumerate(R_GRID[D][f]):
                st = setting_id(D, p, rule, f, r)
                for i in (1, 2, 3):
                    x = R['runs'][f'{st}_s{i}']
                    if x['state'] == 'finished': pts.append((f, r, key(x['M']), x['M']['verdict']['pass_M']))
        for i in (1, 2, 3):
            c = R['controls_L'].get(l_id(D, p, rule, i))
            if c: pts.append((0.5, 0.45, key(c['M']), c['M']['verdict']['pass_M']))
        for f, r, y, ok in pts:
            if y is None: continue
            off = (math.log(r) - math.log(0.2)) * -0.25
            ax.scatter(xs[f] + off, y, s=16, color=RMAP(RNORM(r)), edgecolor='black' if ok else 'none', lw=0.6, zorder=3)
        ax.axhline(line, color='#888888', lw=0.7, ls='--', zorder=1)
        ax.set_title(f'depth {D}, p = {p}, {RULE_TXT[rule]}', fontsize=7.2)
        ax.set_xticks(range(4)); ax.set_xticklabels([f'{f:g}' for f in FRACS]); ax.set_ylim(*ylim)
        ax.grid(True, axis='y', color=A.K['grid'], lw=0.6)
    for a in axs[-1]: a.set_xlabel('Training fraction')
    for a in axs[:, 0]: a.set_ylabel(ylab)
    from matplotlib.colors import ListedColormap
    from matplotlib.ticker import NullFormatter
    cmap = ListedColormap([RMAP(x) for x in np.linspace(0, 1, 256)])
    sm = mcm.ScalarMappable(norm=RNORM, cmap=cmap); cb = fig.colorbar(sm, ax=axs, shrink=0.8, pad=0.01)
    cb.ax.yaxis.set_minor_formatter(NullFormatter())
    cb.set_label('last hidden layer rate r'); cb.set_ticks([0.06, 0.1, 0.2, 0.3, 0.45]); cb.set_ticklabels(['0.06', '0.1', '0.2', '0.3', '0.45'])
    fig.suptitle('Each dot is a run (black edge: pass_M); within a fraction, smaller r sits further right', fontsize=8)
    fig.savefig(out / f'{stem}.pdf', metadata={'Title': stem, 'Author': '', 'Creator': 'plot_M.py', 'CreationDate': None})
    fig.savefig(out / f'{stem}.png', dpi=200); plt.close(fig); print('drew', stem, flush=True)


def draw2(cols, out, stem):
    orig = A.plt.subplots
    A.plt.subplots = lambda *a, **k: orig(*a, **dict(k, squeeze=False))
    try: A.draw2(cols, out, stem, rows=('ce', 'acc', 'err', 'ah'), height=5.6)
    finally: A.plt.subplots = orig


def column(fns_ids, scored, title):
    runs = []
    for (fn, rid), sc in zip(fns_ids, scored):
        d = E.load(fn); w = sc['window']; runs.append(A.deep_run(d, rid, (w['W'], sc['verdict']['pass_M'], 0.0, 0.0))); del d
    c = A.column2(runs, title, [rid for _, rid in fns_ids]); return c


def best_r(R, D, p, rule, f):
    best = None
    for r in R_GRID[D][f]:
        fin = [R['runs'][f'{setting_id(D, p, rule, f, r)}_s{i}'] for i in (1, 2, 3)]
        fin = [x for x in fin if x['state'] == 'finished']
        if not fin: continue
        key = (sum(x['M']['verdict']['pass_M'] for x in fin), sum(x['M']['verdict']['pass_M3'] for x in fin),
               float(np.median([x['M']['window']['q_te'] for x in fin])))
        if best is None or key > best[0]: best = (key, r, fin)
    return best


def group_figures(mres, lres, R, out):
    for D, p, rule in GROUPS:
        for part, fr in (('f0.8-0.6', (0.8, 0.7, 0.6)), ('f0.5', (0.5,))):
            cols = []
            for f in fr:
                for r in R_GRID[D][f]:
                    st = setting_id(D, p, rule, f, r)
                    fin = [R['runs'][f'{st}_s{i}'] for i in (1, 2, 3)]; fin = [x for x in fin if x['state'] == 'finished']
                    if not fin: continue
                    k = sum(x['M']['verdict']['pass_M'] for x in fin)
                    cols.append(column([(os.path.join(mres, x['id'], 'run.json.gz'), x['id']) for x in fin], [x['M'] for x in fin],
                                       f'{f:g}, r = {r:g}\n{k}/{len(fin)} pass_M'))
            if part == 'f0.5':
                ctl = [R['controls_L'][l_id(D, p, rule, i)] for i in (1, 2, 3) if l_id(D, p, rule, i) in R['controls_L']]
                if ctl:
                    k = sum(x['M']['verdict']['pass_M'] for x in ctl)
                    cols.append(column([(os.path.join(lres, x['id'], 'run.json.gz'), x['id']) for x in ctl], [x['M'] for x in ctl],
                                       f'0.5, r = 0.45 (L)\n{k}/{len(ctl)} pass_M'))
            if cols: draw2(cols, out / 'groups', f'M_D{D}_p{p}_{rule}_{part}'); print('drew group', D, p, rule, part, flush=True)
        cols = []
        for f in FRACS:
            b = best_r(R, D, p, rule, f)
            if b is None: continue
            (k, k3, q), r, fin = b
            cols.append(column([(os.path.join(mres, x['id'], 'run.json.gz'), x['id']) for x in fin], [x['M'] for x in fin],
                               f'fraction {f:g}, r = {r:g}\n{k}/{len(fin)} pass_M'))
        if cols: draw2(cols, out / 'ladder', f'M_ladder_D{D}_p{p}_{rule}'); print('drew ladder', D, p, rule, flush=True)


def main(mres, lres, runs_json, out):
    out = Path(out); out.mkdir(parents=True, exist_ok=True); R = json.load(open(runs_json))
    board(R, out, 'M_scoreboard', 'M'); board(R, out, 'M_refit_scoreboard', 'refit')
    scatter(R, out, 'M_share', lambda m: m['window']['q_te'], 0.75, 'Share of the held-out rise\ninside the window', (-0.05, 1.05))
    scatter(R, out, 'M_drawdown', lambda m: m['timing']['drawdown_te90'], 0.03, 'CE drawdown below its running max\nat held-out 0.9', (-0.01, 0.3))
    group_figures(mres, lres, R, out)


if __name__ == '__main__':
    main(*sys.argv[1:5])
