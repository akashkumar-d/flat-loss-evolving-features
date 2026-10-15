#!/usr/bin/env python3
"""Campaign K, development stage: figures (claude/1hl-figures, 2026-10-08).

    python3 -I plot_K.py RESULTS_DIR OUT_DIR

RESULTS_DIR holds <run>/run.json.gz (e.g. collected/results). Draws with the paper's own renderer (acrossp_figs.py next to
this file, copied unchanged from the code release's paper_figures/bias/, SHA-256 7667d4b4...):

  K_depth{3,4}_<setting>.pdf/png   one column per prime (23, 47, 61, 101), the three development seeds of one setting:
      training CE / log p, accuracies (training, held-out, train-only refit), misclassified training pairs, hidden Fourier
      A_H (last and first hidden layer); thin lines are seeds, thick lines medians; the grey band is the median of campaign
      C's loss-only band W (the window the per-run endpoint scores). <setting> is sqrt_lw0.015 (lam_v = 0.2 (31/p)^(1/2),
      lam_w = 0.015, the rule that keeps p = 31's per-class head scale), sqrt_lw0.025, lin_lw0.015 or lin_lw0.025.
  K_scoreboard.pdf/png   passes of campaign C's per-run endpoint for every cell and setting, with the failing checks, next to
      the paper's setting (lam_v = 0.2; campaigns F and G, 10 seeds each, as tabulated in PLAN.md).
  K_figures_stats.json   per column: run ids, passes, band, and every run's endpoint fields.
"""
import gzip, json, math, os, statistics as S, sys
from pathlib import Path

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import numpy as np
import acrossp_figs as A
import endpoints_i as E
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb

PRIMES = (23, 47, 61, 101)
SETTINGS = (('sqrt', 0.015), ('sqrt', 0.025), ('lin', 0.015), ('lin', 0.025))
RULE_TXT = dict(sqrt='$\\lambda_V = 0.2\\sqrt{31/p}$', lin='$\\lambda_V = 0.2\\cdot 31/p$')
BASE = {(23, 3): (0, 10, 'share'), (23, 4): (0, 10, 'share'), (47, 3): (2, 10, 'fit'), (47, 4): (0, 10, 'fit'),
        (61, 3): (0, 10, 'fit'), (61, 4): (0, 10, 'fit'), (101, 3): (0, 10, 'fit'), (101, 4): (0, 10, 'fit')}


def lam_v(p, rule): return round(0.2 * math.sqrt(31 / p), 4) if rule == 'sqrt' else round(0.2 * 31 / p, 4)


def setting_id(D, p, rule, lw): return f'K_D{D}_p{p}_lv{lam_v(p, rule):.4f}_lw{lw:.3f}'


def load(fn):
    with gzip.open(fn, 'rt') as f: return json.load(f)


def relapses_in_band(d, W):
    s, m = E.series(d['obs'], 'train_margin_min')
    return int(np.sum(m[(s >= W[0]) & (s <= W[1])] <= 0)) if W else None


def score(res, rid):
    d = load(os.path.join(res, rid, 'run.json.gz')); f = E.fields(d)
    if f.get('W') is None:
        w = A.window_record(d); W, ok = w['W2'], False
    else:
        W, ok = f['W'], bool(f['primary'])
    r = A.deep_run(d, rid, (W, ok, 0.0, 0.0)); r['verdict'] = "campaign C's endpoint"
    f['relapses_in_band'] = relapses_in_band(d, f.get('W'))
    return r, f


def dynamics(res, out, stats):
    for D in (3, 4):
        for rule, lw in SETTINGS:
            cols = []
            for p in PRIMES:
                st = setting_id(D, p, rule, lw); ids = [f'{st}_s{i}' for i in (1, 2, 3)]
                ids = [i for i in ids if os.path.exists(os.path.join(res, i, 'run.json.gz'))]
                scored = [score(res, i) for i in ids]
                c = A.column2([r for r, _ in scored], f'p = {p}', ids)
                k = sum(bool(f.get('primary')) for _, f in scored)
                c['title'] = f'p = {p}  ({k}/{len(ids)} pass)\n$\\lambda_V$ = {lam_v(p, rule):g}'
                cols.append(c)
                stats.setdefault('columns', []).append(dict(depth=D, p=p, setting=st, rule=rule, lam_w=lw, passes=k, n=len(ids),
                                                            band_median=c['band'], xmax=c['xmax'],
                                                            runs={i: {kk: f.get(kk) for kk in ('primary', 'why_not', 'share', 'W',
                                                                                              'CE_peak_over_logp', 't10', 't90',
                                                                                              'final_test', 'dAH_last', 'knee',
                                                                                              'relapses_in_band')}
                                                                  for i, (_, f) in zip(ids, scored)}))
            stem = f'K_depth{D}_{rule}_lw{lw}'
            A.draw2(cols, out, stem, rows=('ce', 'acc', 'err', 'ah'), height=5.6)
            print('drew', stem, [(c['title'].split('\n')[0]) for c in cols], flush=True)


def scoreboard(stats, out):
    by = {(c['depth'], c['p'], c['rule'], c['lam_w']): c for c in stats['columns']}
    rows = [(D, p) for D in (3, 4) for p in PRIMES]
    heads = ['Paper setting\n$\\lambda_V$ = 0.2, $\\lambda_W$ = 0.015\n(campaigns F, G; 10 seeds)'] + \
            [f'{RULE_TXT[r]}\n$\\lambda_W$ = {lw:g}\n(3 seeds)' for r, lw in SETTINGS]
    fig, ax = plt.subplots(figsize=(6.75, 4.6), layout='constrained')
    ax.set_xlim(0, len(heads)); ax.set_ylim(len(rows), -0.05); ax.axis('off')
    green = np.array(to_rgb(A.K['AH'])); white = np.ones(3)
    for j, h in enumerate(heads):
        ax.text(j + 0.5, -0.12, h, ha='center', va='bottom', fontsize=7.5)
    for i, (D, p) in enumerate(rows):
        ax.text(-0.06, i + 0.5, f'depth {D}, p = {p}', ha='right', va='center', fontsize=8)
        cells = [BASE[(p, D)] + (None,)]
        for r, lw in SETTINGS:
            c = by[(D, p, r, lw)]; runs = list(c['runs'].values())
            why = sorted({x['why_not'] or 'pass' for x in runs})
            checks = sorted({w.strip() for x in runs if x['why_not'] for w in x['why_not'].split(',')})
            rel = [x['relapses_in_band'] for x in runs if x['relapses_in_band'] is not None]
            shares = [x['share'] for x in runs if x.get('share') is not None]
            name = dict(share='share', final='final acc.', F02='$A_H$ rise', fit='fit')
            sub = 'pass' if not checks else ', '.join(name[x] for x in ('fit', 'share', 'F02', 'final') if x in checks)
            extra = []
            if 'fit' in checks and rel: extra.append(f'median relapses {S.median(rel):.0f}')
            elif 'share' in checks and shares: extra.append(f'min share {min(shares):.2f}')
            cells.append((c['passes'], c['n'], sub, ', '.join(extra) if extra else None))
        for j, (k, n, sub, extra) in enumerate(cells):
            frac = k / n; col = white * (1 - 0.85 * frac) + green * 0.85 * frac if frac > 0 else np.array([0.955] * 3)
            ax.add_patch(plt.Rectangle((j + 0.03, i + 0.05), 0.94, 0.90, color=col, lw=0))
            tc = 'white' if frac > 0.6 else 'black'
            ax.text(j + 0.5, i + 0.27, f'{k}/{n}', ha='center', va='center', fontsize=9, fontweight='bold', color=tc)
            ax.text(j + 0.5, i + 0.55, sub, ha='center', va='center', fontsize=6.6, color=tc)
            if extra: ax.text(j + 0.5, i + 0.78, extra, ha='center', va='center', fontsize=6.2, color='#444444' if tc == 'black' else tc)
        if i == 3: ax.plot([-1.2, len(heads)], [4, 4], color='#BBBBBB', lw=0.6, clip_on=False)
    fig.savefig(out / 'K_scoreboard.pdf', metadata={'Title': 'K_scoreboard', 'Author': '', 'Creator': 'plot_K.py', 'CreationDate': None})
    fig.savefig(out / 'K_scoreboard.png', dpi=200); plt.close(fig)
    print('drew K_scoreboard', flush=True)


def main(res, out, only_scoreboard=False):
    out = Path(out); out.mkdir(parents=True, exist_ok=True); stats = dict(note=__doc__.split('\n\n')[0])
    if only_scoreboard: stats = json.loads((out / 'K_figures_stats.json').read_text())
    else: dynamics(res, out, stats)
    scoreboard(stats, out)
    (out / 'K_figures_stats.json').write_text(json.dumps(stats, indent=1, default=float) + '\n')


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2], '--scoreboard-only' in sys.argv)
