#!/usr/bin/env python3
import argparse, json, math, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import _paths as P
import endpoints as E
from make_figs_campaigns import Runs
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter, MaxNLocator

K = dict(train="#0072B2", train_acc="#000000", test="#D55E00", refit="#CC79A7", AH="#009E73", window="#EBEBEB", grid="#E6E6E6",
         layer="#444444", agop="#E69F00", null="#888888")
plt.rcParams.update(matplotlib.rcParamsDefault)
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 8, 'axes.titlesize': 8, 'axes.labelsize': 8, 'xtick.labelsize': 8,
                     'ytick.labelsize': 8, 'legend.fontsize': 8, 'axes.spines.top': False, 'axes.spines.right': False,
                     'axes.edgecolor': '#555555', 'axes.linewidth': 0.7})
KFMT = FuncFormatter(lambda v, _: f"{v / 1000:g}")
FIRST, LAST = '-.', '-'


def run_series(d):
    c = d['cfg']; p = c['p']; D = c['depth']; L = np.asarray(d['loss'], float); obs = d['obs']
    s, te = E.series(obs, 'test_acc'); _, tr = E.series(obs, 'train_acc')
    sr, rf = E.series(obs, 'v1_ridge_acc_te'); refit_kind = 'canonical'
    if not len(sr): sr, rf = E.series(obs, 'refit_acc'); refit_kind = 'legacy'
    ag = {}
    for name in ('first', 'last'):
        sg, pur = E.series(obs, f'v1_{name}_agop_purity'); sg2, rmed = E.series(obs, f'v1_{name}_agop_rand_med')
        ag[name] = (sg, pur, sg2, rmed) if len(sg) and len(sg2) else None
    sm, mse = E.series(obs, 'v1_ridge_mse_te')
    sa, ahL = E.series(obs, f'AH_full_h{D}'); sa1, ah1 = E.series(obs, 'AH_full_h1')
    f = E.run_fields(d); ntr = int(c.get('frac', 0.8) * p * p)
    return dict(p=p, depth=D, L=L, test=(s, te), train=(s, tr), err=(s, (1 - tr) * ntr), refit=(sr, rf), ahL=(sa, ahL),
                ah1=(sa1, ah1) if D > 1 else None, refit_kind=refit_kind, agop=ag, mse=(sm, mse) if len(sm) else None,
                W=f.get('W'), primary=bool(f.get('primary')), share=f.get('share'),
                level=f.get('CE_peak_over_logp'), why=f.get('why_not'), t90=f.get('t90') or f.get('t90_first'), ntr=ntr)


def on_grid(x, y, g):
    x = np.asarray(x, float); y = np.asarray(y, float); ok = np.isfinite(y); x, y = x[ok], y[ok]
    return np.interp(g, x, y, left=y[0], right=y[-1])


def column(runs, title, ids):
    runs = [r for r in runs if r['W']]
    t90 = [r['t90'] or r['W'][1] for r in runs]
    xmax = int(math.ceil((1.15 * max(np.median([r['W'][1] for r in runs]), np.percentile(t90, 90)) + 1500) / 1000) * 1000)
    g = np.arange(0, xmax + 1, max(10, xmax // 2000), dtype=float)
    cm = lambda r: (r['L'][:-1] + r['L'][1:]) / 2 / math.log(r['p'])
    S = dict(ce=np.array([on_grid(np.arange(len(r['L']) - 1), cm(r), g) for r in runs]),
             test=np.array([on_grid(*r['test'], g) for r in runs]), train=np.array([on_grid(*r['train'], g) for r in runs]),
             err=np.array([on_grid(*r['err'], g) for r in runs]), refit=np.array([on_grid(*r['refit'], g) for r in runs]),
             ahL=np.array([on_grid(*r['ahL'], g) for r in runs]),
             ah1=np.array([on_grid(*r['ah1'], g) for r in runs]) if runs[0]['ah1'] is not None else None)
    for name in ('first', 'last'):
        ok = all(r['agop'][name] is not None for r in runs)
        S[f'ag_{name}'] = np.array([on_grid(*r['agop'][name][:2], g) for r in runs]) if ok else None
        S[f'agr_{name}'] = np.array([on_grid(*r['agop'][name][2:], g) for r in runs]) if ok else None
    S['mse'] = np.array([on_grid(*r['mse'], g) for r in runs]) if all(r['mse'] is not None for r in runs) else None
    band = (float(np.median([r['W'][0] for r in runs])), float(np.median([r['W'][1] for r in runs])))
    stats = dict(title=title, runs=ids, n=len(runs), passes=int(sum(r['primary'] for r in runs)), band_median=band,
                 level_median=float(np.median([r['level'] for r in runs])),
                 level_q10_q90=[float(v) for v in np.percentile([r['level'] for r in runs], [10, 90])],
                 share_median=float(np.median([r['share'] for r in runs])),
                 failed_checks=sorted({w for r in runs if r['why'] for w in r['why'].split(', ')}),
                 refit=sorted({r['refit_kind'] for r in runs}))
    return dict(g=g, S=S, band=band, xmax=xmax, title=title, stats=stats, ntr=max(r['ntr'] for r in runs))


def draw(cols, out, stem, rows=('ce', 'acc', 'err', 'ah'), height=5.2):
    n = len(cols); fig, axs = plt.subplots(len(rows), n, figsize=(6.75, height), sharey='row', layout='constrained')
    axs = np.atleast_2d(axs)
    ah_top = max(float(np.nanmax(c['S']['ahL'])) for c in cols) * 1.08
    err_top = max(c['ntr'] for c in cols) * 1.5
    agop_top = max([float(np.nanmax(c['S'][k])) for c in cols for k in ('ag_first', 'ag_last', 'agr_first', 'agr_last')
                    if c['S'].get(k) is not None] or [1.0]) * 1.08
    mse_top = max([float(np.nanmax(np.percentile(c['S']['mse'], 90, axis=0))) for c in cols if c['S'].get('mse') is not None]
                  or [1.0]) * 1.15
    for j, c in enumerate(cols):
        g, S = c['g'], c['S']
        for i, row in enumerate(rows):
            ax = axs[i, j]
            ax.axvspan(*c['band'], color=K['window'], lw=0, zorder=0)
            ax.grid(True, axis='y', color=K['grid'], lw=0.6); ax.set_xlim(0, c['xmax'])
            ax.xaxis.set_major_formatter(KFMT); ax.xaxis.set_major_locator(MaxNLocator(2 if n > 5 else 3, integer=True))
            thin = dict(lw=0.45, alpha=0.28, zorder=1)
            if row == 'ce':
                for y in S['ce']: ax.plot(g, y, color=K['train'], **thin)
                ax.plot(g, np.median(S['ce'], 0), color=K['train'], lw=1.4, zorder=3); ax.set_ylim(0, 1.05)
            elif row == 'acc':
                for y in S['test']: ax.plot(g, y, color=K['test'], **thin)
                ax.plot(g, np.median(S['train'], 0), color=K['train_acc'], lw=1.0, ls='--', zorder=3)
                ax.plot(g, np.median(S['refit'], 0), color=K['refit'], lw=1.4, ls=':', zorder=4)
                ax.plot(g, np.median(S['test'], 0), color=K['test'], lw=1.4, zorder=5); ax.set_ylim(-0.02, 1.04)
            elif row == 'err':
                for y in S['err']: ax.plot(g, y, color=K['train_acc'], **thin)
                ax.plot(g, np.median(S['err'], 0), color=K['train_acc'], lw=1.1, zorder=3)
                ax.set_yscale('symlog', linthresh=1); ax.set_ylim(-0.2, err_top)
                ticks = [t for t in (0, 1, 10, 100, 1000, 10000) if t <= err_top]
                ax.set_yticks(ticks); ax.set_yticklabels([{1000: '1k', 10000: '10k'}.get(t, str(t)) for t in ticks])
            elif row == 'agop':
                if S['ag_last'] is not None:
                    for y in S['ag_last']: ax.plot(g, y, color=K['agop'], **thin)
                    ax.plot(g, np.median(S['agr_last'], 0), color=K['null'], lw=0.9, ls=':', zorder=2)
                    ax.plot(g, np.median(S['ag_last'], 0), color=K['agop'], lw=1.4, ls=LAST, zorder=3)
                if S['ag_first'] is not None:
                    ax.plot(g, np.median(S['agr_first'], 0), color=K['null'], lw=0.9, ls='--', zorder=2)
                    ax.plot(g, np.median(S['ag_first'], 0), color=K['agop'], lw=1.2, ls=FIRST, zorder=3)
                ax.set_ylim(0, agop_top)
            elif row == 'mse':
                for y in S['mse']: ax.plot(g, y, color=K['refit'], **thin)
                ax.plot(g, np.median(S['mse'], 0), color=K['refit'], lw=1.4, ls=':', zorder=3); ax.set_ylim(0, mse_top)
            elif row == 'ah':
                for y in S['ahL']: ax.plot(g, y, color=K['AH'], **thin)
                ax.plot(g, np.median(S['ahL'], 0), color=K['AH'], lw=1.4, ls=LAST, zorder=3)
                if S['ah1'] is not None: ax.plot(g, np.median(S['ah1'], 0), color=K['AH'], lw=1.2, ls=FIRST, zorder=3)
                ax.set_ylim(0, ah_top)
            if i < len(rows) - 1: ax.tick_params(labelbottom=False)
        st = c['stats']; axs[0, j].set_title(f"{c['title']}\n{st['passes']}/{st['n']} pass", loc='center')
    lab = dict(ce='Training CE / log p\n(cycle mean)', acc='Accuracy', err='Misclassified\ntraining pairs', ah='Hidden Fourier\n$A_H$',
               agop='Hidden AGOP purity\n(class-centered, q = 4)', mse='Refit held-out\nMSE')
    if height < 4:
        lab.update(ce='Train CE\n/ log p', ah='Hidden\nFourier $A_H$')
    for i, row in enumerate(rows): axs[i, 0].set_ylabel(lab[row])
    axs[-1, n // 2].set_xlabel('Updates (thousands)')
    h = [Line2D([], [], color=K['train'], lw=1.4), Line2D([], [], color=K['train_acc'], lw=1.0, ls='--'),
         Line2D([], [], color=K['test'], lw=1.4), Line2D([], [], color=K['refit'], lw=1.4, ls=':')]
    l = ['Train CE', 'Train acc.', 'Held-out', 'Refit']
    if 'err' in rows: h.append(Line2D([], [], color=K['train_acc'], lw=1.1)); l.append('Misclassified')
    if 'agop' in rows:
        h += [Line2D([], [], color=K['agop'], lw=1.4), Line2D([], [], color=K['null'], lw=0.9, ls='--')]
        l += ['AGOP purity', 'Random frames']
    if 'ah' in rows and any(c['S']['ah1'] is not None for c in cols):
        h += [Line2D([], [], color=K['layer'], lw=1.4, ls=LAST), Line2D([], [], color=K['layer'], lw=1.2, ls=FIRST)]
        l += ['Last hidden', 'First hidden']
    elif 'ah' in rows: h.append(Line2D([], [], color=K['AH'], lw=1.4)); l.append('$A_H$')
    fig.legend(h, l, loc='outside lower center', ncol=len(h) if len(h) <= 7 else (len(h) + 1) // 2, frameon=False,
               handlelength=1.8, columnspacing=0.9, fontsize=8)
    for ext in ('pdf', 'png'): fig.savefig(os.path.join(out, f'{stem}.{ext}'), dpi=300)
    plt.close(fig)


def specs():
    H = lambda p, lw, w=None: [f"H{'W' if w else ''}_p{p}{f'_w{w}' if w else ''}_lw{lw:.4f}_s{i}" for i in range(1, 11)]
    Dd = lambda D, p, r: [f'D{D}_p{p}_r{r:.2f}_s{i}' for i in range(1, 11)]
    C2 = lambda D: (f'C1_D{D}_s', f'C2_D{D}_s')
    d2 = {23: [f'T16_D2_p23_lw0.015_r0.45_s{i}_T80k' for i in range(10)], 31: [f'seedD2_r0.5_s{i}' for i in range(10)],
          47: [f'T3_D2_p47_r0.5_s{i}' for i in range(10)]}
    p31 = [('Depth 1 (width 512)', H(31, 0.015)), ('Depth 2', d2[31]), ('Depth 3', C2(3)), ('Depth 4', C2(4)), ('Depth 5', C2(5))]
    return {
        'fig3_depths_p31': (p31, ('ce', 'acc', 'ah'), 3.0),
        'depths_p31_full': (p31, ('ce', 'acc', 'err', 'ah'), 5.2),
        'acrossp_1hl': ([('p = 23', H(23, 0.01)), ('p = 31', H(31, 0.015)), ('p = 47', H(47, 0.02)), ('p = 61', H(61, 0.02)),
                         ('p = 97', H(97, 0.02)), ('p = 97 (2048)', H(97, 0.015, 2048))], ('ce', 'acc', 'err', 'ah'), 5.2),
        'acrossp_depth3': ([('p = 23', Dd(3, 23, 0.45)), ('p = 31', C2(3)), ('p = 47', Dd(3, 47, 0.45)), ('p = 61', Dd(3, 61, 0.45)),
                            ('p = 97', Dd(3, 97, 0.45))], ('ce', 'acc', 'err', 'ah'), 5.2),
        'acrossp_depth4': ([('p = 23', Dd(4, 23, 0.32)), ('p = 31', C2(4)), ('p = 47', Dd(4, 47, 0.45)), ('p = 61', Dd(4, 61, 0.45)),
                            ('p = 97', Dd(4, 97, 0.45))], ('ce', 'acc', 'err', 'ah'), 5.2),
        'acrossp_depth2': ([('p = 23', d2[23]), ('p = 31', d2[31]), ('p = 47', d2[47])], ('ce', 'acc', 'err', 'ah'), 5.2),
        'fig2_acrossp_1hl': ([('p = 23', H(23, 0.01)), ('p = 31', H(31, 0.015)), ('p = 47', H(47, 0.02)), ('p = 61', H(61, 0.02)),
                              ('p = 101', H(101, 0.02))], ('ce', 'acc', 'ah'), 2.6),
        'fig3_depths_2to5': (p31[1:], ('ce', 'acc', 'ah'), 2.6),
        'acrossp_depth3_3row': ([('p = 23', Dd(3, 23, 0.45)), ('p = 31', C2(3)), ('p = 47', Dd(3, 47, 0.45)), ('p = 61', Dd(3, 61, 0.45)),
                                 ('p = 101', Dd(3, 101, 0.45))], ('ce', 'acc', 'ah'), 2.3, dict(legend=False)),
        'acrossp_depth4_3row': ([('p = 23', Dd(4, 23, 0.32)), ('p = 31', C2(4)), ('p = 47', Dd(4, 47, 0.45)), ('p = 61', Dd(4, 61, 0.45)),
                                 ('p = 101', Dd(4, 101, 0.45))], ('ce', 'acc', 'ah'), 2.6),
        'campaign_medians_D345': ([('Depth 3', C2(3)), ('Depth 4', C2(4)), ('Depth 5', C2(5))],
                                  ('ce', 'acc', 'agop', 'ah', 'mse'), 6.4),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--out', default=os.path.join(HERE, 'out'))
    ap.add_argument('--only', default='')
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    stores = [Runs(str(P.CAMPAIGN['C']), 'C'), Runs(str(P.CAMPAIGN['D']), 'D'), Runs(str(P.CAMPAIGN['F']), 'F'),
              Runs(str(P.CAMPAIGN['G']), 'G'), Runs(str(P.DEV_RUNS), 'dev')]
    def load(rid):
        hits = [s for s in stores if s.has(rid)]
        if len(hits) > 1: raise SystemExit(f'{rid} is in more than one archive folder: {[s.kind for s in hits]}')
        return hits[0].load(rid) if hits else None
    cache, report = {}, {}
    for fid, (cols, rows, height) in specs().items():
        if a.only and fid not in a.only.split(','): continue
        built = []
        for title, ids in cols:
            if isinstance(ids, tuple):
                ids = sorted(k for k in stores[0].idx if k.startswith(ids))
            got = []
            for rid in ids:
                if rid not in cache:
                    d = load(rid); cache[rid] = run_series(d) if d is not None else None
                if cache[rid] is not None: got.append((rid, cache[rid]))
            missing = [rid for rid in ids if cache.get(rid) is None]
            built.append(column([r for _, r in got], title, [rid for rid, _ in got]))
            built[-1]['stats']['missing'] = missing
        draw(built, a.out, fid, rows=rows, height=height)
        report[fid] = [c['stats'] for c in built]
        print(fid, ' | '.join(f"{c['title']}: {c['stats']['passes']}/{c['stats']['n']}" for c in built))
    path = os.path.join(a.out, 'median_figs_stats.json')
    if a.only and os.path.exists(path):
        old = json.load(open(path)); old.update(report); report = old
    json.dump(report, open(path, 'w'), indent=1)


if __name__ == '__main__':
    main()
