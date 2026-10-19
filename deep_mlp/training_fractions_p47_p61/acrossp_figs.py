#!/usr/bin/env python3
import argparse, glob, gzip, json, math, os, sys
from collections import deque
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig_acrossp")
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter, MaxNLocator

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import endpoints_i as E

K = dict(train="#0072B2", train_acc="#000000", test="#D55E00", refit="#CC79A7", AH="#009E73", window="#EBEBEB", grid="#E6E6E6",
         layer="#444444", agop="#E69F00", null="#888888")
plt.rcParams.update(matplotlib.rcParamsDefault)
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 8, 'axes.titlesize': 8, 'axes.labelsize': 8, 'xtick.labelsize': 8,
                     'ytick.labelsize': 8, 'legend.fontsize': 8, 'axes.spines.top': False, 'axes.spines.right': False,
                     'axes.edgecolor': '#555555', 'axes.linewidth': 0.7})
KFMT = FuncFormatter(lambda v, _: f"{v / 1000:g}")
FIRST, LAST = '-.', '-'
K = dict(K, refit="#7B3294")
GAIN = 0.2


def series(obs, key):
    s = [(o['step'], o[key]) for o in obs if key in o and o[key] is not None]
    return np.array([x[0] for x in s], float), np.array([x[1] for x in s], float)


def run_series(d):
    c = d['cfg']; p = c['p']; D = c['depth']; L = np.asarray(d['loss'], float); obs = d['obs']
    s, te = series(obs, 'test_acc'); _, tr = series(obs, 'train_acc')
    sr, rf = series(obs, 'v1_ridge_acc_te'); refit_kind = 'canonical'
    if not len(sr): sr, rf = series(obs, 'refit_acc'); refit_kind = 'legacy'
    sa, ahL = series(obs, f'AH_full_h{D}'); sa1, ah1 = series(obs, 'AH_full_h1')
    f = E.fields(d); ntr = int(c.get('frac', 0.8) * p * p)
    return dict(p=p, depth=D, L=L, test=(s, te), train=(s, tr), err=(s, (1 - tr) * ntr), refit=(sr, rf), ahL=(sa, ahL),
                ah1=(sa1, ah1) if D > 1 else None, refit_kind=refit_kind, agop={'first': None, 'last': None}, mse=None,
                W=f.get('W'), primary=bool(f.get('primary')), share=f.get('share'), level=f.get('CE_peak_over_logp'),
                why=f.get('why_not'), t90=f.get('t90') or f.get('t90_first'), ntr=ntr, knee=f.get('knee'))


def on_grid(x, y, g):
    x = np.asarray(x, float); y = np.asarray(y, float); ok = np.isfinite(y); x, y = x[ok], y[ok]
    return np.interp(g, x, y, left=y[0], right=y[-1])


def stretches(c, start, tol=1.03):
    n = len(c); amin = np.full(n, -1, int); dq = deque(); a = start
    for t in range(start, n):
        while dq and c[dq[-1]] <= c[t]: dq.pop()
        dq.append(t)
        while c[dq[0]] > tol * c[t]:
            a += 1
            while dq[0] < a: dq.popleft()
        amin[t] = a
    return amin


def new_window(c, knee):
    start = int(min(max(knee or 0, 0), len(c) - 1)); amin = stretches(c, start); n = len(c)
    S = [(int(amin[t]), t) for t in range(start, n) if t == n - 1 or amin[t + 1] != amin[t]]
    return max(S, key=lambda s: (float(np.max(c[s[0]:s[1] + 1]) - c[s[0]]), s[1] - s[0]))


def inside_gain(x, y, a, b):
    m = (np.asarray(x) >= a) & (np.asarray(x) <= b)
    return 0.0 if m.sum() < 2 else float(np.asarray(y)[m][-1] - np.asarray(y)[m][0])


def window_record(d):
    c = d['cfg']; lp = math.log(c['p']); L = np.asarray(d['loss'], float) / lp; cm = (L[:-1] + L[1:]) / 2
    f = E.fields(d); t, te = series(d['obs'], 'test_acc'); tf, rf = series(d['obs'], 'v1_ridge_acc_te')
    if not len(tf): tf, rf = series(d['obs'], 'refit_acc')
    a, b = new_window(cm, f.get('knee'))
    g_te, g_rf = inside_gain(t, te, a, b), inside_gain(tf, rf, a, b)
    ts, tr = series(d['obs'], 'train_acc'); m = (ts >= a) & (ts <= b); ntr = int(c.get('frac', 0.8) * c['p'] ** 2)
    return dict(W2=[a, b], ok2=bool(g_te >= GAIN or g_rf >= GAIN), g_te=g_te, g_rf=g_rf, knee=f.get('knee'),
                mis_in_W=int(round(float(np.max(1 - tr[m])) * ntr)) if m.any() else None)


class Store:
    def __init__(self, data):
        self.path = {}
        for kind in ('C', 'D', 'F', 'G'):
            for d in sorted(glob.glob(os.path.join(data, kind, '*'))):
                rid = os.path.basename(d)
                f = os.path.join(d, 'main.json.gz' if kind == 'C' else 'run.json.gz')
                if os.path.exists(f):
                    assert rid not in self.path, rid
                    self.path[rid] = (kind, f)

    def load(self, rid):
        with gzip.open(self.path[rid][1], 'rt') as f: return json.load(f)


def v2_verdicts(winv2):
    out = {}
    for f in sorted(glob.glob(os.path.join(winv2, '*', 'settings.jsonl'))):
        for line in open(f):
            for run in json.loads(line)['runs']:
                out[run['id']] = (run['W2'], run['ok2'], run['g_te'], run['g_rf'])
    return out


def deep_run(d, rid, verdict):
    r = run_series(d); L = r['L']; c = (L[:-1] + L[1:]) / 2 / math.log(r['p']); t_c = np.arange(len(c))
    W, ok, g_te, g_rf = verdict
    pk = W[0] + int(np.argmax(c[W[0]:W[1] + 1]))
    r.update(ce=(t_c, c), W=list(W), primary=bool(ok), g_te=g_te, g_rf=g_rf, peak=int(pk), level=float(c[pk]), id=rid)
    return r


def column2(runs, title, ids):
    t90 = [r['t90'] or r['W'][1] for r in runs]
    xmax = int(math.ceil((1.15 * max(np.median([r['W'][1] for r in runs]), np.percentile(t90, 90)) + 1500) / 1000) * 1000)
    xmax = min(xmax, int(max(r['ce'][0][-1] for r in runs)))
    g = np.arange(0, xmax + 1, max(10, xmax // 2000), dtype=float)
    on = on_grid
    S = dict(ce=np.array([on(*r['ce'], g) for r in runs]),
             test=np.array([on(*r['test'], g) for r in runs]), train=np.array([on(*r['train'], g) for r in runs]),
             err=np.array([on(*r['err'], g) for r in runs]), refit=np.array([on(*r['refit'], g) for r in runs]),
             ahL=np.array([on(*r['ahL'], g) for r in runs]),
             ah1=np.array([on(*r['ah1'], g) for r in runs]) if runs[0]['ah1'] is not None else None)
    band = (float(np.median([r['W'][0] for r in runs])), float(np.median([r['W'][1] for r in runs])))
    stats = dict(title=title, n=len(runs), passes=int(sum(r['primary'] for r in runs)), band_median=band,
                 peak_median=float(np.median([r['peak'] for r in runs])),
                 runs=[dict(id=r['id'], W=r['W'], primary=r['primary'], g_te=round(r['g_te'], 4), g_rf=round(r['g_rf'], 4),
                            peak=r['peak'], start=r.get('start'), start_rule=r.get('start_rule', 'knee'),
                            verdict=r.get('verdict', 'scored here')) for r in runs],
                 refit=sorted({r['refit_kind'] for r in runs}))
    return dict(g=g, S=S, band=band, xmax=xmax, title=title, stats=stats, ntr=max(r['ntr'] for r in runs))


def draw2(cols, out, stem, rows=('ce', 'acc', 'err', 'ah'), height=5.2, legend=True):
    n = len(cols)
    fig, axs = plt.subplots(len(rows), n, figsize=(6.75, height), sharey='row', layout='constrained')
    axs = np.atleast_2d(axs)
    ah_top = max(float(np.nanmax(c['S']['ahL'])) for c in cols) * 1.08
    err_top = max(c['ntr'] for c in cols) * 1.5
    for j, c in enumerate(cols):
        g, S = c['g'], c['S']
        for i, row in enumerate(rows):
            ax = axs[i, j]
            if c['band'][1] - c['band'][0] > 0.004 * c['xmax']: ax.axvspan(*c['band'], color=K['window'], lw=0, zorder=0)
            ax.grid(True, axis='y', color=K['grid'], lw=0.6); ax.set_xlim(0, c['xmax'])
            ax.xaxis.set_major_formatter(KFMT); ax.xaxis.set_major_locator(MaxNLocator(2 if n > 5 else 3, integer=True))
            thin = dict(lw=0.45, alpha=0.28, zorder=1)
            if row == 'ce':
                for y in S['ce']: ax.plot(g, y, color=K['train'], **thin)
                ax.plot(g, np.median(S['ce'], 0), color=K['train'], lw=1.4, zorder=3)
                ax.set_ylim(0, 1.05)
            elif row == 'acc':
                for y in S['test']: ax.plot(g, y, color=K['test'], **thin)
                ax.plot(g, np.median(S['train'], 0), color=K['train_acc'], lw=1.0, ls='--', zorder=3)
                ax.plot(g, np.median(S['refit'], 0), color=K['refit'], lw=1.4, zorder=4)
                ax.plot(g, np.median(S['test'], 0), color=K['test'], lw=1.4, zorder=5); ax.set_ylim(-0.02, 1.04)
            elif row == 'err':
                for y in S['err']: ax.plot(g, y, color=K['train_acc'], **thin)
                ax.plot(g, np.median(S['err'], 0), color=K['train_acc'], lw=1.1, zorder=3)
                ax.set_yscale('symlog', linthresh=1); ax.set_ylim(-0.2, err_top)
                ticks = [t for t in (0, 1, 10, 100, 1000, 10000) if t <= err_top]
                ax.set_yticks(ticks); ax.set_yticklabels([{1000: '1k', 10000: '10k'}.get(t, str(t)) for t in ticks])
            elif row == 'ah':
                for y in S['ahL']: ax.plot(g, y, color=K['AH'], **thin)
                ax.plot(g, np.median(S['ahL'], 0), color=K['AH'], lw=1.4, ls=LAST, zorder=3)
                if S['ah1'] is not None: ax.plot(g, np.median(S['ah1'], 0), color=K['AH'], lw=1.2, ls=FIRST, zorder=3)
                ax.set_ylim(0, ah_top)
            if i < len(rows) - 1: ax.tick_params(labelbottom=False)
        axs[0, j].set_title(c['title'], loc='center')
    lab = dict(ce='Training CE / log p\n(cycle mean)', acc='Accuracy', err='Misclassified\ntraining pairs', ah='Hidden Fourier\n$A_H$')
    if height < 4:
        lab.update(ce='Train CE\n/ log p', ah='Hidden\nFourier $A_H$')
    for i, row in enumerate(rows): axs[i, 0].set_ylabel(lab[row])
    for j in ((0, 1) if n == 2 else (n // 2,)): axs[-1, j].set_xlabel('Updates (thousands)')
    h = [Line2D([], [], color=K['train'], lw=1.4), Line2D([], [], color=K['train_acc'], lw=1.0, ls='--'),
         Line2D([], [], color=K['test'], lw=1.4), Line2D([], [], color=K['refit'], lw=1.4)]
    l = ['Train CE', 'Train acc.', 'Held-out', 'Refit']
    if 'err' in rows: h.append(Line2D([], [], color=K['train_acc'], lw=1.1)); l.append('Misclassified')
    if 'ah' in rows and any(c['S']['ah1'] is not None for c in cols):
        h += [Line2D([], [], color=K['layer'], lw=1.4, ls=LAST), Line2D([], [], color=K['layer'], lw=1.2, ls=FIRST)]
        l += ['Last hidden', 'First hidden']
    elif 'ah' in rows: h.append(Line2D([], [], color=K['AH'], lw=1.4)); l.append('$A_H$')
    if legend: fig.legend(h, l, loc='outside lower center', ncol=len(h) if len(h) <= 7 else (len(h) + 1) // 2, frameon=False,
                          handlelength=1.8, columnspacing=0.9, fontsize=8)
    out.mkdir(parents=True, exist_ok=True)
    fig.canvas.draw()
    fig.savefig(out / f'{stem}.pdf', metadata={'Title': stem, 'Author': '', 'Creator': 'acrossp_figs.py', 'CreationDate': None})
    fig.savefig(out / f'{stem}.png', dpi=200); plt.close(fig)


def specs(last):
    H = lambda p, lw: [f'H_p{p}_lw{lw:.4f}_s{i}' for i in range(1, 11)]
    Dd = lambda D, p, r: [f'D{D}_p{p}_r{r:.2f}_s{i}' for i in range(1, 11)]
    C2 = lambda D: (f'C1_D{D}_s', f'C2_D{D}_s')
    q = last
    return {
        'fig2_acrossp_1hl': ([('p = 23', H(23, 0.01)), ('p = 31', H(31, 0.015)), ('p = 47', H(47, 0.02)), ('p = 61', H(61, 0.02)),
                              (f'p = {q}', H(q, 0.02))], ('ce', 'acc', 'ah'), 2.6, {}),
        'acrossp_depth3_3row': ([('p = 23', Dd(3, 23, 0.45)), ('p = 31', C2(3)), ('p = 47', Dd(3, 47, 0.45)), ('p = 61', Dd(3, 61, 0.45)),
                                 (f'p = {q}', Dd(3, q, 0.45))], ('ce', 'acc', 'ah'), 2.3, dict(legend=False)),
        'acrossp_depth4_3row': ([('p = 23', Dd(4, 23, 0.32)), ('p = 31', C2(4)), ('p = 47', Dd(4, 47, 0.45)), ('p = 61', Dd(4, 61, 0.45)),
                                 (f'p = {q}', Dd(4, q, 0.45))], ('ce', 'acc', 'ah'), 2.6, {}),
    }


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--data', required=True); ap.add_argument('--winv2', required=True)
    ap.add_argument('--out', type=Path, required=True); ap.add_argument('--last', type=int, choices=(97, 101), required=True)
    ap.add_argument('--check-windows', action='store_true')
    a = ap.parse_args()
    store = Store(a.data); V = v2_verdicts(a.winv2); cache, report, scored = {}, {}, {}
    if a.check_windows:
        bad, n = [], 0
        for rid, (W, ok, gte, grf) in sorted(V.items()):
            if rid not in store.path or store.path[rid][0] not in ('D', 'F'): continue
            w = window_record(store.load(rid)); n += 1
            if w['W2'] != list(W) or w['ok2'] != ok or abs(round(w['g_te'], 4) - gte) > 1e-9 or abs(round(w['g_rf'], 4) - grf) > 1e-9: bad.append((rid, w, W, ok, gte, grf))
        print(f'window check: {n - len(bad)}/{n} runs of campaigns D and F equal the committed W2, ok2, g_te and g_rf (4 decimals)'); [print('  DIFF', b) for b in bad[:5]]

    def load(rid):
        if rid not in cache:
            d = store.load(rid)
            if rid in V: cache[rid] = deep_run(d, rid, V[rid]); cache[rid]['verdict'] = 'v2 settings.jsonl'
            else:
                w = window_record(d); scored[rid] = w
                cache[rid] = deep_run(d, rid, (w['W2'], w['ok2'], w['g_te'], w['g_rf'])); cache[rid]['verdict'] = 'scored here'
        return cache[rid]
    for fid, (cols_spec, rows, height, kw) in specs(a.last).items():
        cols = []
        for title, ids in cols_spec:
            if isinstance(ids, tuple): ids = sorted(k for k in store.path if store.path[k][0] == 'C' and k.startswith(ids))
            ids = [i for i in ids if i in store.path]
            cols.append(column2([load(i) for i in ids], title, ids))
        draw2(cols, a.out, fid, rows=rows, height=height, **kw); report[fid] = [c['stats'] for c in cols]
        print(fid, [(c['title'], c['stats']['passes'], c['stats']['n'], [round(x) for x in c['stats']['band_median']]) for c in cols])
    report['_windows_scored_here'] = scored
    (a.out / 'acrossp_stats.json').write_text(json.dumps(report, indent=1, default=float) + '\n')


if __name__ == '__main__':
    main()
