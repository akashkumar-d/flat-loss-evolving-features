#!/usr/bin/env python3
import argparse, gzip, json, math, os, sys, tarfile
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig_acrossp")
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter, MaxNLocator

K = dict(train="#0072B2", train_acc="#000000", test="#D55E00", refit="#7B3294", AH="#009E73", window="#EBEBEB", grid="#E6E6E6")
plt.rcParams.update(matplotlib.rcParamsDefault)
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 8, 'axes.titlesize': 8, 'axes.labelsize': 8, 'xtick.labelsize': 8,
                     'ytick.labelsize': 8, 'legend.fontsize': 8, 'axes.spines.top': False, 'axes.spines.right': False,
                     'axes.edgecolor': '#555555', 'axes.linewidth': 0.7})
KFMT = FuncFormatter(lambda v, _: f"{v / 1000:g}")
ARMS = [('ev3e-4', r'$\eta_V=3\cdot10^{-4}$'), ('ev1e-3', r'$\eta_V=10^{-3}$')]
T_A, N_TRAIN = 1000, 5050


def series(obs, key):
    s = [(o['step'], o[key]) for o in obs if key in o and o[key] is not None]
    return np.array([x[0] for x in s], float), np.array([x[1] for x in s], float)


def on_grid(x, y, g):
    x = np.asarray(x, float); y = np.asarray(y, float); ok = np.isfinite(y); x, y = x[ok], y[ok]
    return np.interp(g, x, y, left=y[0], right=y[-1])


def first(s, v, thr):
    i = np.where(v >= thr)[0]
    return int(s[i[0]]) if len(i) else None


def read_runs(src):
    out = {}
    for e in sorted(os.listdir(src)):
        p = os.path.join(src, e)
        if os.path.isdir(p) and os.path.exists(os.path.join(p, 'run.json.gz')):
            with gzip.open(os.path.join(p, 'run.json.gz'), 'rt') as fh: out[e] = json.load(fh)
        elif e.endswith('.tar'):
            with tarfile.open(p) as t:
                for m in t.getmembers():
                    if m.isfile() and m.name.endswith('/run.json.gz'):
                        out[m.name.split('/')[-2]] = json.loads(gzip.decompress(t.extractfile(m).read()).decode())
    return out


def load_arm(allruns, arm):
    runs = []
    ids = sorted((k for k in allruns if k.startswith(f'T41_{arm}_s')), key=lambda k: int(k.rsplit('_s', 1)[1]))
    for rid in ids:
        d = allruns[rid]
        lp = math.log(d['cfg']['p']); L = np.asarray(d['loss'], float); acc = np.asarray(d['train_acc'], float)
        st, te = series(d['obs'], 'test_acc'); sr, rf = series(d['obs'], 'refit_thm_acc'); sa, ah = series(d['obs'], 'AH')
        sc, rc = series(d['obs'], 'refit_canon_acc')
        a, b = first(sr, rf, 0.1), first(sr, rf, 0.9)
        seg = L[T_A:]; inb = L[a:b + 1]
        runs.append(dict(id=rid, cfg=d['cfg'], L=L / lp, acc=acc, test=(st, te), refit=(sr, rf), ah=(sa, ah),
                         stats=dict(v_1k_T=float((seg.max() - seg.min()) / seg.min()), refit10=a, refit90=b,
                                    v_refit_rise=float((inb.max() - inb.min()) / inb.min()), CE_over_logp_mean=float(seg.mean() / lp),
                                    AH_at_refit10=float(np.interp(a, sa, ah)), AH_at_refit90=float(np.interp(b, sa, ah)), AH_T=float(ah[-1]),
                                    test_at_refit90=float(np.interp(b, st, te)), test_T=float(te[-1]), refit_T=float(rf[-1]),
                                    canon_T=float(rc[-1]), train_T=float(acc[-1]),
                                    mis_at_refit10=int(round((1 - acc[a]) * N_TRAIN)), mis_at_refit90=int(round((1 - acc[b]) * N_TRAIN)),
                                    mis_T=int(round((1 - acc[-1]) * N_TRAIN)), refit50=first(sr, rf, 0.5), test50=first(st, te, 0.5),
                                    T=len(L) - 1)))
    return runs


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--out', type=Path, required=True); ap.add_argument('--runs', type=Path, default=None)
    a = ap.parse_args(); out = a.out.resolve(); out.mkdir(parents=True, exist_ok=True)
    if a.runs is None:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        import _paths as P
        a.runs = P.CAMPAIGN['H']
    allruns = read_runs(str(a.runs))
    data = {arm: load_arm(allruns, arm) for arm, _ in ARMS}
    T = max(r['stats']['T'] for rs in data.values() for r in rs)
    g = np.arange(0, T + 1, 20, dtype=float)
    fig, axs = plt.subplots(2, 3, figsize=(6.75, 3.3), sharex=True, layout='constrained')
    thin = dict(lw=0.45, alpha=0.28, zorder=1)
    ce_lo, ce_hi = 0.983, 1.003
    stats = {}
    for i, (arm, label) in enumerate(ARMS):
        runs = data[arm]
        band = (float(np.median([r['stats']['refit10'] for r in runs])), float(np.median([r['stats']['refit90'] for r in runs])))
        stats[arm] = dict(n=len(runs), band_median=band, runs={r['id']: r['stats'] for r in runs})
        for j in range(3):
            ax = axs[i, j]; ax.axvspan(*band, color=K['window'], lw=0, zorder=0); ax.grid(True, axis='y', color=K['grid'], lw=0.6)
            ax.set_xlim(0, T); ax.xaxis.set_major_formatter(KFMT); ax.xaxis.set_major_locator(MaxNLocator(4, integer=True))
        ax = axs[i, 0]
        C = np.array([on_grid(np.arange(len(r['L'])), r['L'], g) for r in runs])
        for y in C: ax.plot(g, y, color=K['train'], **thin)
        ax.plot(g, np.median(C, 0), color=K['train'], lw=1.4, zorder=3); ax.set_ylim(ce_lo, ce_hi)
        ax.set_yticks([0.985, 0.99, 0.995, 1.0]); ax.set_yticklabels(['0.985', '0.99', '0.995', '1'])
        ax = axs[i, 1]
        Te = np.array([on_grid(*r['test'], g) for r in runs])
        for y in Te: ax.plot(g, y, color=K['test'], **thin)
        ax.plot(g, np.median(np.array([on_grid(np.arange(len(r['acc'])), r['acc'], g) for r in runs]), 0), color=K['train_acc'], lw=1.0,
                ls='--', zorder=3)
        ax.plot(g, np.median(np.array([on_grid(*r['refit'], g) for r in runs]), 0), color=K['refit'], lw=1.4, zorder=4)
        ax.plot(g, np.median(Te, 0), color=K['test'], lw=1.4, zorder=5); ax.set_ylim(-0.02, 1.04); ax.set_yticks([0, 0.5, 1])
        ax.set_yticklabels(['0', '0.5', '1'])
        ax = axs[i, 2]
        A = np.array([on_grid(*r['ah'], g) for r in runs])
        for y in A: ax.plot(g, y, color=K['AH'], **thin)
        ax.plot(g, np.median(A, 0), color=K['AH'], lw=1.4, zorder=3); ax.set_ylim(0, 0.075)
        axs[i, 0].set_ylabel(f'{label}\n({len(runs)} runs)')
        if i == 0:
            for j, t in enumerate(('Training CE / log p', 'Accuracy', r'Hidden Fourier $A_H$')): axs[0, j].set_title(t, loc='center')
        else:
            axs[1, 1].set_xlabel('Updates (thousands)')
    h = [Line2D([], [], color=K['train'], lw=1.4), Line2D([], [], color=K['train_acc'], lw=1.0, ls='--'),
         Line2D([], [], color=K['test'], lw=1.4), Line2D([], [], color=K['refit'], lw=1.4), Line2D([], [], color=K['AH'], lw=1.4)]
    fig.legend(h, ['Train CE', 'Train acc.', 'Held-out', 'Refit', '$A_H$'], loc='outside lower center', ncol=5, frameon=False,
               handlelength=1.8, columnspacing=0.9, fontsize=8)
    fig.canvas.draw()
    fig.savefig(out / 'thm41_config_p101.pdf', metadata={'Title': 'thm41_config_p101', 'Author': '', 'Creator': 'thm41_config_fig.py',
                                                          'CreationDate': None})
    fig.savefig(out / 'thm41_config_p101.png', dpi=200); plt.close(fig)
    (out / 'thm41_config_stats.json').write_text(json.dumps(stats, indent=1) + '\n')
    for arm in stats:
        rs = list(stats[arm]['runs'].values()); q = lambda k: (min(r[k] for r in rs), max(r[k] for r in rs))
        print(arm, 'band', stats[arm]['band_median'], {k: tuple(round(x, 4) for x in q(k)) for k in
              ('v_1k_T', 'v_refit_rise', 'CE_over_logp_mean', 'AH_at_refit10', 'AH_at_refit90', 'AH_T', 'test_at_refit90', 'test_T',
               'refit_T', 'canon_T', 'train_T', 'mis_at_refit10', 'mis_at_refit90', 'mis_T', 'refit90')})


if __name__ == '__main__':
    main()
