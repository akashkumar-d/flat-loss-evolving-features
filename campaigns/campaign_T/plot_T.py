#!/usr/bin/env python3
import glob, gzip, json, math, os, sys
import numpy as np
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter

K = dict(train="#0072B2", train_acc="#000000", test="#D55E00", refit="#7B3294", AH="#009E73", window="#EBEBEB", grid="#E6E6E6")
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 8, 'axes.titlesize': 8, 'axes.labelsize': 8, 'xtick.labelsize': 7,
                     'ytick.labelsize': 7, 'legend.fontsize': 8, 'axes.spines.top': False, 'axes.spines.right': False,
                     'axes.edgecolor': '#555555', 'axes.linewidth': 0.7})
KFMT = FuncFormatter(lambda v, _: f"{v / 1000:g}")
COLS = [('T_A1_f0.3_adamw', 'AdamW, 1 layer'),
        ('T_P1_f0.3_lam_w0.01_eta_w0.1_lam_v1', 'Polar, 1 layer\n$\\lambda_W$ 0.01, $\\eta_W$ 0.1, $\\lambda_V$ 1'),
        ('T_P1_f0.3_lam_w0.02_eta_w0.1_lam_v0.2', 'Polar, 1 layer\n$\\lambda_W$ 0.02, $\\eta_W$ 0.1, $\\lambda_V$ 0.2'),
        ('T_P2_f0.3_lam_w0.01_eta_w0.1_lam_v1', 'Polar, 2 layers\n$\\lambda_W$ 0.01, $\\eta_W$ 0.1, $\\lambda_V$ 1')]
XMAX = {'T_A1_f0.3_adamw': 12000, 'T_P1_f0.3_lam_w0.01_eta_w0.1_lam_v1': 14000, 'T_P1_f0.3_lam_w0.02_eta_w0.1_lam_v0.2': 8000,
        'T_P2_f0.3_lam_w0.01_eta_w0.1_lam_v1': 30000}


def series(obs, key):
    s = [(o['step'], o[key]) for o in obs if o.get(key) is not None]
    return np.array([x[0] for x in s], float), np.array([x[1] for x in s], float)


def first(s, v, thr):
    i = np.where(v >= thr)[0]; return int(s[i[0]]) if len(i) else None


def load(res, rid):
    with gzip.open(os.path.join(res, rid, 'run.json.gz'), 'rt') as f: return json.load(f)


def smooth_ce(L, k=100):
    lg = np.log(np.maximum(np.asarray(L, float), 1e-12)); c = np.convolve(lg, np.ones(k) / k, mode='valid')
    return np.arange(len(c)) + k // 2, np.exp(c)


def dynamics(res, out):
    fig, axs = plt.subplots(3, len(COLS), figsize=(6.75, 4.6), layout='constrained', sharey='row')
    for j, (st, title) in enumerate(COLS):
        for i, alpha in ((1, 1.0), (2, 0.35)):
            rid = f'{st}_s{i}'
            if not os.path.exists(os.path.join(res, rid, 'run.json.gz')): continue
            d = load(res, rid); L = d['loss']
            s, te = series(d['obs'], 'test_acc'); _, tr = series(d['obs'], 'train_acc'); sr, rf = series(d['obs'], 'refit_test_acc')
            c = d.get('cfg_full', d['cfg']); sa, ah = series(d['obs'], f"AH_full_h{c['n_layers']}")
            if i == 1:
                r10, r90 = first(sr, rf, 0.1), first(sr, rf, 0.9)
                if r10 is not None and r90 is not None:
                    for r in range(3): axs[r, j].axvspan(r10, r90, color=K['window'], lw=0, zorder=0)
            x, y = smooth_ce(L); axs[0, j].semilogy(x, y, color=K['train'], lw=1.1, alpha=alpha)
            axs[1, j].plot(s, tr, color=K['train_acc'], lw=0.8, ls='--', alpha=alpha)
            axs[1, j].plot(s, te, color=K['test'], lw=1.1, alpha=alpha); axs[1, j].plot(sr, rf, color=K['refit'], lw=1.1, alpha=alpha)
            axs[2, j].plot(sa, ah, color=K['AH'], lw=1.1, alpha=alpha)
        axs[0, j].set_title(title)
        for r in range(3):
            ax = axs[r, j]; ax.set_xlim(0, XMAX[st]); ax.xaxis.set_major_formatter(KFMT); ax.grid(True, axis='y', color=K['grid'], lw=0.6)
            if r < 2: ax.tick_params(labelbottom=False)
    axs[0, 0].set_ylim(1e-8, 10); axs[1, 0].set_ylim(-0.02, 1.04); axs[2, 0].set_ylim(0, 1.0)
    axs[0, 0].set_ylabel('Train CE\n(log scale)'); axs[1, 0].set_ylabel('Accuracy'); axs[2, 0].set_ylabel('Last MLP\nFourier $A_H$')
    for j in range(len(COLS)): axs[2, j].set_xlabel('Updates (thousands)')
    h = [Line2D([], [], color=K['train'], lw=1.1), Line2D([], [], color=K['train_acc'], lw=0.8, ls='--'), Line2D([], [], color=K['test'], lw=1.1),
         Line2D([], [], color=K['refit'], lw=1.1), Line2D([], [], color=K['AH'], lw=1.1)]
    fig.legend(h, ['Train CE', 'Train acc.', 'Held-out', 'Refit', '$A_H$'], loc='outside lower center', ncol=5, frameon=False, fontsize=8)
    fig.savefig(os.path.join(out, 'T_dynamics.pdf'), metadata={'Creator': 'plot_T.py', 'CreationDate': None}); fig.savefig(os.path.join(out, 'T_dynamics.png'), dpi=200)
    plt.close(fig)


def scatter(res, out):
    pts = []
    for fn in sorted(glob.glob(os.path.join(res, '*', 'run.json.gz'))):
        rid = os.path.basename(os.path.dirname(fn)); d = load(res, rid); L = np.maximum(np.asarray(d['loss'], float), 1e-12); lg = np.log(L)
        gm = lambda t, h=250: float(np.exp(lg[max(0, t - h):min(len(L), t + h + 1)].mean()))
        s, te = series(d['obs'], 'test_acc'); sr, rf = series(d['obs'], 'refit_test_acc')
        r10, r90 = first(sr, rf, 0.1), first(sr, rf, 0.9)
        if r10 is None or r90 is None or r90 <= r10: continue
        c = d.get('cfg_full', d['cfg'])
        pts.append(dict(id=rid, opt=c['optimizer'], layers=c['n_layers'], ce0=gm(r10), ratio=gm(r90) / gm(r10), final=float(te[-1])))
    fig, ax = plt.subplots(figsize=(4.6, 3.3), layout='constrained')
    cm = plt.get_cmap('viridis')
    for q in pts:
        mk = 'o' if q['opt'] == 'polar' else 's'; ec = 'black' if q['layers'] == 2 else 'none'
        ax.scatter(q['ratio'], q['ce0'], s=26, marker=mk, color=cm(min(max((q['final'] - 0.9) / 0.1, 0), 1)), edgecolors=ec, linewidths=0.8, zorder=3)
    ax.set_xscale('log'); ax.set_yscale('log'); ax.axvline(1, color='#888888', lw=0.8, ls='--')
    ax.axhspan(1e-8, 1e-6, color=K['window'], lw=0, zorder=0); ax.text(1.5e-3, 2e-7, 'CE at the float32 floor', fontsize=7, color='#555555')
    ax.set_xlabel('Train CE at the end / start of the refit rise'); ax.set_ylabel('Train CE when the refit starts to rise')
    ax.set_xlim(1e-3, 3); ax.grid(True, color=K['grid'], lw=0.5)
    sm = plt.cm.ScalarMappable(cmap=cm, norm=plt.Normalize(0.9, 1.0)); cb = fig.colorbar(sm, ax=ax, pad=0.02); cb.set_label('Final held-out accuracy', fontsize=7)
    h = [Line2D([], [], marker='o', ls='', color='#777777', label='Polar, 1 layer'), Line2D([], [], marker='o', ls='', color='#777777', markeredgecolor='black', label='Polar, 2 layers'),
         Line2D([], [], marker='s', ls='', color='#777777', label='AdamW')]
    ax.legend(handles=h, loc='lower left', fontsize=7, frameon=False)
    fig.savefig(os.path.join(out, 'T_ce_during_refit.pdf'), metadata={'Creator': 'plot_T.py', 'CreationDate': None}); fig.savefig(os.path.join(out, 'T_ce_during_refit.png'), dpi=200)
    plt.close(fig); json.dump(pts, open(os.path.join(out, 'T_ce_during_refit.json'), 'w'), indent=1)
    return pts


if __name__ == '__main__':
    res, out = sys.argv[1], sys.argv[2]; os.makedirs(out, exist_ok=True)
    dynamics(res, out); pts = scatter(res, out)
    flat = [q for q in pts if q['ratio'] >= 0.8]
    print(len(pts), 'runs with a refit rise;', len(flat), 'with CE flat or rising (end/start >= 0.8):', [(q['id'], round(q['ratio'], 2), f"{q['ce0']:.1e}") for q in flat])
