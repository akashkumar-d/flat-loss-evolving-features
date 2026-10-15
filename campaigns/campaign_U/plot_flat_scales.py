#!/usr/bin/env python3
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.ticker
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import flat_windows as F

K = dict(train="#0072B2", train_acc="#000000", test="#D55E00", refit="#7B3294", band="#EBEBEB", grid="#E6E6E6")
BLUES = {100: '#9ECAE1', 1000: '#4292C6', 5000: '#08306B'}
WS = (1, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000)
DEFAULT = ('T_A1_f0.3_adamw_s1', 'T_P1_f0.3_lam_w0.02_eta_w0.1_lam_v0.2_s1', 'T_P2_f0.3_lam_w0.01_eta_w0.1_lam_v1_s1',
           'T_P1_f0.3_lam_w0.01_eta_w0.1_lam_v1_s1')


def title(rid, c):
    rms = ', RMS' if c.get('final_norm') == 'rms' else ''
    if c.get('optimizer') == 'adamw':
        return f"AdamW, {c['n_layers']} layer{'s' if c['n_layers'] > 1 else ''}{rms}\nwd = {c['wd']:g}, lr = {c['lr']:g}"
    return (f"Polar rule, {c['n_layers']} layer{'s' if c['n_layers'] > 1 else ''}{rms}\n"
            f"$\\lambda_W$ = {c['lam_w']:g}, $\\eta_W$ = {c['eta_w']:g}, $\\lambda_V$ = {c['lam_v']:g}")


def main(res, out, ids, stem='T_flat_scales'):
    plt.rcParams.update({'font.size': 7.5, 'axes.titlesize': 7.5, 'axes.labelsize': 7.5, 'legend.fontsize': 7,
                         'axes.spines.top': False, 'axes.spines.right': False})
    fig, axs = plt.subplots(3, len(ids), figsize=(max(1.95 * len(ids) + 0.4, 6.2), 5.9), layout='constrained',
                            gridspec_kw=dict(height_ratios=(1.25, 1, 1)), squeeze=False)
    stats = {}
    for j, rid in enumerate(ids):
        d = F.load(os.path.join(res, rid, 'run.json.gz')); c = d.get('cfg_full', d['cfg']); L = np.array(d['loss'], float)
        t = np.arange(len(L)); prof = F.profile(d, L, {(k, w): F.smooth(L, w, k) for k in F.KINDS for w in F.WINDOWS})
        a, b = prof['I']; xmax = min(len(L) - 1, int(b + max(4000, 0.6 * (b - a))))
        ax = axs[0, j]; ax.axvspan(a, b, color=K['band'], lw=0, zorder=0)
        ax.plot(t[:xmax:5], L[:xmax:5], color='#BBBBBB', lw=0.3, zorder=1)
        for w, col in BLUES.items(): ax.plot(t[:xmax], F.smooth(L, w, 'geo')[:xmax], color=col, lw=1.0 if w < 5000 else 1.4, zorder=2)
        ax.plot(t[:xmax], F.smooth(L, 5000, 'arith')[:xmax], color=BLUES[5000], lw=1.0, ls='--', zorder=2)
        ax.set_yscale('log'); ax.set_xlim(0, xmax); ax.set_title(title(rid, c)); ax.grid(color=K['grid'], lw=0.4)
        ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(4))
        lo = max(np.percentile(L[:xmax], 1), 1e-9); ax.set_ylim(lo / 3, 10)
        if j == 0: ax.set_ylabel('training CE')
        ax = axs[1, j]; ax.axvspan(a, b, color=K['band'], lw=0, zorder=0)
        s, te = F.series(d['obs'], 'test_acc'); st, tr = F.series(d['obs'], 'train_acc'); sr, rf = F.series(d['obs'], 'refit_test_acc')
        ax.plot(st, tr, color=K['train_acc'], lw=0.9, ls='--'); ax.plot(s, te, color=K['test'], lw=1.1); ax.plot(sr, rf, color=K['refit'], lw=1.1)
        sp, pv = F.series(d['obs'], 'probe_test_acc')
        if len(sp): ax.plot(sp, pv, color=K['refit'], lw=1.0, ls=':')
        ax.set_xlim(0, xmax); ax.set_ylim(-0.03, 1.03); ax.grid(color=K['grid'], lw=0.4); ax.set_xlabel('update')
        ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(4))
        if j == 0: ax.set_ylabel('accuracy')
        ax = axs[2, j]; rec = {}
        for kind, ls in (('geo', '-'), ('arith', '--')):
            dd = []
            for w in WS:
                seg = F.smooth(L, w, kind)[a:b + 1]; dd.append(float(np.max(1 - seg / np.maximum.accumulate(seg))))
            ax.plot(WS, dd, color='#08306B', lw=1.1, ls=ls, marker='o', ms=2.2); rec[kind] = dict(zip(map(str, WS), dd))
        ax.axhline(0.1, color='#999999', lw=0.6, ls=':'); ax.axhline(0.3, color='#999999', lw=0.6, ls=':')
        ax.axvline(b - a, color='#444444', lw=0.8, ls=':'); ax.text(b - a, 1.02, ' |I|', fontsize=6.5, va='bottom', ha='left', color='#444444')
        ax.set_xscale('log'); ax.set_ylim(-0.02, 1.08); ax.set_xlabel('averaging window (updates)'); ax.grid(color=K['grid'], lw=0.4)
        if j == 0: ax.set_ylabel('drawdown of the average over I')
        stats[rid] = dict(I=[a, b], level_geo500=prof['geo']['500']['level'], drawdown=rec)
    h = [Line2D([], [], color='#BBBBBB', lw=0.8)] + [Line2D([], [], color=c_, lw=1.2) for c_ in BLUES.values()] + \
        [Line2D([], [], color=BLUES[5000], lw=1.0, ls='--'), Line2D([], [], color=K['train_acc'], lw=0.9, ls='--'),
         Line2D([], [], color=K['test'], lw=1.1), Line2D([], [], color=K['refit'], lw=1.1)]
    h.append(Line2D([], [], color=K['refit'], lw=1.0, ls=':'))
    fig.legend(h, ['CE (raw)', 'geometric avg. 100', 'geometric avg. 1000', 'geometric avg. 5000', 'arithmetic avg. 5000',
                   'training acc.', 'held-out acc.', 'ridge refit acc.', 'logistic probe acc.'], loc='outside lower center',
               ncol=5 if len(ids) >= 4 else 3, frameon=False)
    os.makedirs(out, exist_ok=True)
    fig.savefig(os.path.join(out, stem + '.pdf'), metadata={'Creator': 'plot_flat_scales.py', 'CreationDate': None})
    fig.savefig(os.path.join(out, stem + '.png'), dpi=200); plt.close(fig)
    import json; json.dump(stats, open(os.path.join(out, stem + '.json'), 'w'), indent=1)
    for k, v in stats.items(): print(k, v['I'], f"{v['level_geo500']:.2e}", {kk: round(vv, 2) for kk, vv in v['drawdown']['geo'].items()})


if __name__ == '__main__':
    a = sys.argv[1:]; stem = 'T_flat_scales'
    if '--stem' in a: i = a.index('--stem'); stem = a[i + 1]; del a[i:i + 2]
    main(a[0], a[1], tuple(a[2:]) or DEFAULT, stem)
