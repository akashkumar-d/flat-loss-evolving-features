#!/usr/bin/env python3
import json, math, os, re, statistics as S, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.ticker
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import flat_windows as F
import confirm_V as CV

K = dict(train_acc="#000000", test="#D55E00", refit="#7B3294", grid="#E6E6E6", ok="#009E73", nok="#BDBDBD")
BLUES = {100: '#9ECAE1', 1000: '#4292C6', 5000: '#08306B'}
GROUPS = [
    ('Confirmation (fresh seeds 6-15, frac 0.3)', [
        ('V_C1_f0.3_lam_w0.02_eta_w0.1_lam_v0.2', 'C1: 1 layer, $\\lambda_W$ 0.02, $\\lambda_V$ 0.2'),
        ('V_C2_f0.3_lam_w0.01_eta_w0.1_lam_v1', 'C2: 2 layers, $\\lambda_W$ 0.01, $\\lambda_V$ 1')]),
    ('Final RMS division, frac 0.2 (seeds 1-3)', [
        ('V_E1_f0.2_rms_lam_w0.01_eta_w0.03_lam_v0.5', '1 layer, $\\lambda_V$ 0.5, $\\eta_W$ 0.03'),
        ('V_E1_f0.2_rms_lam_w0.01_eta_w0.1_lam_v0.5', '1 layer, $\\lambda_V$ 0.5, $\\eta_W$ 0.1'),
        ('V_E1_f0.2_rms_lam_w0.01_eta_w0.03_lam_v0.75', '1 layer, $\\lambda_V$ 0.75, $\\eta_W$ 0.03'),
        ('V_E1_f0.2_rms_lam_w0.01_eta_w0.1_lam_v0.75', '1 layer, $\\lambda_V$ 0.75, $\\eta_W$ 0.1'),
        ('V_E1_f0.2_rms_lam_w0.01_eta_w0.03_lam_v1', '1 layer, $\\lambda_V$ 1, $\\eta_W$ 0.03'),
        ('V_E1_f0.2_rms_lam_w0.01_eta_w0.1_lam_v1', '1 layer, $\\lambda_V$ 1, $\\eta_W$ 0.1'),
        ('V_E2_f0.2_rms_lam_w0.01_eta_w0.1_lam_v0.5', '2 layers, $\\lambda_V$ 0.5, $\\eta_W$ 0.1'),
        ('V_E2_f0.2_rms_lam_w0.01_eta_w0.1_lam_v1', '2 layers, $\\lambda_V$ 1, $\\eta_W$ 0.1')]),
    ('AdamW (lr 1e-3, wd 1), frac 0.2 (seeds 1-3)', [
        ('V_A1_f0.2_adamw_wd1', '1 layer'), ('V_A1_f0.2_rms_adamw_wd1', '1 layer, RMS')])]
LABEL = {st: lab for _, items in GROUPS for st, lab in items}
PREFIX = {'C1': 'C1 (confirmation, frac 0.3): ', 'C2': 'C2 (confirmation, frac 0.3): ', 'E1': 'final RMS division, frac 0.2: ',
          'E2': 'final RMS division, frac 0.2: ', 'A1': 'AdamW (lr 1e-3, wd 1), frac 0.2: '}
def title_of(st):
    arm = st.split('_')[1]; lab = LABEL.get(st, st)
    return PREFIX[arm] + (lab.split(': ', 1)[1] if ': ' in lab else lab)
def plain(s): return s.replace('$\\lambda_W$', 'λW').replace('$\\lambda_V$', 'λV').replace('$\\eta_W$', 'ηW')


def setting(rid): return re.sub(r'_s\d+$', '', rid)


def style():
    plt.rcParams.update({'font.size': 7.5, 'axes.titlesize': 7.5, 'axes.labelsize': 7.5, 'legend.fontsize': 7,
                         'axes.spines.top': False, 'axes.spines.right': False})


def cell(ax, x, y, k, n, txt=None, wd=1.0):
    green = np.array(to_rgb(K['ok'])); white = np.ones(3); frac = k / n if n else 0
    col = white * (1 - 0.85 * frac) + green * 0.85 * frac if frac > 0 else np.array([0.955] * 3)
    ax.add_patch(plt.Rectangle((x + 0.04, y + 0.1), wd - 0.08, 0.84, color=col, lw=0))
    txt = txt or f'{k}/{n}'
    ax.text(x + wd / 2, y + 0.55, txt, ha='center', va='center', fontsize=6.2 if len(txt) <= 6 else 5.5,
            color='white' if frac > 0.6 else 'black')


def scoreboard(conf, runs, fw, out):
    style()
    by = {}
    for r in runs: by.setdefault(r['setting'], []).append(r)
    fwb = {}
    for r in fw['runs']: fwb.setdefault(setting(r['id']), []).append(r)
    W = fw['windows']; rows = []
    for g, items in GROUPS:
        rows.append(('group', g))
        for st, lab in items: rows.append(('row', st, lab))
    CONF = [('pass_', 'rule'), ('probe', 'probe'), ('loose', 'loose'), ('raw3', 'raw3')]
    TW, CW, WW = 1.45, 1.0, 1.3
    x_conf = 2 * TW + 0.2; x_geo = x_conf + len(CONF) * CW + 0.4; x_ar = x_geo + len(W) * WW + 0.25; ncol = x_ar + len(W) * WW
    fig, ax = plt.subplots(figsize=(9.4, 0.25 * len(rows) + 1.35), layout='constrained')
    ax.set_xlim(-5.4, ncol); ax.set_ylim(len(rows), -1.45); ax.axis('off')
    ax.text(TW / 2, -0.45, 'CE / log p\nin stretch', ha='center', va='bottom', fontsize=6.3)
    ax.text(TW * 1.5, -0.45, 'final\nheld-out', ha='center', va='bottom', fontsize=6.3)
    ax.text(x_conf + len(CONF) * CW / 2, -1.3, 'confirmation rule', ha='center', va='bottom', fontsize=7.0)
    for i, (_, lab) in enumerate(CONF): ax.text(x_conf + (i + 0.5) * CW, -0.45, lab, ha='center', va='bottom', fontsize=6.4)
    for x0, kind in ((x_geo, 'geometric'), (x_ar, 'arithmetic')):
        ax.text(x0 + len(W) * WW / 2, -1.3, f'key message, {kind} average over w', ha='center', va='bottom', fontsize=7.0)
        for i, w in enumerate(W): ax.text(x0 + (i + 0.5) * WW, -0.45, (f'{w // 1000}k' if w >= 1000 else f'{w}'), ha='center', va='bottom', fontsize=6.4)
    for i, row in enumerate(rows):
        if row[0] == 'group':
            ax.text(-5.35, i + 0.62, row[1], ha='left', va='center', fontsize=7.0, fontweight='bold'); continue
        _, st, lab = row; rs = by.get(st, []); n = len(rs)
        ax.text(-5.25, i + 0.55, f'{lab}  ({n})', ha='left', va='center', fontsize=6.5)
        lv = [r['stretch']['ce_over_logp'] for r in rs if r.get('stretch')]
        ax.text(TW / 2, i + 0.55, (f'{S.median(lv):.2g}' if lv else '-'), ha='center', va='center', fontsize=6.3)
        ax.text(TW * 1.5, i + 0.55, f"{S.median([r['final_test'] for r in rs]):.2f}", ha='center', va='center', fontsize=6.3)
        for j, (k, _) in enumerate(CONF): cell(ax, x_conf + j * CW, i, sum(r['verdict'][k] for r in rs), n, wd=CW)
        fr = fwb.get(st, [])
        for x0, kind in ((x_geo, 'geo'), (x_ar, 'arith')):
            for j, w in enumerate(W):
                ks = [r['key'][f'{kind}_{w}'] for r in fr]; p = sum(x['pass'] for x in ks); p3 = sum(x['pass3'] for x in ks)
                cell(ax, x0 + j * WW, i, p, len(ks), f'{p}/{len(ks)}' + (f' ({p3})' if p3 else ''), wd=WW)
    fig.savefig(os.path.join(out, 'V_scoreboard.pdf'), metadata={'Creator': 'plot_V.py', 'CreationDate': None})
    fig.savefig(os.path.join(out, 'V_scoreboard.png'), dpi=200); plt.close(fig)


def events(runs, out):
    style()
    order = [st for _, items in GROUPS for st, _ in items]
    rs = sorted(runs, key=lambda r: (order.index(r['setting']), r['seed']))
    fig, ax = plt.subplots(figsize=(7.6, 0.15 * len(rs) + 1.7), layout='constrained')
    M = [('t_fit97', 's', 'white', 'training acc. (avg. 500) 0.97'), ('t_fit', 's', K['train_acc'], 'training acc. (avg. 500) 0.99'),
         ('t50', 'o', '#F0A35E', 'held-out 0.5'), ('t90', 'o', K['test'], 'held-out 0.9'), ('t99', 'D', '#8C2D04', 'held-out 0.99'),
         ('r90', '^', K['refit'], 'ridge refit 0.9')]
    prev = None; yt = []; yl = []
    for i, r in enumerate(rs):
        if r['setting'] != prev and prev is not None: ax.axhline(i - 0.5, color='#CCCCCC', lw=0.6)
        if r['setting'] != prev: yt.append(i); yl.append(plain(title_of(r['setting'])))
        prev = r['setting']
        ax.plot([1e2, r['updates']], [i, i], color='#E0E0E0', lw=0.8, zorder=0)
        ax.text(1.06e2, i, f"s{r['seed']}", va='center', ha='left', fontsize=4.8, color='#888888')
        for key, mk, col, _ in M:
            t = r['events'][key]
            if t is None: continue
            if col == 'white': ax.plot(max(t, 100), i, marker=mk, ms=3.8, mfc='white', mec='black', mew=0.6, lw=0, zorder=3)
            else: ax.plot(max(t, 100), i, marker=mk, ms=3.6 if mk != 'D' else 3.0, color=col, lw=0, zorder=4, mec='white', mew=0.3)
        v = r['ce_over_logp']['t50']
        ax.text(1.15e5, i, (f"{v:.2g}" if v is not None else '-'), va='center', ha='left', fontsize=5.8, color='#333333')
        ax.text(3.3e5, i, 'pass' if r['verdict']['pass_'] else '', va='center', ha='left', fontsize=5.8, color=K['ok'])
    ax.set_xscale('log'); ax.set_xlim(1e2, 6e5); ax.set_ylim(len(rs) - 0.5, -0.8)
    ax.set_yticks(yt); ax.set_yticklabels(yl, fontsize=6.0); ax.tick_params(axis='y', length=0)
    ax.set_xticks([1e2, 1e3, 1e4, 1e5]); ax.set_xlabel('update (log scale; grey line: run length)')
    ax.text(1.15e5, -0.9, 'CE / log p at\nheld-out 0.5', fontsize=5.6, va='bottom'); ax.text(3.3e5, -0.9, 'rule', fontsize=5.6, va='bottom')
    ax.grid(axis='x', color=K['grid'], lw=0.4)
    h = [Line2D([], [], marker=mk, color=('black' if col == 'white' else col), mfc=col, lw=0, ms=4) for _, mk, col, _ in M]
    fig.legend(h, [lab for *_, lab in M], loc='outside lower center', ncol=3, frameon=False)
    fig.savefig(os.path.join(out, 'V_events.pdf'), metadata={'Creator': 'plot_V.py', 'CreationDate': None})
    fig.savefig(os.path.join(out, 'V_events.png'), dpi=200); plt.close(fig)


def dynamics(res, runs, out, st, ids, stem):
    style()
    ncol = min(5, len(ids)); blocks = math.ceil(len(ids) / ncol)
    fig, axs = plt.subplots(2 * blocks, ncol, figsize=(1.75 * ncol + 0.6, 3.15 * blocks + 0.5), layout='constrained',
                            gridspec_kw=dict(height_ratios=(1.15, 1) * blocks), squeeze=False)
    rr = {r['id']: r for r in runs}
    for q, rid in enumerate(ids):
        bi, j = divmod(q, ncol); a0, a1 = axs[2 * bi, j], axs[2 * bi + 1, j]
        d = F.load(os.path.join(res, rid, 'run.json.gz')); c = d.get('cfg_full', d['cfg']); L = np.array(d['loss'], float)
        t = np.arange(len(L)); r = rr[rid]; logp = math.log(c['p'])
        stv = r.get('stretch')
        for ax in (a0, a1):
            if stv: ax.axvspan(stv['a'], stv['b'], color=K['ok'] if r['verdict']['pass_'] else K['nok'], alpha=0.16 if r['verdict']['pass_'] else 0.35, lw=0, zorder=0)
            if r['events']['t_fit97'] is not None: ax.axvline(r['events']['t_fit97'], color='#000000', lw=0.7, ls=':', zorder=1)
            ax.set_xlim(0, len(L) - 1); ax.grid(color=K['grid'], lw=0.4); ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(3))
        a0.plot(t[::5], L[::5] / logp, color='#BBBBBB', lw=0.3, zorder=1)
        for w, col in BLUES.items(): a0.plot(t, F.smooth(L, w, 'geo') / logp, color=col, lw=1.0 if w < 5000 else 1.4, zorder=2)
        a0.set_yscale('log'); lo = max(np.percentile(L / logp, 1), 1e-9); a0.set_ylim(lo / 3, 1.5)
        a0.set_title(f"seed {r['seed']}" + ('  (pass)' if r['verdict']['pass_'] else ''), color=K['ok'] if r['verdict']['pass_'] else 'black')
        if j == 0: a0.set_ylabel('training CE / log p')
        s, te = F.series(d['obs'], 'test_acc'); sx, tr = F.series(d['obs'], 'train_acc'); sr, rf = F.series(d['obs'], 'refit_test_acc')
        sp, pv = F.series(d['obs'], 'probe_test_acc'); every = int(np.median(np.diff(sx)))
        a1.plot(sx, tr, color='#9E9E9E', lw=0.5); a1.plot(sx, CV.centred_mean(tr, max(1, 500 // every)), color=K['train_acc'], lw=0.9, ls='--')
        a1.plot(s, te, color=K['test'], lw=1.1); a1.plot(sr, rf, color=K['refit'], lw=1.1)
        if len(sp): a1.plot(sp, pv, color=K['refit'], lw=1.0, ls=':')
        a1.set_ylim(-0.03, 1.03); a1.set_xlabel('update')
        if j == 0: a1.set_ylabel('accuracy')
    for q in range(len(ids), blocks * ncol):
        bi, j = divmod(q, ncol); axs[2 * bi, j].axis('off'); axs[2 * bi + 1, j].axis('off')
    fig.suptitle(title_of(st), fontsize=8)
    h = [Line2D([], [], color='#BBBBBB', lw=0.8)] + [Line2D([], [], color=c_, lw=1.2) for c_ in BLUES.values()] + \
        [Line2D([], [], color='#9E9E9E', lw=0.6), Line2D([], [], color=K['train_acc'], lw=0.9, ls='--'), Line2D([], [], color=K['test'], lw=1.1),
         Line2D([], [], color=K['refit'], lw=1.1), Line2D([], [], color=K['refit'], lw=1.0, ls=':'),
         Patch(color=K['ok'], alpha=0.3), Patch(color=K['nok'], alpha=0.5), Line2D([], [], color='black', lw=0.7, ls=':')]
    fig.legend(h, ['CE (raw)', 'geo. avg. 100', 'geo. avg. 1000', 'geo. avg. 5000', 'training acc.', 'training acc., avg. 500',
                   'held-out acc.', 'ridge refit acc.', 'logistic probe acc.', 'passing stretch', 'best stretch (fails)',
                   'training acc. (avg. 500) 0.97'], loc='outside lower center', ncol=6 if ncol >= 5 else 3, frameon=False)
    fig.savefig(os.path.join(out, stem + '.pdf'), metadata={'Creator': 'plot_V.py', 'CreationDate': None})
    fig.savefig(os.path.join(out, stem + '.png'), dpi=170); plt.close(fig)


HIGH = [('V_E1_f0.2_rms_lam_w0.01_eta_w0.03_lam_v0.5_s1', 'Polar, RMS, 1 layer\n$\\lambda_V$ 0.5, $\\eta_W$ 0.03, seed 1 (passes)'),
        ('V_E1_f0.2_rms_lam_w0.01_eta_w0.1_lam_v0.5_s1', 'Polar, RMS, 1 layer\n$\\lambda_V$ 0.5, $\\eta_W$ 0.1, seed 1 (passes)'),
        ('V_E2_f0.2_rms_lam_w0.01_eta_w0.1_lam_v0.5_s2', 'Polar, RMS, 2 layers\n$\\lambda_V$ 0.5, $\\eta_W$ 0.1, seed 2 (fails)'),
        ('V_A1_f0.2_rms_adamw_wd1_s1', 'AdamW, RMS, 1 layer\nwd 1, seed 1 (fails)')]


def highloss(res, runs, out, xmax=(24000, 12000, 30000, 12000)):
    style(); rr = {r['id']: r for r in runs}
    fig, axs = plt.subplots(2, len(HIGH), figsize=(7.4, 3.9), layout='constrained', gridspec_kw=dict(height_ratios=(1, 1.1)), squeeze=False)
    for j, ((rid, lab), xm) in enumerate(zip(HIGH, xmax)):
        d = F.load(os.path.join(res, rid, 'run.json.gz')); c = d.get('cfg_full', d['cfg']); L = np.array(d['loss'], float)
        r = rr[rid]; logp = math.log(c['p']); t = np.arange(len(L)); xm = min(xm, len(L) - 1)
        a0, a1 = axs[0, j], axs[1, j]
        for ax in (a0, a1):
            stv = r.get('stretch')
            if stv and r['verdict']['pass_']: ax.axvspan(stv['a'], min(stv['b'], xm), color=K['ok'], alpha=0.16, lw=0, zorder=0)
            if r['events']['t_fit97'] is not None: ax.axvline(r['events']['t_fit97'], color='#000000', lw=0.7, ls=':', zorder=1)
            ax.set_xlim(0, xm); ax.grid(color=K['grid'], lw=0.4); ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(3))
        a0.plot(t[:xm:3], L[:xm:3] / logp, color='#BBBBBB', lw=0.3)
        a0.plot(t[:xm], F.smooth(L, 500, 'geo')[:xm] / logp, color=BLUES[5000], lw=1.3)
        a0.set_ylim(0, 1.05); a0.set_title(lab, fontsize=6.8)
        if j == 0: a0.set_ylabel('training CE / log p')
        s, te = F.series(d['obs'], 'test_acc'); sx, tr = F.series(d['obs'], 'train_acc'); sr, rf = F.series(d['obs'], 'refit_test_acc')
        sp, pv = F.series(d['obs'], 'probe_test_acc'); every = int(np.median(np.diff(sx)))
        a1.plot(sx, CV.centred_mean(tr, max(1, 500 // every)), color=K['train_acc'], lw=0.9, ls='--')
        a1.plot(s, te, color=K['test'], lw=1.0); a1.plot(sr, rf, color=K['refit'], lw=1.1)
        if len(sp): a1.plot(sp, pv, color=K['refit'], lw=1.0, ls=':')
        a1.set_ylim(-0.03, 1.03); a1.set_xlabel('update')
        if j == 0: a1.set_ylabel('accuracy')
    h = [Line2D([], [], color='#BBBBBB', lw=0.8), Line2D([], [], color=BLUES[5000], lw=1.3), Line2D([], [], color=K['train_acc'], lw=0.9, ls='--'),
         Line2D([], [], color=K['test'], lw=1.0), Line2D([], [], color=K['refit'], lw=1.1), Line2D([], [], color=K['refit'], lw=1.0, ls=':'),
         Patch(color=K['ok'], alpha=0.3), Line2D([], [], color='black', lw=0.7, ls=':')]
    fig.legend(h, ['CE (raw)', 'CE, geo. avg. 500', 'training acc. (avg. 500)', 'held-out acc.', 'ridge refit acc.', 'logistic probe acc.',
                   'passing stretch', 'training acc. reaches 0.97'], loc='outside lower center', ncol=4, frameon=False)
    fig.savefig(os.path.join(out, 'V_highloss.pdf'), metadata={'Creator': 'plot_V.py', 'CreationDate': None})
    fig.savefig(os.path.join(out, 'V_highloss.png'), dpi=200); plt.close(fig)


if __name__ == '__main__':
    res, an = sys.argv[1:3]
    conf = json.load(open(os.path.join(an, 'confirm_V.json')))['runs']
    runs = json.load(open(os.path.join(an, 'runs_V.json')))['runs']
    fw = json.load(open(os.path.join(an, 'key_message_by_window_V_d0.1.json')))
    scoreboard(conf, runs, fw, an); events(runs, an); highloss(res, runs, an)
    if '--no-dynamics' not in sys.argv:
        os.makedirs(os.path.join(an, 'settings'), exist_ok=True)
        by = {}
        for r in runs: by.setdefault(r['setting'], []).append(r)
        for st, rs in by.items():
            ids = [r['id'] for r in sorted(rs, key=lambda r: r['seed'])]
            dynamics(res, runs, os.path.join(an, 'settings'), st, ids, 'V_dyn_' + st)
