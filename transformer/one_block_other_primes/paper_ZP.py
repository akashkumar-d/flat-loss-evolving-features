#!/usr/bin/env python3
import json, math, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.environ.get('CAMPAIGN_Z_DIR', os.path.join(HERE, '..', 'gain_recipe_contrasts_two_blocks')))
import plot_Z as PZ
import plot_X as PX
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter, MaxNLocator, FixedLocator

plt.rcParams.update(plt.rcParamsDefault)
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 8, 'axes.titlesize': 8, 'axes.labelsize': 8, 'xtick.labelsize': 8,
                     'ytick.labelsize': 8, 'legend.fontsize': 8, 'axes.spines.top': False, 'axes.spines.right': False,
                     'axes.edgecolor': '#555555', 'axes.linewidth': 0.7, 'legend.frameon': False})
C_CE, C_TRAIN, C_NET, C_REFIT, C_FOUR, C_AH, C_BAND, C_GRID, C_REF = (
    '#0072B2', '#000000', '#D55E00', '#7B3294', '#56B4E9', '#009E73', '#EBEBEB', '#E6E6E6', '#888888')
DASH, DOT = (0, (4, 2)), (0, (1, 1.6))
P_STYLE = {61: ('#86b6ef', 'o'), 89: ('#3987e5', 's'), 113: ('#1c5cab', 'D'), 149: ('#0d366b', '^')}
KFMT = FuncFormatter(lambda v, _: f'{v / 1000:g}')
PRIMES = (61, 89, 113, 149)
SCATTER = ('P_p61_f0.3_lam_v0.45', 'P_p61_f0.33_lam_v0.45', 'P_p61_f0.35_lam_v0.45', 'P_p89_f0.25_lam_v0.33',
           'P_p89_f0.275_lam_v0.33', 'Y_G2', 'P_p149_f0.15_lam_v0.23', 'P_p149_f0.16_lam_v0.23')


def save(fig, outdir, stem):
    fig.savefig(os.path.join(outdir, stem + '.pdf'), metadata={'Title': stem, 'Author': '', 'Creator': 'paper_ZP.py', 'CreationDate': None})
    fig.savefig(os.path.join(outdir, stem + '.png'), dpi=200); plt.close(fig); print('drew', stem, flush=True)


def band(ax, x, Y, color, ls='-', lw=1.4, iqr=True, z=3):
    m = np.nanmedian(Y, 0)
    if iqr: ax.fill_between(x, np.nanpercentile(Y, 25, 0), np.nanpercentile(Y, 75, 0), color=color, alpha=0.18, lw=0, zorder=z - 1)
    ax.plot(x, m, color=color, ls=ls, lw=lw, zorder=z)
    return m


def mirror_acc(runs, grid):
    Ya, Yn = PX.on_grid(runs, 'test_acc', grid), PX.on_grid(runs, 'test_acc_nm', grid)
    nho = np.array([d['rec']['n_heldout'] for d in runs], float)[:, None]
    nnm = np.array([d['rec']['n_nm'] for d in runs], float)[:, None]
    nm_ = np.array([d['rec']['n_mirror'] for d in runs], float)[:, None]
    assert np.all(nnm + nm_ == nho)
    return Ya, Yn, (nho * Ya - nnm * Yn) / nm_


def at(s, v, t): return float(v[int(np.argmin(np.abs(s - t)))])


def counts(cell):
    R = cell['runs']; n = len(R)
    nm_only = sum(bool(r['nm'].get('window', {}).get('ok_strict')) and (r['nm']['final'] or 0) >= 0.99 for r in R)
    return n, cell['summary']['pass_strict'], cell['summary']['pass_loose'], nm_only


def column_title(p, cell):
    r = cell['runs'][0]; n, st, lo, nmo = counts(cell)
    extra = f'; no-mirror {nmo}/{n}' if nmo > st else (f'; loose fit {lo}/{n}' if lo > st else '')
    return f"p = {p}, fraction {r['frac']:g}\n{st}/{n} pass{extra}"


def acrossp(A, outdir):
    rows = ('ce', 'acc', 'err', 'ah')
    cols = [(p, A['cells'][A['appendix'][str(p)]['cell']]) for p in PRIMES]
    data = []
    for p, cell in cols:
        runs = PZ.load_runs(cell); T = min(len(d['loss']) for d in runs) - 1; g = np.arange(0, T + 1, 20)
        W = [r['nm']['window']['W'] for r in cell['runs'] if r['nm'].get('window', {}).get('W')]
        data.append(dict(p=p, cell=cell, runs=runs, T=T, g=g, W=(np.median([w[0] for w in W]), np.median([w[1] for w in W])),
                         ce=PZ.ce_grid(runs, g) / math.log(p)))
    ce_top = 1.15 * max(np.nanpercentile(d['ce'][:, d['g'] >= 400], 99) for d in data)
    ntr_top = 1.5 * max(d['cell']['runs'][0]['n_train'] for d in data)
    fig, axs = plt.subplots(len(rows), len(cols), figsize=(6.75, 5.6), sharey='row', layout='constrained', squeeze=False)
    for j, d in enumerate(data):
        g, runs = d['g'], d['runs']
        for i, row in enumerate(rows):
            ax = axs[i, j]
            ax.axvspan(d['W'][0], min(d['W'][1], d['T']), color=C_BAND, lw=0, zorder=0)
            ax.grid(True, axis='y', color=C_GRID, lw=0.6); ax.set_xlim(0, d['T'])
            ax.xaxis.set_major_formatter(KFMT); ax.xaxis.set_major_locator(MaxNLocator(3, integer=True))
            if row == 'ce':
                band(ax, g, d['ce'], C_CE); ax.set_ylim(0, ce_top)
            elif row == 'acc':
                band(ax, g, PX.on_grid(runs, 'train_acc', g), C_TRAIN, ls='--', lw=0.9, iqr=False, z=3)
                band(ax, g, PX.on_grid(runs, 'refit_test_acc', g), C_REFIT, lw=1.3, z=4)
                band(ax, g, PX.on_grid(runs, 'fref_top8_test_acc', g), C_FOUR, lw=1.3, z=4)
                band(ax, g, PX.on_grid(runs, 'test_acc_nm', g), C_NET, ls=DASH, lw=1.1, iqr=False, z=5)
                band(ax, g, PX.on_grid(runs, 'test_acc', g), C_NET, lw=1.6, z=6)
                ax.set_ylim(-0.02, 1.04)
            elif row == 'err':
                E = np.round((1 - PX.on_grid(runs, 'train_acc', g)) * np.array([x['rec']['n_train'] for x in runs])[:, None])
                for y in E: ax.plot(g, y, color=C_TRAIN, lw=0.45, alpha=0.3, zorder=1)
                ax.plot(g, np.nanmedian(E, 0), color=C_TRAIN, lw=1.1, zorder=3)
                ax.set_yscale('symlog', linthresh=1); ax.set_ylim(-0.2, ntr_top)
                ticks = [t for t in (0, 1, 10, 100, 1000) if t <= ntr_top]
                ax.set_yticks(ticks); ax.set_yticklabels([{1000: '1k'}.get(t, str(t)) for t in ticks])
            elif row == 'ah':
                band(ax, g, PX.on_grid(runs, 'AH_full_resid', g), C_AH); ax.set_ylim(0, 1)
            if i < len(rows) - 1: ax.tick_params(labelbottom=False)
        axs[0, j].set_title(column_title(d['p'], d['cell']))
    lab = dict(ce='Training CE / log p\n(201-update avg.)', acc='Accuracy', err='Misclassified\ntraining pairs',
               ah='Fourier $A_H$\n(head input)')
    for i, r in enumerate(rows): axs[i, 0].set_ylabel(lab[r])
    axs[-1, 1].set_xlabel('Updates (thousands)'); axs[-1, 1].xaxis.set_label_coords(1.08, -0.3)
    h = [Line2D([], [], color=C_CE, lw=1.4), Line2D([], [], color=C_TRAIN, lw=0.9, ls='--'),
         Line2D([], [], color=C_NET, lw=1.6), Line2D([], [], color=C_NET, lw=1.1, ls=DASH),
         Line2D([], [], color=C_REFIT, lw=1.3), Line2D([], [], color=C_FOUR, lw=1.3),
         Line2D([], [], color=C_TRAIN, lw=1.1), Line2D([], [], color=C_AH, lw=1.4), Patch(color=C_BAND, lw=0)]
    l = ['Train CE', 'Train acc.', 'Held-out (all pairs)', 'Held-out (no-mirror pairs)', 'Refit', 'Fourier refit (top 8)',
         'Misclassified', 'Fourier $A_H$', 'Window (median)']
    fig.legend(h, l, loc='outside lower center', ncol=5, frameon=False, handlelength=1.8, columnspacing=1.0, fontsize=8)
    save(fig, outdir, 'ZP_acrossp_paper')


def per_run_mirror(A):
    out = []
    for key in SCATTER:
        cell = A['cells'][key]
        for d in PZ.load_runs(cell):
            r = d['rec']; s, va = PX.series(d['obs'], 'test_acc'); _, vn = PX.series(d['obs'], 'test_acc_nm')
            vm = (r['n_heldout'] * va - r['n_nm'] * vn) / r['n_mirror']
            Wa = r['all'].get('window', {}).get('W'); Wn = r['nm'].get('window', {}).get('W')
            cr = lambda v, th: (int(s[np.where(v >= th)[0][0]]) if (v >= th).any() else None)
            rec = dict(cell=key, id=r['id'], p=r['p'], frac=r['frac'], mirror_share=r['n_mirror'] / r['n_heldout'],
                       first_fit=r['first_fit'], mirror_t50=cr(vm, 0.5), mirror_t90=cr(vm, 0.9), nm_t10=cr(vn, 0.1),
                       W_all=Wa, W_nm=Wn, q_all=r['all'].get('window', {}).get('q_te'), q_nm=r['nm'].get('window', {}).get('q_te'),
                       pass_strict=r['pass_strict'], pass_loose=r['pass_loose'], why=r['why'], final_all=r['all']['final'])
            if Wa:
                rec.update(a0_all=at(s, va, Wa[0]), mirror_at_W=at(s, vm, Wa[0]), nm_at_W=at(s, vn, Wa[0]))
                rec['bound_all'] = min(1.0, (0.9 - max(rec['a0_all'], 0.1)) / 0.8)
            if Wn: rec['a0_nm'] = at(s, vn, Wn[0])
            out.append(rec)
    return out


def mirror_figure(A, outdir):
    M = per_run_mirror(A)
    json.dump(dict(doc='Per-run numbers behind ZP_mirror_paper (paper_ZP.py). a0_all: held-out accuracy on all pairs at the start '
                       'of the all-pairs window; mirror_at_W / nm_at_W: the same on the mirror and no-mirror pairs; bound_all = '
                       '(0.9 - max(a0_all, 0.1)) / 0.8; q_all / q_nm: the shares of analysis_ZP.json.', runs=M),
              open(os.path.join(outdir, 'mirror_ZP.json'), 'w'), indent=1)
    fig = plt.figure(figsize=(6.75, 5.0), layout='constrained')
    gs = fig.add_gridspec(2, 4, height_ratios=[1.0, 1.2])
    top = [fig.add_subplot(gs[0, 0])]; top += [fig.add_subplot(gs[0, j], sharey=top[0]) for j in (1, 2, 3)]
    for j, p in enumerate(PRIMES):
        ax = top[j]; cell = A['cells'][A['appendix'][str(p)]['cell']]; runs = PZ.load_runs(cell)
        T = min(len(d['loss']) for d in runs) - 1; g = np.arange(20, T + 1, 20)
        Ya, Yn, Ym = mirror_acc(runs, g)
        W = [d['rec']['all']['window']['W'] for d in runs if d['rec']['all'].get('window', {}).get('W')]
        a, b = np.median([w[0] for w in W]), np.median([w[1] for w in W])
        ax.axvspan(a, min(b, T), color=C_BAND, lw=0, zorder=0)
        band(ax, g, PX.on_grid(runs, 'train_acc', g), C_TRAIN, ls='--', lw=0.9, iqr=False, z=2)
        band(ax, g, Ym, C_NET, ls=DOT, lw=1.7, iqr=False, z=4)
        band(ax, g, Yn, C_NET, ls=DASH, lw=1.1, iqr=False, z=4)
        band(ax, g, Ya, C_NET, lw=1.6, z=5)
        a0 = np.median([x['a0_all'] for x in M if x['cell'] == A['appendix'][str(p)]['cell'] and 'a0_all' in x])
        ax.plot([a], [a0], 'o', ms=4.5, color=C_NET, mec='white', mew=0.8, zorder=7)
        ax.annotate(f'{a0:.2f}', (a, a0), xytext=(-3, 7), textcoords='offset points', ha='right', fontsize=8, color='#222222')
        ax.set_xscale('log'); ax.set_xlim(60, T); ax.set_ylim(-0.02, 1.04); ax.grid(True, axis='y', color=C_GRID, lw=0.6)
        ax.xaxis.set_major_locator(FixedLocator([100, 1000, 10000]))
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: {100: '100', 1000: '1k', 10000: '10k'}.get(int(round(v)), '')))
        ax.xaxis.set_minor_formatter(FuncFormatter(lambda v, _: ''))
        share = np.median([d['rec']['n_mirror'] / d['rec']['n_heldout'] for d in runs])
        ax.set_title(f"p = {p}, fraction {cell['runs'][0]['frac']:g}\n{100 * share:.0f}% mirror pairs")
        if j: ax.tick_params(labelleft=False)
    top[0].set_ylabel('Held-out accuracy')
    top[1].set_xlabel('Updates (log scale)'); top[1].xaxis.set_label_coords(1.08, -0.16)
    axa = fig.add_subplot(gs[1, 0:2]); axb = fig.add_subplot(gs[1, 2:4])
    x = np.linspace(0.12, 0.38, 10)
    axa.plot(x, x, color=C_REF, lw=0.9, ls='--', zorder=1)
    axa.text(0.128, 0.139, 'all mirror pairs correct', fontsize=8, color='#555555', rotation=45,
             rotation_mode='anchor', transform_rotates_text=True)
    axa.axhline(0.3, color=C_REF, lw=0.8, ls=DOT, zorder=1)
    axa.text(0.125, 0.307, 'above 0.3: share < 0.75', fontsize=8, color='#555555', va='bottom')
    for rec in M:
        if 'a0_all' not in rec or rec['final_all'] < 0.9: continue
        c, mk = P_STYLE[rec['p']]; ok = rec['q_all'] >= 0.75
        kw = dict(marker=mk, s=22, lw=0.9, edgecolor=c, facecolor=c if ok else 'white', zorder=3)
        axa.scatter([rec['mirror_share']], [rec['a0_all']], **kw)
        axb.scatter([rec['a0_all']], [rec['q_all']], **kw)
        if 'a0_nm' in rec and rec['q_nm'] is not None:
            axb.scatter([rec['a0_nm']], [rec['q_nm']], marker='x', s=14, lw=0.8, color='#777777', zorder=2)
    axa.set_xlim(0.12, 0.38); axa.set_ylim(-0.01, 0.42)
    axa.set_xlabel('Share of held-out pairs that are mirrors of training pairs'); axa.set_ylabel('All-pairs held-out accuracy\nwhen the window opens')
    axa.grid(True, color=C_GRID, lw=0.6); axa.set_axisbelow(True)
    z = np.linspace(0, 0.42, 200); axb.plot(z, np.minimum(1, (0.9 - np.maximum(z, 0.1)) / 0.8), color=C_REF, lw=0.9, ls='--', zorder=1)
    axb.axhline(0.75, color=C_REF, lw=0.8, ls=DOT, zorder=1); axb.axvline(0.3, color=C_REF, lw=0.8, ls=DOT, zorder=1)
    axb.text(0.415, 0.757, 'pass: 0.75', ha='right', va='bottom', fontsize=8, color='#555555')
    axb.text(0.205, 0.905, r'$(0.9 - a_0)\,/\,0.8$', fontsize=8, color='#555555', rotation=-33)
    axb.set_xlim(-0.01, 0.42); axb.set_ylim(0.6, 1.03)
    axb.set_xlabel('Held-out accuracy when the window opens, $a_0$'); axb.set_ylabel('Share of the held-out rise\ninside the window')
    axb.grid(True, color=C_GRID, lw=0.6); axb.set_axisbelow(True)
    hp = [Line2D([], [], ls='', marker=P_STYLE[p][1], markersize=5, color=P_STYLE[p][0]) for p in PRIMES]
    hl = [f'p = {p}' for p in PRIMES]
    hp += [Line2D([], [], ls='', marker='o', markersize=5, markerfacecolor='white', markeredgecolor='#444444'),
           Line2D([], [], ls='', marker='x', markersize=5, color='#777777')]
    hl += ['open: share < 0.75', 'no-mirror pairs']
    axb.legend(hp, hl, loc='lower left', ncol=1, fontsize=8, handletextpad=0.3, borderaxespad=0.2, labelspacing=0.25)
    h = [Line2D([], [], color=C_NET, lw=1.6), Line2D([], [], color=C_NET, lw=1.1, ls=DASH), Line2D([], [], color=C_NET, lw=1.7, ls=DOT),
         Line2D([], [], color=C_TRAIN, lw=0.9, ls='--'), Patch(color=C_BAND, lw=0)]
    l = ['Held-out: all pairs', 'No-mirror pairs', 'Mirror pairs', 'Train acc.', 'Window (all pairs)']
    fig.legend(h, l, loc='outside lower center', ncol=5, frameon=False, handlelength=2.0, columnspacing=1.2, fontsize=8)
    save(fig, outdir, 'ZP_mirror_paper')
    return M


def squeeze_figure(A, M, outdir):
    keys = ('P_p61_f0.3_lam_v0.45', 'P_p61_f0.33_lam_v0.45', 'P_p61_f0.35_lam_v0.45')
    col, mk = P_STYLE[61][0], P_STYLE[61][1]; dark = '#1c5cab'
    fig, axs = plt.subplots(1, 3, figsize=(6.75, 2.35), layout='constrained')
    for k, key in enumerate(keys):
        rs = sorted([x for x in M if x['cell'] == key], key=lambda x: x['id'])
        for i, x in enumerate(rs):
            xx = k + (i - 2) * 0.07
            fail = not x['pass_strict']
            axs[0].scatter([xx], [x['a0_all']], marker=mk, s=24, lw=0.9, edgecolor=dark, facecolor='white' if fail else dark, zorder=3)
            axs[1].scatter([xx], [x['q_all']], marker=mk, s=24, lw=0.9, edgecolor=dark, facecolor='white' if fail else dark, zorder=3)
            axs[1].scatter([xx], [x['q_nm']], marker='x', s=16, lw=0.9, color='#777777', zorder=2)
            axs[2].scatter([xx], [x['final_all']], marker=mk, s=24, lw=0.9, edgecolor=dark, facecolor='white' if fail else dark, zorder=3)
        ms = np.median([x['mirror_share'] for x in rs])
        axs[0].plot([k - 0.22, k + 0.22], [ms, ms], color=C_NET, lw=1.6, ls=DOT, zorder=2)
    axs[0].axhline(0.3, color=C_REF, lw=0.8, ls='--', zorder=1); axs[0].set_ylim(0.15, 0.42)
    axs[0].set_ylabel('All-pairs held-out accuracy\nwhen the window opens')
    axs[1].axhline(0.75, color=C_REF, lw=0.8, ls='--', zorder=1); axs[1].set_ylim(0.6, 1.03)
    axs[1].set_ylabel('Share of the held-out rise\ninside the window')
    axs[2].axhline(0.99, color=C_REF, lw=0.8, ls='--', zorder=1); axs[2].set_ylim(0.97, 1.0015)
    axs[2].set_ylabel('Final held-out accuracy\n(all pairs)')
    for ax in axs:
        ax.set_xticks(range(3)); ax.set_xticklabels(['0.30', '0.33', '0.35']); ax.set_xlim(-0.5, 2.5)
        ax.set_xlabel('Training fraction (p = 61)'); ax.grid(True, axis='y', color=C_GRID, lw=0.6); ax.set_axisbelow(True)
    h = [Line2D([], [], ls='', marker=mk, markersize=5, color=dark),
         Line2D([], [], ls='', marker=mk, markersize=5, markerfacecolor='white', markeredgecolor=dark),
         Line2D([], [], ls='', marker='x', markersize=5, color='#777777'), Line2D([], [], color=C_NET, lw=1.6, ls=DOT),
         Line2D([], [], color=C_REF, lw=0.8, ls='--')]
    l = ['run passes', 'run fails', 'no-mirror pairs', 'mirror share of held-out pairs', 'thresholds: 0.3, 0.75, 0.99']
    fig.legend(h, l, loc='outside lower center', ncol=3, frameon=False, handlelength=1.8, columnspacing=1.5, fontsize=8)
    save(fig, outdir, 'ZP_p61_fraction_paper')


def main(ana, outdir):
    A = json.load(open(ana)); os.makedirs(outdir, exist_ok=True)
    acrossp(A, outdir); M = mirror_figure(A, outdir); squeeze_figure(A, M, outdir)


if __name__ == '__main__':
    main(*sys.argv[1:3])
