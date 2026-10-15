#!/usr/bin/env python3
import json, math, os, sys
from pathlib import Path

REL, DATA = os.path.abspath(sys.argv[1]), os.path.abspath(sys.argv[2])
NRES = os.path.join(DATA, 'campaign_N', 'collected', 'results')
os.environ['CAMPAIGN_X_DIR'] = f'{REL}/campaign_X'
os.environ['CAMPAIGN_Y_DIR'] = f'{REL}/campaign_Y'
sys.path.insert(0, f'{REL}/campaign_N')
import acrossp_figs as A
import plot_N as P
import plot_N_cut as C
sys.path.insert(0, f'{REL}/campaign_Z')
import plot_Z as PZ
import plot_X as PX

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator

K = A.K
C_FOUR = PX.C_FOUR
DASH = (0, (4, 2))
THIN = dict(lw=0.45, alpha=0.28, zorder=1)


def style():
    plt.rcParams.update(matplotlib.rcParamsDefault)
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 8, 'axes.titlesize': 8, 'axes.labelsize': 8,
                         'xtick.labelsize': 8, 'ytick.labelsize': 8, 'legend.fontsize': 8, 'axes.spines.top': False,
                         'axes.spines.right': False, 'axes.edgecolor': '#555555', 'axes.linewidth': 0.7})


def deep_cols(R, D, p):
    A.column2 = C.column2_cut
    cols = []
    for f in P.FRACS:
        c = R['cells'][f'D{D}_p{p}_f{f:g}']
        fin = sorted([x for x in R['runs'].values() if x['depth'] == D and x['p'] == p and x['dial']['frac'] == f
                      and x['state'] == 'finished'], key=lambda x: x['seed'])
        assert len(fin) == c['n'] == 10, (D, p, f)
        col = P.column(NRES, fin, f'Fraction {f:g}')
        col.update(kind='deep', passes=c['pass_M'], n=c['n'], r=c['r'])
        cols.append(col)
    return cols


def tf_col(title, cell, cdir, xmax=None):
    runs = []
    for r in cell['runs']:
        d = PX.load(os.path.join(cdir, r['file'])); d['rec'] = r; runs.append(d)
    p = cell['runs'][0]['p']; ntr = cell['runs'][0]['n_train']
    end = min(len(d['loss']) for d in runs) - 1
    xm = min(xmax, end) if xmax else end
    g = np.arange(0, xm + 1, 20)
    S = dict(ce=PZ.ce_grid(runs, g) / math.log(p), train=PX.on_grid(runs, 'train_acc', g),
             test=PX.on_grid(runs, 'test_acc', g), test_nm=PX.on_grid(runs, 'test_acc_nm', g),
             refit=PX.on_grid(runs, 'refit_test_acc', g), four=PX.on_grid(runs, 'fref_top8_test_acc', g),
             ahL=PX.on_grid(runs, 'AH_full_resid', g), ah1=None)
    S['err'] = np.round((1 - S['train']) * ntr)
    for k in ('ce', 'train', 'test', 'test_nm', 'refit', 'four', 'ahL'):
        assert np.isfinite(S[k]).all(), (title, k)
    W = [r['all'].get('window', {}).get('W') for r in cell['runs']]; W = [w for w in W if w]
    band = (float(np.median([w[0] for w in W])), float(np.median([w[1] for w in W]))) if W else None
    del runs
    return dict(kind='tf', g=g.astype(float), S=S, band=band, xmax=xm, title=title, ntr=ntr, n=len(cell['runs']),
                passes=cell['summary']['pass_strict'], loose=cell['summary']['pass_loose'], end=end)


def draw(cols, out, stem, rows, height, legend=True, suptitle=None, ce_shared=None, width=6.75, grid_below=False, refit=True):
    n = len(cols); kind = cols[0]['kind']
    ce_shared = (kind == 'deep') if ce_shared is None else ce_shared
    fig, axs = plt.subplots(len(rows), n, figsize=(width, height), layout='constrained', squeeze=False)
    if kind == 'deep':
        ah_top = max(float(np.nanmax(c['S']['ahL'])) for c in cols) * 1.08
    else:
        ah_top = 1.0
    err_top = max(c['ntr'] for c in cols) * 1.5
    for j, c in enumerate(cols):
        g, S = c['g'], c['S']
        for i, row in enumerate(rows):
            ax = axs[i, j]
            if c['band'] and c['band'][1] - c['band'][0] > 0.004 * c['xmax']:
                ax.axvspan(c['band'][0], min(c['band'][1], c['xmax']), color=K['window'], lw=0, zorder=0)
            ax.grid(True, axis='y', color=K['grid'], lw=0.6); ax.set_xlim(0, c['xmax'])
            if grid_below: ax.set_axisbelow(True)
            ax.xaxis.set_major_formatter(A.KFMT); ax.xaxis.set_major_locator(MaxNLocator(3, integer=True))
            if row == 'ce':
                for y in S['ce']: ax.plot(g, y, color=K['train'], **THIN)
                med = np.median(S['ce'], 0); ax.plot(g, med, color=K['train'], lw=1.4, zorder=3)
                if ce_shared: ax.set_ylim(0, 1.05)
                else:
                    top = 1.18 * float(np.percentile(med[g >= 400], 99)); ax.set_ylim(0, top)
                    ax.yaxis.set_major_locator(MaxNLocator(3))
            elif row == 'acc':
                for y in S['test']: ax.plot(g, y, color=K['test'], **THIN)
                ax.plot(g, np.median(S['train'], 0), color=K['train_acc'], lw=1.0, ls='--', zorder=3)
                if refit: ax.plot(g, np.median(S['refit'], 0), color=K['refit'], lw=1.3, zorder=4)
                if kind == 'tf':
                    ax.plot(g, np.median(S['four'], 0), color=C_FOUR, lw=1.3, zorder=4)
                    ax.plot(g, np.median(S['test_nm'], 0), color=K['test'], lw=1.1, ls=DASH, zorder=5)
                ax.plot(g, np.median(S['test'], 0), color=K['test'], lw=1.4, zorder=6); ax.set_ylim(-0.02, 1.04)
            elif row == 'err':
                for y in S['err']: ax.plot(g, y, color=K['train_acc'], **THIN)
                ax.plot(g, np.median(S['err'], 0), color=K['train_acc'], lw=1.1, zorder=3)
                ax.set_yscale('symlog', linthresh=1); ax.set_ylim(-0.2, err_top)
                ticks = [t for t in (0, 1, 10, 100, 1000) if t <= err_top]
                ax.set_yticks(ticks); ax.set_yticklabels([{1000: '1k'}.get(t, str(t)) for t in ticks])
            elif row == 'ah':
                for y in S['ahL']: ax.plot(g, y, color=K['AH'], **THIN)
                ax.plot(g, np.median(S['ahL'], 0), color=K['AH'], lw=1.4, ls=A.LAST, zorder=3)
                if S.get('ah1') is not None:
                    ax.plot(g, np.median(S['ah1'], 0), color=K['AH'], lw=1.2, ls=A.FIRST, zorder=3)
                ax.set_ylim(0, ah_top)
            if i < len(rows) - 1: ax.tick_params(labelbottom=False)
            shared = row != 'ce' or ce_shared
            if j > 0 and shared: ax.tick_params(labelleft=False)
        axs[0, j].set_title(c['title'], loc='center')
    compact = height < 4
    lab = dict(ce='Train CE\n/ log p',
               acc='Accuracy', err='Misclassified\ntraining pairs',
               ah=('Hidden\nFourier $A_H$' if kind == 'deep' else 'Fourier\n$A_H$'))
    for i, row in enumerate(rows): axs[i, 0].set_ylabel(lab[row])
    for ax in axs[-1]: ax.set_xlabel(' ')
    if suptitle: fig.suptitle(suptitle, fontsize=8.5, x=0.01, ha='left', fontweight='bold')
    if legend:
        h = [Line2D([], [], color=K['train'], lw=1.4), Line2D([], [], color=K['train_acc'], lw=1.0, ls='--'),
             Line2D([], [], color=K['test'], lw=1.4)]
        l = ['Train CE', 'Train acc.', 'Held-out']
        if kind == 'tf': h.append(Line2D([], [], color=K['test'], lw=1.1, ls=DASH)); l.append('Held-out, no-mirror')
        if refit: h.append(Line2D([], [], color=K['refit'], lw=1.3)); l.append('Refit')
        if kind == 'tf': h.append(Line2D([], [], color=C_FOUR, lw=1.3)); l.append('Fourier refit')
        if 'err' in rows: h.append(Line2D([], [], color=K['train_acc'], lw=1.1)); l.append('Misclassified')
        if kind == 'deep':
            h += [Line2D([], [], color=K['layer'], lw=1.4, ls=A.LAST), Line2D([], [], color=K['layer'], lw=1.2, ls=A.FIRST)]
            l += ['Last hidden', 'First hidden']
        else:
            h.append(Line2D([], [], color=K['AH'], lw=1.4)); l.append('$A_H$')
        ncol = len(h) if len(h) <= 7 else (len(h) + 1) // 2
        fig.legend(h, l, loc='outside lower center', ncol=ncol, frameon=False, handlelength=1.6, columnspacing=0.9, fontsize=8)
    fig.canvas.draw(); fig.set_layout_engine('none')
    bb = axs[-1, 0].xaxis.label.get_window_extent(fig.canvas.get_renderer())
    y = fig.transFigure.inverted().transform((0, (bb.y0 + bb.y1) / 2))[1]
    x0, x1 = axs[-1, 0].get_position().x0, axs[-1, -1].get_position().x1
    fig.text((x0 + x1) / 2, y, 'Updates (thousands)', ha='center', va='center', fontsize=8)
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / f'{stem}.pdf', metadata={'Title': stem, 'Author': '', 'Creator': 'paper_uniform.py', 'CreationDate': None})
    fig.savefig(out / f'{stem}.png', dpi=200); plt.close(fig)
    stats = [dict(title=c['title'], n=c['n'], passes=c['passes'], xmax=int(c['xmax']),
                  band=[round(x) for x in c['band']] if c['band'] else None) for c in cols]
    print(stem, json.dumps(stats), flush=True)
    return stats


def main(outdir):
    out = Path(outdir); style(); report = {}
    RN = json.load(open(f'{REL}/campaign_N/analysis/analysis_N.json'))
    report['N_D3_p47_main'] = draw(deep_cols(RN, 3, 47), out, 'N_D3_p47_main', ('ce', 'acc', 'ah'), 2.25)
    for k, (D, p) in enumerate(((3, 61), (4, 47), (4, 61))):
        report[f'N_D{D}_p{p}_app'] = draw(deep_cols(RN, D, p), out, f'N_D{D}_p{p}_app', ('ce', 'acc', 'ah'), 2.25,
                                          legend=(k == 2), suptitle=f'Depth {D}, p = {p}')
    style()
    AZ = json.load(open(f'{REL}/campaign_Z/analysis/analysis_Z.json'))['cells']
    AP = json.load(open(f'{REL}/campaign_ZP/analysis/analysis_ZP.json'))
    zdir, pdir = os.path.join(DATA, 'campaign_Z'), os.path.join(DATA, 'campaign_ZP')
    cols = [tf_col("Polar (the paper's rule)", AZ['Y_G2'], zdir, 9000),
            tf_col('Normalized gradient (NGD)', AZ['C_NGD_G2'], zdir, 9000),
            tf_col('AdamW', AZ['C_ADAMW_G2'], zdir, 9000)]
    report['TF_contrast_main'] = draw(cols, out, 'TF_contrast_main', ('ce', 'acc', 'ah'), 2.25)
    cols = []
    for p_ in ('61', '89', '113', '149'):
        a = AP['appendix'][p_]; cell = AP['cells'][a['cell']]
        cols.append(tf_col(f"p = {p_}, fraction {a['frac']:g}", cell, pdir))
    report['TF_acrossp_app'] = draw(cols, out, 'TF_acrossp_app', ('ce', 'acc', 'err', 'ah'), 4.0)
    cols = []
    for key, ttl in (('B_f0.2_lam_v0.275_tau10k', 'Fraction 0.2, $\\lambda_V$ = 0.275'),
                     ('B_f0.25_lam_v0.4_tau10k', 'Fraction 0.25, $\\lambda_V$ = 0.4'),
                     ('B_f0.3_lam_v0.275_tau5k', 'Fraction 0.3, $\\lambda_V$ = 0.275'),
                     ('B_f0.3_lam_v0.5_tau5k', 'Fraction 0.3, $\\lambda_V$ = 0.5')):
        cols.append(tf_col(ttl, AZ[key], zdir))
    report['TF_two_blocks_app'] = draw(cols, out, 'TF_two_blocks_app', ('ce', 'acc', 'err', 'ah'), 4.0)
    (out / 'paper_uniform_stats.json').write_text(json.dumps(report, indent=1) + '\n')


if __name__ == '__main__':
    main(sys.argv[3])
