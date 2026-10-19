#!/usr/bin/env python3
import json, math, os, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.environ.get('CAMPAIGN_X_DIR', os.path.join(HERE, '..', 'one_block_no_gain_p113')))
sys.path.insert(0, os.environ.get('CAMPAIGN_Y_DIR', os.path.join(HERE, '..', 'one_block_gain_p113')))
sys.path.insert(0, HERE)
import plot_X as PX
from analyze_Y import smooth

plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 8, 'axes.spines.top': False, 'axes.spines.right': False,
                     'axes.titlesize': 8.2, 'legend.frameon': False})
C_GAIN, C_MIRROR, C_H1, C_H2 = '#444444', '#888888', '#7FC97F', '#1B7837'
DASH = (0, (4, 2))
KFMT = FuncFormatter(lambda x_, _: f'{x_ / 1000:g}k')


def load_runs(cell):
    out = []
    for r in cell['runs']:
        fn = r['file'] if os.path.exists(r['file']) else os.path.join(os.getcwd(), r['file'])
        out.append(dict(PX.load(fn), rec=r))
    return out


def ce_grid(runs, grid):
    Y = np.full((len(runs), len(grid)), np.nan)
    for i, d in enumerate(runs):
        cm = smooth(np.asarray(d['loss'], float)); ok = grid < len(cm); Y[i, ok] = cm[grid[ok]]
    return Y


def draw(ax, grid, Y, color, ls='-', lw=1.6, label=None, z=3, thin=False):
    if thin and Y.shape[0] <= 3:
        for row in Y: ax.plot(grid, row, color=color, ls=ls, lw=0.6, alpha=0.45, zorder=z - 1)
        return PX.med_band(ax, grid, Y, color, ls=ls, lw=lw, label=label, band=False, z=z)
    return PX.med_band(ax, grid, Y, color, ls=ls, lw=lw, label=label, band=(ls == '-'), z=z)


def column_figure(cols, out, title, rows=('ce', 'acc', 'ah', 'gain'), xmax=None, ce_logp=False):
    hr = dict(ce=1.0, acc=1.35, ah=0.75, gain=0.6)
    fig, axs = plt.subplots(len(rows), len(cols), figsize=(3.0 * len(cols) + 0.6, 1.75 * sum(hr[r] for r in rows) + 1.0),
                            sharex='col', squeeze=False, gridspec_kw=dict(height_ratios=[hr[r] for r in rows]))
    for j, (ttl, cell) in enumerate(cols):
        runs = load_runs(cell); T = int(np.median([len(d['loss']) for d in runs]))
        T = min(T, xmax) if xmax else T; grid = np.arange(0, T, 20)
        W = [r['nm'].get('window', {}).get('W') for r in cell['runs']]; W = [w for w in W if w]
        for i, rname in enumerate(rows):
            ax = axs[i, j]
            if W:
                a, b = np.median([w[0] for w in W]), np.median([w[1] for w in W])
                ax.axvspan(a, min(b, T), color=PX.C_BAND, lw=0, zorder=0)
            if rname == 'ce':
                Y = ce_grid(runs, grid)
                if ce_logp: Y = Y / math.log(runs[0]['rec']['p'])
                draw(ax, grid, Y, PX.C_CE, thin=True)
                top = np.nanpercentile(Y[:, grid >= 300], 99) if (grid >= 300).any() else np.nanmax(Y)
                ax.set_ylim(0, 1.15 * top if np.isfinite(top) and top > 0 else None)
                S = cell['summary']; ch = S['ce_change']
                if ch['median'] is not None:
                    ax.text(0.97, 0.95, f"held-out 0.1 -> 0.9 (no mirror): CE {100 * ch['median']:+.0f}%, up in {ch['n_rising']}/{ch['n']}",
                            transform=ax.transAxes, ha='right', va='top', fontsize=6.8, color=PX.C_CE)
                ax.set_title(f"{ttl}\npass (both definitions) {S['pass_strict']}/{S['n']} strict, {S['pass_loose']}/{S['n']} loose")
            elif rname == 'acc':
                draw(ax, grid, PX.on_grid(runs, 'train_acc', grid), PX.C_TRAIN, ls='--', lw=0.9, label='Training accuracy')
                draw(ax, grid, PX.on_grid(runs, 'fref_top8_test_acc', grid), PX.C_FOUR, lw=1.5, label='Fourier-filtered refit (top 8), all held-out')
                draw(ax, grid, PX.on_grid(runs, 'fref_top8_test_acc_nm', grid), PX.C_FOUR, ls=DASH, lw=1.1, label='Fourier-filtered refit (top 8), no-mirror')
                draw(ax, grid, PX.on_grid(runs, 'refit_test_acc', grid), PX.C_REFIT, ls=':', lw=1.4, label="Paper's refit, all held-out")
                draw(ax, grid, PX.on_grid(runs, 'test_acc', grid), PX.C_NET, lw=1.9, label='Network, all held-out pairs')
                draw(ax, grid, PX.on_grid(runs, 'test_acc_nm', grid), PX.C_NET, ls=DASH, lw=1.3, label='Network, no-mirror held-out pairs')
                ax.set_ylim(-0.02, 1.02); ax.grid(alpha=0.25, lw=0.5)
            elif rname == 'ah':
                draw(ax, grid, PX.on_grid(runs, 'AH_full_resid', grid), PX.C_AH, lw=1.5, label='$A_H$, head input')
                L = runs[0]['rec']['n_layers']
                for l, col in zip(range(L), (C_H1, C_H2)):
                    draw(ax, grid, PX.on_grid(runs, f'AH_full_h{l + 1}', grid), col, ls=DASH, lw=1.0, label=f'$A_H$, MLP neurons of block {l + 1}')
                ax.set_ylim(0, 1)
            elif rname == 'gain':
                if any('gain' in o for o in runs[0]['obs']):
                    draw(ax, grid, PX.on_grid(runs, 'gain', grid), C_GAIN, lw=1.4, label='Gain g')
                else:
                    ax.plot([0, T], [1, 1], color=C_GAIN, lw=1.4)
                ax.set_ylim(0.8, 3.1)
            if i == len(rows) - 1:
                ax.set_xlabel('Updates'); ax.xaxis.set_major_formatter(KFMT); ax.set_xlim(0, T)
    lab = dict(ce='Training CE / log p\n(201-update average)' if ce_logp else 'Training CE (nats,\n201-update average)',
               acc='Accuracy', ah='$A_H$', gain='Gain g')
    for i, r in enumerate(rows): axs[i, 0].set_ylabel(lab[r])
    h, l = [], []
    for i, r in enumerate(rows):
        if r in ('acc', 'ah'):
            for ax in axs[i]:
                hh, ll = ax.get_legend_handles_labels()
                for a_, b_ in zip(hh, ll):
                    if b_ not in l: h.append(a_); l.append(b_)
    h.append(plt.Rectangle((0, 0), 1, 1, color=PX.C_BAND)); l.append("Paper's window (median), CE never > 3% below its running max")
    fig.legend(h, l, loc='lower center', ncol=3, fontsize=7.0, bbox_to_anchor=(0.5, 0.0))
    fig.suptitle(title, fontsize=9.3, y=0.995)
    fig.subplots_adjust(left=0.07, right=0.99, top=0.9, bottom=0.17, hspace=0.18, wspace=0.2)
    fig.savefig(out, dpi=160); plt.close(fig)


def recipe_figure(cells, out):
    G0 = (1.5, 2.0, 3.0); TAU = (2500, 5000, 10000, 20000, None)
    def find(g0, tau):
        for k, v in cells.items():
            r = v['runs'][0]
            if not k.startswith('R_') and k != 'Y_G2': continue
            if r['gain_init'] == g0 and (r['tau'] == tau or (tau is None and not r['eta_g'])) and r['p'] == 113 and r['n_layers'] == 1 \
                    and r['lam_v'] == 0.275 and r['eta_w'] == 0.1 and r['lam_w'] == 0.01 and r['frac'] == 0.2 and r['optimizer'] == 'polar' \
                    and (r.get('hidden_mode') or 'polar') == 'polar':
                R = [x for x in v['runs'] if int(x['id'].rsplit('_s', 1)[1]) in (11, 12, 13)]
                return R
        return None
    fig, axs = plt.subplots(1, 3, figsize=(13.5, 3.6), gridspec_kw=dict(width_ratios=[1, 1, 1.5]))
    for k, (ax, what) in enumerate(zip(axs[:2], ('ce', 'q'))):
        M = np.full((len(G0), len(TAU)), np.nan); txt = {}
        for i, g0 in enumerate(G0):
            for j, tau in enumerate(TAU):
                R = find(g0, tau)
                if not R: continue
                if what == 'ce':
                    x = [r['rise']['ce_change'] for r in R if 'rise' in r]; M[i, j] = np.median(x) if x else np.nan
                else:
                    x = [min(r['all'].get('window', {}).get('q_te', 0), r['nm'].get('window', {}).get('q_te', 0)) for r in R]; M[i, j] = np.median(x)
                txt[(i, j)] = f"{sum(r['pass_strict'] for r in R)}/{len(R)}"
        if what == 'ce':
            im = ax.imshow(np.clip(M, -0.5, 0.5), cmap='RdBu', vmin=-0.5, vmax=0.5, aspect='auto')
            ax.set_title('CE change across the held-out rise (median, 3 seeds)\nnumbers: seeds passing (strict)')
        else:
            im = ax.imshow(M, cmap='Greens', vmin=0, vmax=1, aspect='auto')
            ax.set_title("Share of the rise inside the paper's window\n(the smaller of the two definitions; median)")
        for (i, j), t in txt.items():
            v = M[i, j]
            rgb = im.cmap(im.norm(np.clip(v, -0.5, 0.5) if what == 'ce' else v))[:3] if np.isfinite(v) else (1, 1, 1)
            ink = 'white' if 0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2] < 0.45 else 'black'
            ax.text(j, i, (f'{100 * v:+.0f}%\n' if what == 'ce' else f'{v:.2f}\n') + t if np.isfinite(v) else t,
                    ha='center', va='center', fontsize=7, color=ink)
        ax.set_xticks(range(len(TAU))); ax.set_xticklabels([f'{t / 1000:g}k' if t else 'fixed' for t in TAU])
        ax.set_yticks(range(len(G0))); ax.set_yticklabels([f'{g:g}' for g in G0])
        ax.set_xlabel('time constant of the gain, 1 / (eta_g lam_g)'); ax.set_ylabel('initial gain g0')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    ax = axs[2]; names = []
    one = [(k, v) for k, v in cells.items() if k.startswith('R_') and not k.startswith('R_G')]
    ref = find(2.0, 5000)
    rows = ([('G2 (campaign Y, seeds 11-13)', ref)] if ref else []) + [(k[2:], v['runs']) for k, v in sorted(one)]
    for i, (nm, R) in enumerate(rows):
        x = [r['rise']['ce_change'] if 'rise' in r else np.nan for r in R]
        ax.scatter(np.full(len(x), i), 100 * np.array(x), color=PX.C_CE, s=14, zorder=3)
        names.append(f"{nm}\n{sum(r['pass_strict'] for r in R)}/{len(R)} pass")
    ax.axhline(0, color='k', lw=0.6); ax.set_xticks(range(len(names))); ax.set_xticklabels(names, rotation=35, ha='right', fontsize=6.5)
    ax.set_ylabel('CE change across the rise (%)'); ax.set_title('One factor at a time around G2 (g0 2, time constant 5k)')
    fig.subplots_adjust(left=0.05, right=0.99, top=0.84, bottom=0.3, wspace=0.35)
    fig.savefig(out, dpi=160); plt.close(fig)


def main(ana, outdir):
    A = json.load(open(ana))['cells']; os.makedirs(outdir, exist_ok=True)
    have = lambda *ks: [k for k in ks if k in A]
    cols = [(t, A[k]) for t, k in (('Polar (the paper\'s rule), gain 2 -> 1 [campaign Y]', 'Y_G2'), ('NGD feature steps, gain 2 -> 1', 'C_NGD_G2'),
                                   ('AdamW, gain 2 -> 1', 'C_ADAMW_G2')) if k in A]
    if cols: column_figure(cols, os.path.join(outdir, 'Z_contrast_gain.png'), 'Optimizer contrast with the gain (one block, p = 113, frac 0.2)'); print('drew Z_contrast_gain')
    cols = [(t, A[k]) for t, k in (('Polar, no gain [campaign X R5]', 'X_R5'), ('NGD feature steps, no gain', 'C_NGD_NOGAIN'),
                                   ('AdamW, no gain', 'C_ADAMW_NOGAIN')) if k in A]
    if cols: column_figure(cols, os.path.join(outdir, 'Z_contrast_nogain.png'), 'Optimizer contrast without the gain (one block, p = 113, frac 0.2)', rows=('ce', 'acc', 'ah')); print('drew Z_contrast_nogain')
    if any(k.startswith('R_') for k in A): recipe_figure(A, os.path.join(outdir, 'Z_recipe.png')); print('drew Z_recipe')
    P = sorted([k for k in A if k.startswith('P_')], key=lambda k: (A[k]['runs'][0]['p'], A[k]['runs'][0]['frac']))
    if P:
        cols = [(f"p = {A[k]['runs'][0]['p']}, frac {A[k]['runs'][0]['frac']:g}, lam_v {A[k]['runs'][0]['lam_v']:g}", A[k]) for k in P]
        if 'Y_G2' in A: cols.insert(sum(A[k]['runs'][0]['p'] < 113 for k in P), ('p = 113, frac 0.2, lam_v 0.275 [campaign Y, G2]', A['Y_G2']))
        column_figure(cols, os.path.join(outdir, 'Z_p.png'), 'Other p (one block, gain 2 -> 1)', ce_logp=True); print('drew Z_p')
    B = sorted([k for k in A if k.startswith('B_')])
    if B:
        cols = [(f"2 blocks, frac {A[k]['runs'][0]['frac']:g}, lam_v {A[k]['runs'][0]['lam_v']:g}, g0 {A[k]['runs'][0]['gain_init']:g}, tau {A[k]['runs'][0]['tau'] / 1000:g}k", A[k]) for k in B]
        column_figure(cols, os.path.join(outdir, 'Z_blocks.png'), 'Two blocks (p = 113, gain on the RMS head input)'); print('drew Z_blocks')


if __name__ == '__main__':
    main(*sys.argv[1:3])
