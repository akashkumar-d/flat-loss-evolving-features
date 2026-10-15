#!/usr/bin/env python3
import glob, json, os, sys
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
sys.path.insert(0, '/home/claude/kt/campaign_X'); sys.path.insert(0, '/home/claude/kt/campaign_Y')
import analyze_X as AX, plot_X as PX, analyze_Y as AY

plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 8, 'axes.spines.top': False, 'axes.spines.right': False,
                     'axes.titlesize': 8.5, 'legend.frameon': False})


def counts(c):
    p = c['p']; a = np.repeat(np.arange(p), p); b = np.tile(np.arange(p), p)
    perm = np.random.default_rng(c['split_seed']).permutation(p * p); n = int(c['frac'] * p * p)
    tr = np.sort(perm[:n]); te = np.sort(perm[n:]); intr = np.zeros(p * p, bool); intr[tr] = True
    nm = te[~intr[b[te] * p + a[te]]]; return len(te), len(nm)


def per_run(fn):
    d = AX.load(fn); obs = d['obs']; c = d.get('cfg_full') or d['cfg']
    N, Nnm = counts(c); Nmir = N - Nnm
    L = np.asarray(d['loss'], float); cm = AY.smooth(L); st, tr = AX.series(obs, 'train_acc'); fit = AX.first_logged(st, tr, 1.0)
    out = dict(N=N, N_nm=Nnm, N_mirror=Nmir)
    s, va = AX.series(obs, 'test_acc'); s2, vn = AX.series(obs, 'test_acc_nm'); assert np.array_equal(s, s2)
    vm = (va * N - vn * Nnm) / Nmir
    out['mirror_at_fit'] = float(AY.at(s, vm, fit)) if fit is not None else None
    out['acc_at_fit'] = dict(all=float(AY.at(s, va, fit)), nm=float(AY.at(s, vn, fit))) if fit is not None else None
    for tag, suf in (('all', ''), ('nm', '_nm')):
        sv, v = AX.series(obs, 'test_acc' + suf)
        net = {q: AX.interp_cross(sv, v, q) for q in (0.1, 0.5, 0.9)}
        R = dict(net={str(q): t for q, t in net.items()}, final=float(v[-1]))
        for name, key in (('fourier_top8', 'fref_top8_test_acc'), ('paper_refit', 'refit_test_acc')):
            sr, vr = AX.series(obs, key + suf)
            R[name] = {str(q): (net[q] - AX.interp_cross(sr, vr, q)) if (net[q] is not None and AX.interp_cross(sr, vr, q) is not None) else None
                       for q in (0.5, 0.9)}
        if fit is not None:
            q, I = AY.paper_window(cm, int(fit), sv, v); R['q_te'] = q; R['W'] = I
        out[tag] = R
    return out


def main(ydir, xdir, outdir):
    os.makedirs(outdir, exist_ok=True)
    G = {}
    for fn in sorted(glob.glob(os.path.join(ydir, '*', 'run.json.gz'))):
        G.setdefault(os.path.basename(os.path.dirname(fn)).split('_')[1], []).append(fn)
    G = {'X_R5': sorted(f for f in glob.glob(os.path.join(xdir, 'X_R5_*', 'run.json.gz'))), **G}
    res = {g: [dict(per_run(f), id=os.path.basename(os.path.dirname(f))) for f in fs] for g, fs in G.items()}
    json.dump(res, open(os.path.join(outdir, 'heldout_defs.json'), 'w'), indent=1, default=float)
    med = lambda x: float(np.median([y for y in x if y is not None])) if any(y is not None for y in x) else None
    rng = lambda x: (min(y for y in x if y is not None), max(y for y in x if y is not None))
    for g, R in res.items():
        print(f'== {g} ({len(R)} runs): held-out pairs {R[0]["N"]}, no-mirror {R[0]["N_nm"]}-{max(r["N_nm"] for r in R)}, mirror in training {min(r["N_mirror"] for r in R)}-{max(r["N_mirror"] for r in R)}')
        print(f'   at the first full fit: all {med([r["acc_at_fit"]["all"] for r in R]):.3f}, no-mirror {med([r["acc_at_fit"]["nm"] for r in R]):.3f}, mirror-in-training subset {med([r["mirror_at_fit"] for r in R]):.3f}')
        for tag in ('all', 'nm'):
            X = [r[tag] for r in R]
            print(f'   [{tag}] net 0.1/0.5/0.9 medians', [med([x['net'][q] for x in X]) for q in ('0.1', '0.5', '0.9')],
                  '| final', rng([x['final'] for x in X]),
                  '| q_te', rng([x.get('q_te') for x in X]),
                  '| Fourier top-8 lead 0.5/0.9: median', [med([x['fourier_top8'][q] for x in X]) for q in ('0.5', '0.9')],
                  'n_lead', [sum((x['fourier_top8'][q] or -1) > 0 for x in X) for q in ('0.5', '0.9')],
                  '| paper refit lead 0.5:', med([x['paper_refit']['0.5'] for x in X]))
    titles = {'X_R5': 'Campaign X R5: no gain', 'G2': 'G2: gain 2 -> 1 (time constant 5k)', 'G3': 'G3: gain 3 -> 1 (time constant 4k)'}
    fig, axs = plt.subplots(2, 3, figsize=(11.5, 6.0), sharex='col', sharey=True)
    for j, g in enumerate(('X_R5', 'G2', 'G3')):
        runs = [AX.load(f) for f in G[g]]; T = max(len(d['loss']) for d in runs); grid = np.arange(0, T, 20)
        Nc = [counts(d.get('cfg_full') or d['cfg']) for d in runs]
        A = PX.on_grid(runs, 'test_acc', grid); B = PX.on_grid(runs, 'test_acc_nm', grid)
        Mi = np.stack([(A[i] * Nc[i][0] - B[i] * Nc[i][1]) / (Nc[i][0] - Nc[i][1]) for i in range(len(runs))])
        ax = axs[0, j]
        PX.med_band(ax, grid, PX.on_grid(runs, 'train_acc', grid), PX.C_TRAIN, ls='--', lw=0.9, label='Training accuracy', band=False)
        PX.med_band(ax, grid, Mi, PX.C_GREY, ls=':', lw=1.4, label='Held-out pairs whose mirror (b, a) is in training (~2.0k)')
        PX.med_band(ax, grid, A, PX.C_NET, lw=1.8, label='All held-out pairs (10,216)')
        PX.med_band(ax, grid, B, PX.C_NET, ls=(0, (4, 2)), lw=1.4, label='No-mirror held-out pairs (~8.2k)', band=False)
        ax.set_title(titles[g]); ax.grid(alpha=0.25, lw=0.5)
        ax = axs[1, j]
        PX.med_band(ax, grid, PX.on_grid(runs, 'fref_top8_test_acc', grid), PX.C_FOUR, lw=1.6, label='Fourier-filtered refit (top 8), all held-out')
        PX.med_band(ax, grid, PX.on_grid(runs, 'fref_top8_test_acc_nm', grid), PX.C_FOUR, ls=(0, (4, 2)), lw=1.2, band=False, label='Fourier-filtered refit (top 8), no-mirror')
        PX.med_band(ax, grid, PX.on_grid(runs, 'refit_test_acc', grid), PX.C_REFIT, lw=1.4, label="Paper's refit, all held-out")
        PX.med_band(ax, grid, PX.on_grid(runs, 'refit_test_acc_nm', grid), PX.C_REFIT, ls=(0, (4, 2)), lw=1.1, band=False, label="Paper's refit, no-mirror")
        PX.med_band(ax, grid, A, PX.C_NET, lw=1.0, band=False, label='Network, all held-out', z=2)
        ax.grid(alpha=0.25, lw=0.5); ax.set_xlabel('Updates'); ax.set_xlim(0, 9000)
        ax.xaxis.set_major_formatter(FuncFormatter(lambda x_, _: f'{x_ / 1000:g}k'))
    axs[0, 0].set_ylabel('Network accuracy'); axs[1, 0].set_ylabel('Readout accuracy (held-out)')
    for ax in axs.flat: ax.set_ylim(-0.02, 1.02)
    h0, l0 = axs[0, 0].get_legend_handles_labels(); h1, l1 = axs[1, 0].get_legend_handles_labels()
    fig.legend(h0 + h1, l0 + l1, loc='lower center', ncol=3, fontsize=7.3, bbox_to_anchor=(0.5, 0.0))
    fig.suptitle('Held-out accuracy on all held-out pairs vs the no-mirror subset (medians over 10 seeds, interquartile bands)', fontsize=9.5, y=0.995)
    fig.subplots_adjust(left=0.06, right=0.99, top=0.93, bottom=0.2, hspace=0.12, wspace=0.07)
    fig.savefig(os.path.join(outdir, 'heldout_defs.png'), dpi=170); plt.close(fig)


if __name__ == '__main__':
    main(*sys.argv[1:4])
