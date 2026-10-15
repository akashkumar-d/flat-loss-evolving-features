#!/usr/bin/env python3
import glob, gzip, json, math, os, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 8.5, 'axes.spines.top': False, 'axes.spines.right': False,
                     'axes.titlesize': 9, 'legend.frameon': False})
C_CE, C_TRAIN, C_NET, C_REFIT, C_AH, C_FOUR, C_RESTR, C_GREY, C_BAND = ('#0072B2', '#000000', '#D55E00', '#CC79A7', '#009E73',
                                                                         '#56B4E9', '#E69F00', '#888888', '#EBEBEB')


def load(fn):
    with gzip.open(fn, 'rt') as f: return json.load(f)


def series(obs, key):
    s = [(o['step'], o[key]) for o in obs if o.get(key) is not None]
    return np.array([x[0] for x in s], float), np.array([x[1] for x in s], float)


def on_grid(runs, key, grid):
    out = np.full((len(runs), len(grid)), np.nan)
    for i, d in enumerate(runs):
        s, v = series(d['obs'], key)
        if not len(s): continue
        j = np.searchsorted(s, grid, side='right') - 1
        ok = (j >= 0) & (grid <= s[-1]); out[i, ok] = v[j[ok]]
    return out


def med_band(ax, x, Y, color, ls='-', lw=1.6, label=None, band=True, z=3):
    m = np.nanmedian(Y, 0)
    if band and Y.shape[0] > 2:
        lo, hi = np.nanpercentile(Y, 25, 0), np.nanpercentile(Y, 75, 0)
        ax.fill_between(x, lo, hi, color=color, alpha=0.18, lw=0, zorder=z - 1)
    ax.plot(x, m, color=color, ls=ls, lw=lw, label=label, zorder=z)
    return m


def groups(xdir, mdir):
    G = {}
    for fn in sorted(glob.glob(os.path.join(xdir, '*', 'run.json.gz'))):
        rid = os.path.basename(os.path.dirname(fn)); G.setdefault(rid.split('_')[1], []).append(dict(load(fn), id=rid))
    M = [dict(load(fn), id=os.path.basename(fn)) for fn in sorted(glob.glob(os.path.join(mdir, '*.json.gz')))]
    return G, M


def fig_main(A, G, out):
    arms = [a for a in ('R5', 'R4') if a in G]
    fig, axs = plt.subplots(3, len(arms), figsize=(4.6 * len(arms), 7.2), squeeze=False, sharex='col',
                            gridspec_kw=dict(height_ratios=[1, 1.5, 0.8]))
    for j, arm in enumerate(arms):
        R = G[arm]; S = A['groups'][f'transformer {arm}']; lam = R[0]['cfg']['lam_v']
        T = min(len(d['loss']) for d in R) - 1
        hv = np.array(sorted({o['step'] for o in R[0]['obs'] if 'fref_top8_test_acc_nm' in o}), float); hv = hv[hv <= T]
        lg = np.arange(0, T + 1, 20, dtype=float)
        ax = axs[0, j]
        for d in R:
            L = np.asarray(d['loss'][:T + 1], float); cm = np.convolve((L[:-1] + L[1:]) / 2, np.ones(201) / 201, 'valid')
            ax.plot(np.arange(len(cm)) + 100, cm, color=C_CE, lw=0.6, alpha=0.3)
        cms = np.array([np.convolve((np.asarray(d['loss'][:T + 1])[:-1] + np.asarray(d['loss'][:T + 1])[1:]) / 2,
                                    np.ones(201) / 201, 'valid') for d in R])
        mcm = np.median(cms, 0); ax.plot(np.arange(cms.shape[1]) + 100, mcm, color=C_CE, lw=1.8, label='Training CE (201-update cycle mean; median bold)')
        lo_, hi_ = 0.9 * float(mcm.min()), 1.35 * float(mcm[min(len(mcm) - 1, 1400)])
        ax.set_ylim(lo_, hi_); ax.set_ylabel('Training CE (nats)')
        bands = [r['band'] for r in S['runs'] if 'band' in r]
        if bands:
            ch = np.median([b['ce_rel_change'] for b in bands]) * 100; sl = np.median([b['slope_pct_per_1k'] for b in bands])
            ax.text(0.98, 0.92, f'over the grey band: CE {ch:+.1f}% (median), {sl:+.1f}% per 1000 updates', transform=ax.transAxes,
                    ha='right', va='top', fontsize=7.5, color=C_CE)
        ax.set_title(f'Transformer, RMS head, frac 0.2, $\\lambda_V$ = {lam:g} ({len(R)} seed{"s" if len(R) > 1 else ""})')
        ax = axs[1, j]
        tr = on_grid(R, 'train_acc', lg); med_band(ax, lg, tr, C_TRAIN, '--', 1.0, 'Training accuracy', band=False)
        net = on_grid(R, 'test_acc_nm', lg); med_band(ax, lg, net, C_NET, '-', 2.0, 'Network, held-out (no-mirror pairs)', z=5)
        med_band(ax, hv, on_grid(R, 'fref_all_test_acc_nm', hv), C_REFIT, ':', 2.0, "Paper's refit (all frequencies)", band=False)
        med_band(ax, hv, on_grid(R, 'fref_top8_test_acc_nm', hv), C_FOUR, '-', 1.8, 'Fourier-filtered refit (top 8 frequencies)')
        med_band(ax, hv, on_grid(R, 'fnet_top8_test_acc_nm', hv), C_RESTR, '-', 1.3, "Network's logits restricted to the top 8",
                 band=False)
        on = [r['band']['a'] for r in S['runs'] if 'band' in r]; t90 = [r['net']['0.9'] for r in S['runs'] if r['net']['0.9']]
        if on and t90:
            for a_ in axs[:, j]: a_.axvspan(np.median(on), np.median(t90), color=C_BAND, zorder=0, lw=0)
        ax.set_ylim(-0.02, 1.02); ax.set_ylabel('Accuracy'); ax.grid(alpha=0.25, lw=0.5)
        ax = axs[2, j]
        med_band(ax, hv, on_grid(R, 'AH_full_resid', hv), C_AH, '-', 1.6, 'Head-input Fourier $A_H$')
        ax.set_ylabel('$A_H$'); ax.set_xlabel('Updates'); ax.set_ylim(0, 1)
        ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, _: f'{x / 1000:g}k'))
    h, l = [], []
    for ax in axs[:, 0]:
        hh, ll = ax.get_legend_handles_labels(); h += hh; l += ll
    h.append(plt.Rectangle((0, 0), 1, 1, color=C_BAND)); l.append('Filtered-refit onset to network 90% (medians)')
    fig.legend(h, l, loc='lower center', ncol=3, fontsize=7.2, bbox_to_anchor=(0.53, 0.0))
    fig.subplots_adjust(left=0.1, right=0.98, top=0.95, bottom=0.14, hspace=0.12, wspace=0.25)
    fig.savefig(out, dpi=180); plt.close(fig)


def median_seed(S, name='fourier_top8'):
    L = [(r['readouts'][name]['lead']['0.5'], i) for i, r in enumerate(S['runs'])
         if name in r['readouts'] and r['readouts'][name]['lead']['0.5'] is not None]
    if not L: return 0
    L.sort(); return L[len(L) // 2][1]


def fig_freq(A, G, M, out):
    cols = []
    if M: cols.append(('MLP, one hidden layer, p = 31', M, A['groups']['MLP 1HL p=31'], 'test_acc', '_test_acc'))
    for arm in ('R5', 'R4'):
        if arm in G: cols.append((f'Transformer {arm} ($\\lambda_V$ = {G[arm][0]["cfg"]["lam_v"]:g})', G[arm],
                                  A['groups'][f'transformer {arm}'], 'test_acc_nm', '_test_acc_nm'))
    fig, axs = plt.subplots(3, len(cols), figsize=(4.5 * len(cols), 8.4), squeeze=False,
                            gridspec_kw=dict(height_ratios=[1.25, 0.9, 0.9]))
    for j, (title, R, S, nkey, suf) in enumerate(cols):
        i = median_seed(S); d = R[i]; rec = S['runs'][i]
        H = [o for o in d['obs'] if 'fspec_hor2' in o]; ts = np.array([o['step'] for o in H], float)
        R2 = np.clip(np.array([o['fspec_hor2'] for o in H]).T, 0, 1); K = R2.shape[0]; dt = ts[1] - ts[0]
        ax = axs[0, j]
        im = ax.imshow(R2, aspect='auto', origin='lower', extent=[ts[0] - dt / 2, ts[-1] + dt / 2, 0.5, K + 0.5], cmap='Blues',
                       vmin=0, vmax=1, interpolation='nearest')
        for o in H: ax.scatter([o['step']] * 8, o['ftop8'][:8], s=2.2, color=C_NET, lw=0)
        t50 = rec['net']['0.5']
        if t50: ax.axvline(t50, color=C_GREY, ls='--', lw=0.9)
        ax.set_title(f'{title}\nseed {d["id"].split("_s")[-1].split(".")[0]} (median lead)')
        ax.set_ylabel('Frequency k'); ax.yaxis.set_major_locator(MaxNLocator(integer=True, nbins=6))
        ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, _: f'{x / 1000:g}k'))
        ax = axs[1, j]
        o = min(H, key=lambda o: abs(o['step'] - (t50 or 0)))
        k = np.arange(1, K + 1); E = np.array(o['fspec_train']); r2 = np.clip(np.array(o['fspec_hor2']), 0, 1)
        ax.bar(k, E / E.max(), color=C_GREY, alpha=0.55, width=0.8, label='Training energy $E_k$ (scaled to max 1)')
        hd = np.array(o.get('fspec_head', np.zeros(K))); ax.plot(k, hd / hd.max(), color=C_REFIT, lw=0, marker='_', ms=6,
                                                                label='Head Fourier power (scaled)')
        ax.plot(k, r2, color=C_FOUR, lw=0, marker='o', ms=3.2, label='Held-out $R^2_k$ (diagnostic)')
        ax.scatter(o['ftop8'][:8], [1.08] * len(o['ftop8'][:8]), marker='v', s=14, color=C_NET, label='Top 8 by training energy')
        ax.set_ylim(0, 1.15); ax.set_xlim(0.3, K + 0.7); ax.set_xlabel('Frequency k')
        ax.set_title(f'At the network\'s 50% (update {o["step"]})')
        ax = axs[2, j]
        grid = np.array(sorted({o['step'] for o in R[0]['obs'] if 'fspec_hor2' in o}), float)
        cnt = np.full((len(R), len(grid)), np.nan)
        for ii, dd in enumerate(R):
            for o in dd['obs']:
                if 'fspec_hor2' in o and o['step'] <= grid[-1]:
                    cnt[ii, int(np.searchsorted(grid, o['step']))] = (np.array(o['fspec_hor2']) > 0.05).sum()
        med_band(ax, grid, cnt / K, C_FOUR, '-', 1.6, 'Share of frequencies with held-out $R^2_k$ > 0.05')
        med_band(ax, grid, on_grid(R, nkey, grid), C_NET, '-', 1.6, 'Network held-out accuracy', band=False)
        ax.set_ylim(-0.02, 1.02); ax.set_xlabel('Updates'); ax.grid(alpha=0.25, lw=0.5)
        ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, _: f'{x / 1000:g}k'))
        ax.set_title(f'All seeds (median, interquartile band); K = {K} frequencies')
    fig.subplots_adjust(left=0.06, right=0.9, top=0.93, bottom=0.12, hspace=0.45, wspace=0.22)
    pos = axs[0, -1].get_position(); cax = fig.add_axes([0.915, pos.y0, 0.01, pos.height])
    cb = fig.colorbar(im, cax=cax); cb.set_label('Held-out $R^2_k$', fontsize=7.5); cb.ax.tick_params(labelsize=7)
    h, l = axs[1, 0].get_legend_handles_labels(); h2, l2 = axs[2, 0].get_legend_handles_labels()
    dot = plt.Line2D([], [], color=C_NET, marker='o', ls='', ms=3)
    fig.legend(h + h2 + [dot], l + l2 + ['Top 8 by training energy (heatmaps; train-only)'], loc='lower center', ncol=3,
               fontsize=7.5, bbox_to_anchor=(0.48, 0.0))
    fig.savefig(out, dpi=180); plt.close(fig)


def fig_leads(A, out):
    names = [('paper_refit', "Paper's refit (all)"), ('fourier_top4', 'Fourier refit, top 4'), ('fourier_top8', 'Fourier refit, top 8'),
             ('fourier_top16', 'Fourier refit, top 16'), ('phase_top8', 'Phase vote, top 8'),
             ('net_top8', 'Restricted logits, top 8'), ('probe', 'Logistic probe')]
    gs = [g for g in ('MLP 1HL p=31', 'transformer R5', 'transformer R4') if g in A['groups']]
    cols = {'MLP 1HL p=31': '#888888', 'transformer R5': '#0072B2', 'transformer R4': '#222222'}
    fig, axs = plt.subplots(1, 2, figsize=(9.5, 3.9), sharey=True)
    for ax, q in zip(axs, ('0.5', '0.9')):
        for gi, g in enumerate(gs):
            for ni, (n, _) in enumerate(names):
                y0 = len(names) - 1 - ni + (gi - (len(gs) - 1) / 2) * 0.22
                L = [r['readouts'][n]['lead'][q] for r in A['groups'][g]['runs'] if n in r['readouts']]
                have = [x for x in L if x is not None]; miss = len(L) - len(have)
                if have:
                    ax.scatter(have, [y0] * len(have), s=9, color=cols[g], alpha=0.65, lw=0)
                    ax.plot([np.median(have)] * 2, [y0 - 0.09, y0 + 0.09], color=cols[g], lw=2.2)
                if miss: ax.text(0, y0, f' {miss} of {len(L)} not by the end', fontsize=6, va='center', color=cols[g])
        ax.axvline(0, color=C_GREY, ls='--', lw=0.9)
        ax.set_yticks(range(len(names))); ax.set_yticklabels([n[1] for n in names][::-1])
        ax.set_xlabel(f'Lead at held-out {float(q):.0%} (updates)')
        ax.grid(axis='x', alpha=0.25, lw=0.5)
    hs = [plt.Line2D([], [], color=cols[g], marker='o', ls='', ms=4) for g in gs]
    fig.legend(hs, [g.replace('transformer', 'Transformer').replace('MLP 1HL p=31', 'MLP, one hidden layer, p = 31') for g in gs],
               loc='upper center', ncol=3, fontsize=7.5)
    fig.text(0.59, 0.015, 'Lead = network crossing minus readout crossing; > 0: the readout crosses first. Bars: medians.',
             ha='center', fontsize=7)
    fig.subplots_adjust(left=0.2, right=0.98, top=0.88, bottom=0.17, wspace=0.08)
    fig.savefig(out, dpi=180); plt.close(fig)


def fig_family(A, G, M, out):
    cols = []
    if M: cols.append(('MLP, one hidden layer, p = 31', M, 'test_acc', '_test_acc'))
    for arm in ('R5', 'R4'):
        if arm in G: cols.append((f'Transformer {arm}', G[arm], 'test_acc_nm', '_test_acc_nm'))
    fig, axs = plt.subplots(1, len(cols), figsize=(4.3 * len(cols), 3.4), squeeze=False, sharey=True)
    for j, (title, R, nkey, suf) in enumerate(cols):
        ax = axs[0, j]
        hv = np.array(sorted({o['step'] for o in R[0]['obs'] if 'ffam_all' + suf in o}), float)
        lg = np.array(sorted({o['step'] for o in R[0]['obs']}), float); lg = lg[lg <= hv[-1]]
        med_band(ax, lg, on_grid(R, nkey, lg), C_NET, '-', 2.0, 'Network')
        med_band(ax, hv, on_grid(R, 'ffam_all' + suf, hv), C_GREY, '--', 1.5, "Family alone, mean over all pairs (Janati et al.'s filter)")
        med_band(ax, hv, on_grid(R, 'ffam_tr' + suf, hv), C_TRAIN, ':', 1.3, 'Family alone, mean over training pairs only', band=False)
        med_band(ax, hv, on_grid(R, 'ffam_ho' + suf, hv), C_CE, '-', 1.5, 'Family alone, mean over held-out pairs only')
        med_band(ax, hv, on_grid(R, 'fref_top8' + suf, hv), C_FOUR, '-', 1.5, 'Fourier-filtered refit, top 8 (train-only)', band=False)
        ax.set_title(title); ax.set_xlabel('Updates'); ax.set_ylim(-0.02, 1.02); ax.grid(alpha=0.25, lw=0.5)
        ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, _: f'{x / 1000:g}k'))
    axs[0, 0].set_ylabel('Held-out accuracy')
    h, l = axs[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc='lower center', ncol=3, fontsize=7.5, bbox_to_anchor=(0.5, 0.0))
    fig.subplots_adjust(left=0.07, right=0.98, top=0.9, bottom=0.3, wspace=0.08)
    fig.savefig(out, dpi=180); plt.close(fig)


def main(ana, xdir, mdir, outdir):
    os.makedirs(outdir, exist_ok=True)
    A = json.load(open(ana)); G, M = groups(xdir, mdir)
    fig_main(A, G, os.path.join(outdir, 'X_main.png'))
    fig_freq(A, G, M, os.path.join(outdir, 'X_freq.png'))
    fig_leads(A, os.path.join(outdir, 'X_leads.png'))
    fig_family(A, G, M, os.path.join(outdir, 'X_family.png'))
    print('wrote', sorted(os.listdir(outdir)))


if __name__ == '__main__':
    main(*sys.argv[1:5])
