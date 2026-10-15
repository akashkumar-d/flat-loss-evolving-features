#!/usr/bin/env python3
import glob, gzip, json, os, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, FuncFormatter

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.environ.get('CAMPAIGN_X_DIR', os.path.join(HERE, '..', 'campaign_X'))); sys.path.insert(0, HERE)
import plot_X as PX
from analyze_Y import smooth

plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 8.5, 'axes.spines.top': False, 'axes.spines.right': False,
                     'axes.titlesize': 9, 'legend.frameon': False})
C_CE, C_TRAIN, C_NET, C_REFIT, C_AH, C_FOUR, C_RESTR, C_GREY, C_BAND = PX.C_CE, PX.C_TRAIN, PX.C_NET, PX.C_REFIT, PX.C_AH, \
    PX.C_FOUR, PX.C_RESTR, PX.C_GREY, PX.C_BAND
C_GAIN = '#444444'
SET_COL = {'X_R5': '#888888', 'G2': '#4A90C8', 'G3': '#08306B'}
kfmt = FuncFormatter(lambda x, _: f'{x / 1000:g}k')


def title_of(name, R):
    c = R[0]['cfg']
    n = f'{len(R)} seed' + ('s' if len(R) != 1 else '')
    if name == 'X_R5': return f'Campaign X R5: no gain ({n})'
    tau = 1 / (c['eta_g'] * c['lam_g'])
    return f'{name}: gain {c["gain_init"]:g} -> 1, time constant {tau / 1000:g}k ({n})'


def load_groups(A, ydir, xdir):
    G = {}
    for name, v in A['groups'].items():
        G[name] = []
        for r in v['runs']:
            base = xdir if r['id'].startswith('X_') else ydir
            G[name].append(dict(PX.load(os.path.join(base, r['id'], 'run.json.gz')), id=r['id']))
    return G


def order(G): return [g for g in ('X_R5', 'G2', 'G3') if g in G]


def fig_main(A, G, out):
    names = order(G)
    fig, axs = plt.subplots(4, len(names), figsize=(4.4 * len(names), 9.0), squeeze=False, sharex='col',
                            gridspec_kw=dict(height_ratios=[1.1, 1.5, 0.75, 0.6]))
    for j, name in enumerate(names):
        R = G[name]; S = A['groups'][name]; T = min(len(d['loss']) for d in R) - 1
        hv = np.array(sorted({o['step'] for o in R[0]['obs'] if 'fref_top8_test_acc_nm' in o}), float); hv = hv[hv <= T]
        lg = np.arange(0, T + 1, 20, dtype=float)
        ax = axs[0, j]
        cms = np.array([smooth(np.asarray(d['loss'][:T + 1], float)) for d in R]); x = np.arange(T + 1)
        for cm in cms: ax.plot(x, cm, color=C_CE, lw=0.6, alpha=0.3)
        mcm = np.median(cms, 0); ax.plot(x, mcm, color=C_CE, lw=1.8, label='Training CE (201-update average; median bold)')
        start = int(np.median([r['first_fit'] for r in S['runs'] if r['first_fit'] is not None]))
        ax.set_ylim(0, 1.15 * float(np.nanmax(cms[:, start:]))); ax.set_ylabel('Training CE (nats)')
        rs = [r['rise'] for r in S['runs'] if 'rise' in r]; sub = ''
        if rs:
            ch = np.median([x_['ce_change'] for x_ in rs]) * 100; up = sum(x_['ce_change'] > 0 for x_ in rs)
            sub = f'\nCE from held-out 0.1 to 0.9: {ch:+.0f}% (median), up in {up}/{len(rs)}'
        ax.set_title(title_of(name, R) + sub)
        W = [r['paper_window']['W'] for r in S['runs'] if r.get('paper_window', {}).get('W')]
        if W:
            for a_ in axs[:, j]: a_.axvspan(np.median([w[0] for w in W]), np.median([w[1] for w in W]), color=C_BAND, zorder=0, lw=0)
        ax = axs[1, j]
        PX.med_band(ax, lg, PX.on_grid(R, 'train_acc', lg), C_TRAIN, '--', 1.0, 'Training accuracy', band=False)
        PX.med_band(ax, lg, PX.on_grid(R, 'test_acc_nm', lg), C_NET, '-', 2.0, 'Network, held-out (no-mirror pairs)', z=5)
        PX.med_band(ax, hv, PX.on_grid(R, 'fref_all_test_acc_nm', hv), C_REFIT, ':', 2.0, "Paper's refit (all frequencies)", band=False)
        PX.med_band(ax, hv, PX.on_grid(R, 'fref_top8_test_acc_nm', hv), C_FOUR, '-', 1.8, 'Fourier-filtered refit (top 8 frequencies)')
        PX.med_band(ax, hv, PX.on_grid(R, 'fnet_top8_test_acc_nm', hv), C_RESTR, '-', 1.3, "Network's logits restricted to the top 8",
                    band=False)
        ax.set_ylim(-0.02, 1.02); ax.set_ylabel('Accuracy'); ax.grid(alpha=0.25, lw=0.5)
        ax = axs[2, j]
        PX.med_band(ax, hv, PX.on_grid(R, 'AH_full_resid', hv), C_AH, '-', 1.6, 'Head-input Fourier $A_H$')
        ax.set_ylabel('$A_H$'); ax.set_ylim(0, 1)
        ax = axs[3, j]
        if any('gain' in o for o in R[0]['obs']):
            PX.med_band(ax, lg, PX.on_grid(R, 'gain', lg), C_GAIN, '-', 1.4, 'Gain g on the RMS head input', band=False)
        else:
            ax.plot([0, T], [1, 1], color=C_GAIN, lw=1.4)
        ax.set_ylim(0.8, 3.1); ax.set_ylabel('Gain g'); ax.set_xlabel('Updates'); ax.xaxis.set_major_formatter(kfmt)
    h, l = [], []
    for ax in axs[:, -1]:
        hh, ll = ax.get_legend_handles_labels(); h += hh; l += ll
    h.append(plt.Rectangle((0, 0), 1, 1, color=C_BAND)); l.append("Paper's window: CE never > 3% below its running max (medians)")
    fig.legend(h, l, loc='lower center', ncol=3, fontsize=7.2, bbox_to_anchor=(0.5, 0.0))
    fig.subplots_adjust(left=0.08, right=0.98, top=0.95, bottom=0.11, hspace=0.12, wspace=0.25)
    fig.savefig(out, dpi=180); plt.close(fig)


def fig_rise(A, G, out):
    fig, ax = plt.subplots(figsize=(5.6, 3.9))
    for name in order(G):
        for d, r in zip(G[name], A['groups'][name]['runs']):
            if r['net']['0.1'] is None: continue
            s, v = PX.series(d['obs'], 'test_acc_nm'); cm = smooth(np.asarray(d['loss'], float))
            a = int(r['net']['0.1']); m = (s >= a) & (s < len(cm))
            st = s[m].astype(int); acc = np.maximum.accumulate(v[m]); keep = acc <= 0.97
            ax.plot(acc[keep], cm[st[keep]] / cm[a], color=SET_COL[name], lw=0.9, alpha=0.85)
        ax.plot([], [], color=SET_COL[name], lw=1.6, label=title_of(name, G[name]).split(' (')[0])
    ax.axhline(1, color=C_GREY, lw=0.7, ls=':')
    ax.set_xlabel('Held-out accuracy, no-mirror pairs (running max)'); ax.set_ylabel('Training CE / its value at held-out 0.1')
    ax.set_xlim(0.08, 0.98); ax.legend(fontsize=7.2, loc='upper left'); ax.grid(alpha=0.25, lw=0.5)
    ax.set_title('Training CE across the held-out rise, every seed')
    fig.tight_layout(); fig.savefig(out, dpi=180); plt.close(fig)


def fig_freq(A, G, out):
    cols = [(title_of(n, G[n]).split(' (')[0], G[n], A['groups'][n], 'test_acc_nm', '_test_acc_nm') for n in order(G)]
    median_seed, on_grid, med_band = PX.median_seed, PX.on_grid, PX.med_band
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


def main(ana, ydir, xdir, outdir):
    A = json.load(open(ana)); G = load_groups(A, ydir, xdir); os.makedirs(outdir, exist_ok=True)
    for stem, f in (('Y_main', fig_main), ('Y_rise', fig_rise), ('Y_freq', fig_freq)):
        f(A, G, os.path.join(outdir, stem + '.png')); print('drew', stem, flush=True)


if __name__ == '__main__':
    main(*sys.argv[1:5])
