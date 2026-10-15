#!/usr/bin/env python3
import json, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 8.5, 'axes.spines.top': False, 'axes.spines.right': False})
NET = '#D55E00'
LINES = [
    ('ridge', 'Paper refit (all frequencies)', '#CC79A7', ':', 2.0),
    ('filt_top8', 'Fourier-filtered refit, top 8 frequencies', '#56B4E9', '-', 1.7),
    ('filt_top4', 'Fourier-filtered refit, top 4 frequencies', '#56B4E9', '--', 1.3),
    ('phase_top8', 'Phase vote, top 8 frequencies', '#E69F00', '-', 1.2),
    ('net_filt_top8', "Network's own logits restricted to the top 8", '#009E73', '-', 1.2),
    ('probe', 'Logistic probe (all frequencies)', '#444444', '-.', 1.2),
]


def panel(axA, axB, sc, title):
    tf = sc['kind'] == 'tf'; tab = 'resid' if tf else 'last'; key = '_te_nm' if tf else '_te'
    obs = sc['obs']; to = np.array([o['step'] for o in obs]) / 1e3
    vo = np.array([o.get('test_acc_nm' if tf else 'test_acc', np.nan) for o in obs])
    rows = sc['rows']; ts = np.array([r['step'] for r in rows]) / 1e3
    for k, lab, col, ls, lw in LINES:
        v = np.array([r[tab].get(k + key, np.nan) for r in rows])
        if np.all(np.isnan(v)): continue
        axA.plot(ts, v, color=col, ls=ls, lw=lw, label=lab)
    axA.plot(to, vo, color=NET, lw=2.0, label='Network (transformers: no-mirror held-out pairs)', zorder=5)
    axA.set_ylim(-0.02, 1.02); axA.set_ylabel('Held-out accuracy'); axA.set_title(title, fontsize=9)
    axA.grid(alpha=0.25, lw=0.5)
    R2 = np.clip(np.array([r[tab]['ho_r2_spec'] for r in rows]).T, 0, 1)
    K = R2.shape[0]; dt = ts[1] - ts[0]
    im = axB.imshow(R2, aspect='auto', origin='lower', extent=[ts[0] - dt / 2, ts[-1] + dt / 2, 0.5, K + 0.5],
                    cmap='Blues', vmin=0, vmax=1, interpolation='nearest')
    for j, r in enumerate(rows):
        ks = r[tab]['info']['order'][:8]
        axB.scatter([ts[j]] * len(ks), ks, s=2.5, color=NET, marker='o', lw=0)
    axB.set_ylabel('Frequency k'); axB.set_xlabel('Updates (thousands)')
    axB.yaxis.set_major_locator(MaxNLocator(integer=True, nbins=6))
    axA.set_xlim(axB.get_xlim())
    return im


def main():
    out = sys.argv[1]; items = [a.rsplit('=', 1) for a in sys.argv[2:]]
    fig, axs = plt.subplots(2, len(items), figsize=(max(4.6 * len(items), 9.5), 6.2), squeeze=False,
                            gridspec_kw=dict(height_ratios=[1.4, 1]))
    for j, (title, fn) in enumerate(items):
        sc = json.load(open(fn)); im = panel(axs[0, j], axs[1, j], sc, title)
    fig.subplots_adjust(left=0.065, right=0.9, top=0.94, bottom=0.17, hspace=0.3, wspace=0.2)
    pos = axs[1, -1].get_position(); cax = fig.add_axes([0.915, pos.y0, 0.011, pos.height])
    cb = fig.colorbar(im, cax=cax)
    cb.set_label('Held-out $R^2$ of the refit\'s\nprediction of $e^{i w_k y}$', fontsize=7.5); cb.ax.tick_params(labelsize=7)
    h, l = axs[0, 0].get_legend_handles_labels()
    top = plt.Line2D([], [], color=NET, marker='o', ls='', ms=3)
    fig.legend(h + [top], l + ['Top 8 frequencies by training energy (train-only ranking)'], loc='lower center',
               ncol=3, frameon=False, fontsize=7.5, bbox_to_anchor=(0.48, 0.0))
    fig.savefig(out, dpi=170)


if __name__ == '__main__':
    main()
