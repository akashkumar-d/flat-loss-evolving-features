#!/usr/bin/env python3
import os, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', '..', 'campaign_X')); sys.path.insert(0, os.path.join(HERE, '..')); sys.path.insert(0, HERE)
import plot_X as PX
from analyze_Y import smooth
from analyze_G import load_any

plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 8, 'axes.spines.top': False, 'axes.spines.right': False,
                     'axes.titlesize': 8.5, 'legend.frameon': False})
R = os.path.join(HERE, 'runs')
RUNS = [('No gain (campaign W/X R5)', os.path.join(R, 'W_R5_s1_replay_fourier.json.gz')),
        ('Slow head, $\\eta_V$ = 0.0025', os.path.join(R, 'pY_a_etav0.0025_s1_to7000.json.gz')),
        ('Slow head, $\\eta_V$ = 0.0012', os.path.join(R, 'pY_b_etav0.0012_s1_to7000.json.gz')),
        ('Gain 2 -> 1, time constant 5k', os.path.join(R, 'pG1_g2_tau5k_s1.json.gz')),
        ('Gain 3 -> 1, time constant 4k', os.path.join(R, 'pG2_g3_tau4k_s1.json.gz'))]
C_GAIN = '#444444'


def main(out):
    T = 12000
    fig, axs = plt.subplots(4, len(RUNS), figsize=(3.3 * len(RUNS), 7.6), sharex=True, squeeze=False,
                            gridspec_kw=dict(height_ratios=[1.1, 1.4, 0.7, 0.7]))
    for j, (title, fn) in enumerate(RUNS):
        d = load_any(fn); obs = d['obs']; L = np.asarray(d['loss'], float); cm = smooth(L); x = np.arange(len(cm))
        s, v = PX.series(obs, 'test_acc_nm'); a = np.where(v >= 0.1)[0]; b = np.where(v >= 0.9)[0]
        ax = axs[0, j]; ax.plot(x, cm, color=PX.C_CE, lw=1.4); ax.set_ylim(0, 2.0); ax.set_title(title)
        if len(a) and len(b):
            for a_ in axs[:, j]: a_.axvspan(s[a[0]], s[b[0]], color=PX.C_BAND, lw=0, zorder=0)
            ch = cm[int(s[b[0]])] / cm[int(s[a[0]])] - 1
            ax.text(0.97, 0.95, f'held-out 0.1 -> 0.9: CE {100 * ch:+.0f}%', transform=ax.transAxes, ha='right', va='top', fontsize=7.5,
                    color=PX.C_CE)
        ax = axs[1, j]
        st, tr = PX.series(obs, 'train_acc'); ax.plot(st, tr, color=PX.C_TRAIN, ls='--', lw=0.9, label='Training accuracy')
        ax.plot(s, v, color=PX.C_NET, lw=1.8, label='Network, held-out (no-mirror pairs)')
        for key, col, ls, lab in (('fref_top8_test_acc_nm', PX.C_FOUR, '-', 'Fourier-filtered refit (top 8)'),
                                  ('refit_test_acc_nm', PX.C_REFIT, ':', "Paper's refit")):
            sr, vr = PX.series(obs, key); ax.plot(sr, vr, color=col, ls=ls, lw=1.6, label=lab)
        ax.set_ylim(-0.02, 1.02); ax.grid(alpha=0.25, lw=0.5)
        ax = axs[2, j]; sa, ah = PX.series(obs, 'AH_full_resid'); ax.plot(sa, ah, color=PX.C_AH, lw=1.4, label='Head-input Fourier $A_H$'); ax.set_ylim(0, 1)
        ax = axs[3, j]
        sg, g = PX.series(obs, 'gain')
        if len(sg): ax.plot(sg, g, color=C_GAIN, lw=1.4, label='Gain g')
        else: ax.plot([0, T], [1, 1], color=C_GAIN, lw=1.4)
        sn, nv = PX.series(obs, 'normV'); ax.plot(sn, nv / 3.62, color=C_GAIN, ls=':', lw=1.3, label='$\\|W_U\\|$ / 3.62')
        ax.set_ylim(0.8, 3.1); ax.set_xlabel('Updates')
        ax.xaxis.set_major_formatter(FuncFormatter(lambda x_, _: f'{x_ / 1000:g}k'))
    axs[0, 0].set_ylabel('Training CE (nats,\n201-update average)'); axs[1, 0].set_ylabel('Accuracy'); axs[2, 0].set_ylabel('$A_H$')
    axs[3, 0].set_ylabel('Scale')
    for ax in axs[:, 0]: ax.set_xlim(0, T)
    h, l = [], []
    for ax in (axs[1, 0], axs[2, 0], axs[3, -1]):
        hh, ll = ax.get_legend_handles_labels(); h += hh; l += ll
    h.append(plt.Rectangle((0, 0), 1, 1, color=PX.C_BAND)); l.append('Held-out 0.1 -> 0.9')
    fig.legend(h, l, loc='lower center', ncol=4, fontsize=7.5, bbox_to_anchor=(0.5, 0.0))
    fig.suptitle('Pilots on seed 1 of R5 (frac 0.2, $\\lambda_V$ = 0.275): keeping the training CE from falling during the held-out rise',
                 fontsize=9.5, y=0.995)
    fig.subplots_adjust(left=0.07, right=0.99, top=0.93, bottom=0.12, hspace=0.14, wspace=0.18)
    fig.savefig(out, dpi=170); plt.close(fig)


if __name__ == '__main__':
    main(sys.argv[1])
