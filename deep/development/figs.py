import json, numpy as np, glob, math
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from anat import load, series, knee_step
from plateau import onset
C = dict(blue='#2a78d6', orange='#eb6834', aqua='#1baf7a', yellow='#eda100', magenta='#e87ba4', green='#008300', violet='#4a3aa7', red='#e34948',
         ink='#0b0b0b', ink2='#52514e', muted='#898781', grid='#e1e0d9', axis='#c3c2b7', surface='#fcfcfb')
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9, 'axes.edgecolor': C['axis'], 'axes.labelcolor': C['ink2'], 'xtick.color': C['muted'],
                     'ytick.color': C['muted'], 'axes.titlecolor': C['ink'], 'axes.grid': True, 'grid.color': C['grid'], 'grid.linewidth': 0.6,
                     'axes.spines.top': False, 'axes.spines.right': False, 'figure.facecolor': C['surface'], 'axes.facecolor': C['surface'],
                     'legend.frameon': False, 'lines.linewidth': 1.6})
def cm(L): return (L[:-1] + L[1:]) / 2
def shade(a, t0, t1): a.axvspan(t0, t1, color=C['blue'], alpha=0.07, lw=0)
def med(xs):
    return np.median(np.stack([v for _, v in xs]), 0)
def headline(fns, out, title, tol=0.02, span=5000):
    runs = []
    for fn in fns:
        d, L, obs, st = load(fn); k = knee_step(d, obs); tp = onset(L, k, span, tol); runs.append((d, L, obs, k, tp))
    p = runs[0][0]['cfg']['p']; r = p - 1; T = len(runs[0][1]) - 1
    tp_med = int(np.median([x[4] for x in runs]))
    fig, ax = plt.subplots(2, 3, figsize=(15, 8.2))
    a = ax[0, 0]
    for d, L, obs, k, tp in runs: a.semilogy(np.arange(len(L) - 1), cm(L), color=C['blue'], lw=1.0, alpha=0.75)
    shade(a, tp_med, T); a.set_title('a  Training cross-entropy (cycle mean), all seeds', loc='left', fontsize=10)
    a.set_xlabel('step'); a.axhline(math.log(p), color=C['muted'], lw=0.8); a.text(T, math.log(p) * 1.05, 'log p (uniform prediction)', ha='right', va='bottom', color=C['ink2'], fontsize=8)
    a.text(tp_med + 300, a.get_ylim()[0] * 1.3 if a.get_ylim()[0] > 0 else 0.5, 'detected plateau', color=C['ink2'], fontsize=8)
    a = ax[0, 1]
    for d, L, obs, k, tp in runs:
        c_ = cm(L); seg = c_[tp:]; a.plot(np.arange(tp, len(c_)), 100 * (seg / seg.mean() - 1), color=C['blue'], lw=1.0, alpha=0.8)
    a.axhspan(-3, 3, color=C['blue'], alpha=0.06, lw=0); a.axhspan(-1, 1, color=C['blue'], alpha=0.08, lw=0)
    a.set_ylim(-6, 6); a.set_xlim(0, T); a.set_ylabel('% from plateau mean'); a.set_xlabel('step')
    lv = np.mean([cm(L)[tp:].mean() for d, L, obs, k, tp in runs]); Rs = [100 * (cm(L)[tp:].max() - cm(L)[tp:].min()) / cm(L)[tp:].mean() for d, L, obs, k, tp in runs]
    a.set_title(f'b  Inside the plateau (linear): mean CE {lv:.2f} = {lv / math.log(p):.2f}·log p\n    range {min(Rs):.1f}–{max(Rs):.1f}% over {T - tp_med:,} steps; bands ±1%, ±3%', loc='left', fontsize=10)
    a = ax[0, 2]
    for d, L, obs, k, tp in runs:
        s_, v = series(obs, 'train_acc'); a.plot(s_, v, color=C['blue'], lw=1.0, alpha=0.7)
        s_, v = series(obs, 'test_acc'); a.plot(s_, v, color=C['orange'], lw=1.0, alpha=0.8)
        s_, v = series(obs, 'refit_acc'); a.plot(s_, v, color=C['aqua'], lw=1.0, alpha=0.8)
    shade(a, tp_med, T); a.set_ylim(-0.02, 1.02); a.set_xlabel('step'); a.set_title('c  Accuracy', loc='left', fontsize=10)
    a.plot([], [], color=C['blue'], label='train'); a.plot([], [], color=C['orange'], label='test (network)'); a.plot([], [], color=C['aqua'], label='test (train-only ridge refit)')
    a.legend(loc='center right', fontsize=8)
    a = ax[1, 0]
    for d, L, obs, k, tp in runs:
        s_, v = series(obs, f'agop{r}_ho'); a.plot(s_, v, color=C['violet'], lw=1.0, alpha=0.8)
        s_, v = series(obs, f'agop{r}_ho_rand'); a.plot(s_, v, color=C['muted'], lw=1.0, alpha=0.8)
    shade(a, tp_med, T); a.set_ylim(0, 0.6); a.set_xlabel('step')
    a.plot([], [], color=C['violet'], label=f'top-{r} AGOP directions'); a.plot([], [], color=C['muted'], label=f'random {r}-frames (control)'); a.legend(loc='upper left', fontsize=8)
    a.set_title('d  AGOP–Fourier alignment on held-out pairs\n    (mean squared canonical corr. with addition-Fourier basis)', loc='left', fontsize=10)
    a = ax[1, 1]
    for d, L, obs, k, tp in runs:
        s_, v = series(obs, 'AH_full_logit'); a.plot(s_, v, color=C['green'], lw=1.0, alpha=0.8)
        s_, v = series(obs, 'AH_full_h1'); a.plot(s_, v, color=C['yellow'], lw=1.2, alpha=0.9)
    shade(a, tp_med, T); a.set_ylim(0, 0.8); a.set_xlabel('step')
    a.plot([], [], color=C['green'], label='logits'); a.plot([], [], color=C['yellow'], label='hidden layer'); a.legend(loc='upper left', fontsize=8)
    a.set_title('e  Addition-Fourier energy fraction A_H (full grid)', loc='left', fontsize=10)
    a = ax[1, 2]
    for d, L, obs, k, tp in runs:
        hv = [o for o in obs if 'normW' in o]; s_ = [o['step'] for o in hv]
        a.semilogy(s_, [o['normW'][0] for o in hv], color=C['blue'], lw=1.0, alpha=0.8)
        a.semilogy(s_, [o['normV'] for o in hv], color=C['orange'], lw=1.0, alpha=0.8)
    shade(a, tp_med, T); a.set_xlabel('step'); a.plot([], [], color=C['blue'], label='‖W‖ (hidden)'); a.plot([], [], color=C['orange'], label='‖V‖ (head) → 1/λ_V'); a.legend(fontsize=8)
    a.set_title('f  Weight norms: ‖W‖ is still relaxing through grokking;\n    ‖V‖ sits at its cap 1/λ_V', loc='left', fontsize=10)
    fig.suptitle(title, fontsize=11, x=0.01, ha='left', color=C['ink'])
    fig.tight_layout(rect=(0, 0, 1, 0.95)); fig.savefig(out, dpi=130); plt.close(fig)
if __name__ == '__main__':
    fns = sorted(glob.glob('runs/seed_p31_lw0.015_lv0.2_s*.json'))
    headline(fns, 'fig_headline_p31.png',
             'p = 31, one hidden layer (ReLU, width 512), cross-entropy, 80% split; all layers trained throughout with fixed hyperparameters — '
             'hidden: support-Polar/√rank step 0.03, decay 0.015; head: normalized step 0.03, decay 0.2. Five fresh seeds. No rescale, freeze or controller.')
    print('ok')

RAMP = ['#86b6ef', '#5598e7', '#2a78d6', '#1c5cab', '#104281']
def decay_dial(items, out, title):
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.4))
    for (fn, lab), col in zip(items, RAMP):
        d, L, obs, st = load(fn); p = d['cfg']['p']; r = p - 1; t = np.arange(len(L) - 1)
        ax[0].semilogy(t, cm(L), color=col, lw=1.6, label=lab)
        ax[0].text(t[-1] + 300, cm(L)[-1], lab.split(' ')[0].replace('λ_W=', ''), color=C['ink2'], fontsize=7, va='center')
        s_, v = series(obs, 'test_acc'); ax[1].plot(s_, v, color=col, lw=1.6, label=lab)
        s_, v = series(obs, f'agop{r}_ho'); ax[2].plot(s_, v, color=col, lw=1.6, label=lab)
    s_, v = series(obs, f'agop{r}_ho_rand'); ax[2].plot(s_, v, color=C['muted'], lw=1.0, label='random frames')
    ax[0].axhline(math.log(p), color=C['muted'], lw=0.8)
    ax[0].set_title('a  Training cross-entropy (cycle mean)', loc='left', fontsize=10); ax[0].set_xlabel('step'); ax[0].set_xlim(0, 32500)
    ax[1].set_title('b  Test accuracy (network)', loc='left', fontsize=10); ax[1].set_xlabel('step'); ax[1].set_ylim(-0.02, 1.02)
    ax[2].set_title(f'c  AGOP–Fourier alignment, held-out (top-{r})', loc='left', fontsize=10); ax[2].set_xlabel('step'); ax[2].set_ylim(0, 0.7)
    ax[1].legend(fontsize=8, loc='lower right', title='hidden-layer decay', title_fontsize=8)
    fig.suptitle(title, fontsize=10.5, x=0.01, ha='left'); fig.tight_layout(rect=(0, 0, 1, 0.92)); fig.savefig(out, dpi=130); plt.close(fig)

def mechanism(fn, out, title):
    d, L, obs, st = load(fn); p = d['cfg']['p']; r = p - 1; k = knee_step(d, obs); tp = onset(L, k, 5000, 0.02); T = len(L) - 1
    hv = [o for o in obs if 'normW' in o]; s = np.array([o['step'] for o in hv]); m = s >= tp
    base = lambda v: np.array(v)[m] / np.array(v)[m][0]
    ce = np.array([o['loss'] for o in hv]); learned = math.log(p) - ce
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.4))
    a = ax[0]
    for vals, col, lab in [(ce, C['blue'], 'training CE'), (learned, C['aqua'], 'learned part, log p − CE'),
                           ([o['logit_rms'] for o in hv], C['orange'], 'logit scale (RMS over the grid)'),
                           ([o['train_margin_min'] for o in hv], C['violet'], 'smallest training margin'),
                           ([o['normW'][0] for o in hv], C['yellow'], '‖W‖ hidden layer')]:
        a.plot(s[m], base(vals), color=col, lw=1.8, label=lab)
    a.axhline(1, color=C['axis'], lw=0.8); a.set_ylim(0.5, 1.6); a.set_xlabel('step'); a.set_ylabel('value / value at plateau onset')
    a.legend(fontsize=8, loc='center right', bbox_to_anchor=(1.0, 0.65)); a.set_title('a  What stays fixed and what moves after the plateau onset', loc='left', fontsize=10)
    a = ax[1]
    for key, col, lab in [('test_acc', C['orange'], 'test accuracy'), ('refit_acc', C['aqua'], 'train-only ridge refit (test)'), (f'agop{r}_ho', C['violet'], f'AGOP–Fourier alignment (top-{r}, held-out)'),
                          ('AH_full_logit', C['green'], 'logit A_H')]:
        s_, v = series(obs, key); a.plot(s_[s_ >= tp], v[s_ >= tp], color=col, lw=1.8, label=lab)
    s_, v = series(obs, f'agop{r}_ho_rand'); a.plot(s_[s_ >= tp], v[s_ >= tp], color=C['muted'], lw=1.2, label='random-frame control')
    a.set_ylim(0, 1.02); a.set_xlabel('step'); a.legend(fontsize=8, loc='center right', bbox_to_anchor=(1.0, 0.8)); a.set_title('b  Meanwhile: generalization and Fourier structure', loc='left', fontsize=10)
    fig.suptitle(title, fontsize=10.5, x=0.01, ha='left'); fig.tight_layout(rect=(0, 0, 1, 0.92)); fig.savefig(out, dpi=130); plt.close(fig)

def controls(items, out, title, t_switch=5200):
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.4))
    cols = [C['blue'], C['orange'], C['aqua'], C['yellow']]
    for (fn, lab), col in zip(items, cols):
        d, L, obs, st = load(fn); p = d['cfg']['p']; r = p - 1; t = np.arange(len(L) - 1)
        ax[0].semilogy(t, np.maximum(cm(L), 1e-20), color=col, lw=1.6, label=lab)
        s_, v = series(obs, 'test_acc'); ax[1].plot(s_, v, color=col, lw=1.6, label=lab)
        s_, v = series(obs, f'agop{r}_ho'); ax[2].plot(s_, v, color=col, lw=1.6, label=lab)
    for a in ax: a.axvline(t_switch, color=C['axis'], lw=1.0)
    ax[0].set_ylim(1e-20, 10); ax[0].set_title('a  Training cross-entropy (cycle mean)', loc='left', fontsize=10); ax[0].set_xlabel('step')
    ax[1].set_title('b  Test accuracy (network)', loc='left', fontsize=10); ax[1].set_xlabel('step'); ax[1].set_ylim(-0.02, 1.02)
    ax[2].set_title(f'c  AGOP–Fourier alignment, held-out (top-{r})', loc='left', fontsize=10); ax[2].set_xlabel('step'); ax[2].set_ylim(0, 0.6)
    ax[1].legend(fontsize=8, loc='center right'); ax[0].text(t_switch + 300, 3e-19, 'switch (step 5,200)', color=C['ink2'], fontsize=8)
    fig.suptitle(title, fontsize=10.5, x=0.01, ha='left'); fig.tight_layout(rect=(0, 0, 1, 0.92)); fig.savefig(out, dpi=130); plt.close(fig)
