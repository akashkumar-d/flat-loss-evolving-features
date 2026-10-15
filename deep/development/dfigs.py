import numpy as np, math, glob
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from anat import load, series, knee_step, first_persistent
from plateau import onset
from figs import C, cm, RAMP
def layer_cols(D):
    idx = {1: [2], 2: [1, 3], 3: [0, 2, 3], 4: [0, 1, 2, 3], 5: [0, 1, 2, 3, 4]}[D]
    return [RAMP[i] for i in idx]
def grok_window(obs, T):
    from dtable import first_cross
    t10 = first_cross(obs, 'test_acc', 0.1); t90 = first_persistent(obs, 'test_acc', 0.9)
    return t10, (t90 if t90 else T)
def deep_headline(fns, out, title, ymax_agop=0.7, band=(1, 3), flat_tol=0.03, win='longest', h1_top=None, zoom_x=None, shade_span=None, shade_label=None):
    runs = []
    for fn in fns:
        d, L, obs, st = load(fn); T = len(L) - 1; g0, g1 = grok_window(obs, T); runs.append((d, L, obs, g0, g1))
    c0 = runs[0][0]['cfg']; p = c0['p']; r = p - 1; D = c0['depth']; T = len(runs[0][1]) - 1
    G0 = int(np.median([x[3] for x in runs])); G1 = int(np.median([x[4] for x in runs]))
    if shade_span: G0, G1 = shade_span
    shade = lambda a: a.axvspan(G0, G1, color=C['blue'], alpha=0.07, lw=0)
    fig, ax = plt.subplots(2, 3, figsize=(15, 8.4)); lc = layer_cols(D)
    a = ax[0, 0]
    for d, L, obs, g0, g1 in runs: a.semilogy(np.arange(len(L) - 1), cm(L), color=C['blue'], lw=1.0, alpha=0.8)
    shade(a); a.axhline(math.log(p), color=C['muted'], lw=0.8); a.text(T, math.log(p) * 1.06, 'log p (uniform prediction)', ha='right', va='bottom', color=C['ink2'], fontsize=8)
    a.set_xlabel('step'); a.set_title(f'a  Training cross-entropy (cycle mean), {len(runs)} run(s)\n    shaded: ' + (shade_label or 'test accuracy 0.1 → 0.9 (median over runs)'), loc='left', fontsize=10)
    a = ax[0, 1]; lens = []; lvs = []; te0 = []; te1 = []; fas = []; fbs = []; shs = []; longs = []
    from flatwin import longest_flat, best_grok_window
    for d, L, obs, g0, g1 in runs:
        c_ = cm(L); k = knee_step(d, obs) or 0; la, lb = longest_flat(c_, k, flat_tol); s_, v = series(obs, 'test_acc')
        if win == 'grok':
            fa, fb, sh = best_grok_window(c_, s_, v, k, flat_tol); shs.append(sh)
        else:
            fa, fb = la, lb
        seg = c_[fa:fb + 1]; mu = seg.mean(); lvs.append(mu); lens.append(fb - fa); fas.append(fa); fbs.append(fb)
        te0.append(v[np.argmin(np.abs(s_ - fa))]); te1.append(v[np.argmin(np.abs(s_ - fb))])
        a.plot(np.arange(len(c_)), 100 * (c_ / mu - 1), color=C['blue'], lw=1.0, alpha=0.85)
        a.plot([fa, fb], [-band[1] - 1.5] * 2, color=C['ink2'], lw=2.0, solid_capstyle='butt')
        if win == 'grok' and (la, lb) != (fa, fb):
            a.plot([la, lb], [-band[1] - 3.0] * 2, color=C['muted'], lw=2.0, solid_capstyle='butt'); longs.append((la, lb, c_[la:lb + 1].mean() / math.log(p), v[np.argmin(np.abs(s_ - la))]))
    shade(a); a.axhspan(-band[1], band[1], color=C['blue'], alpha=0.06, lw=0); a.axhspan(-band[0], band[0], color=C['blue'], alpha=0.08, lw=0)
    xl = zoom_x or (max(0, min(G0, min(fas)) - 4000), min(T, max(G1, max(fbs)) + 6000))
    a.set_ylim(-12, 12); a.set_xlim(*xl); a.set_xlabel('step')
    if win == 'grok':
        a.set_ylabel('% from mean CE over the black-bar window')
        out_ = [x for x in longs if x[0] > xl[1] or x[1] < xl[0]]
        note = ''
        if out_:
            la, lb, lv, t0 = out_[0]
            note = f'\n(off this view{" in %d runs" % len(out_) if len(out_) > 1 else ""}: e.g. {la:,}–{lb:,} at {lv:.2f}·log p, test {t0:.2f} at its start)'
        a.text(0.01, 0.97, ('black bar: the window with CE range ≤ %.0f%% that holds the largest\nshare of the test 0.1 → 0.9 rise (chosen knowing test accuracy)\ngrey bar: longest such window, from the loss alone' % (100 * flat_tol)) + note, transform=a.transAxes, fontsize=7.5, color=C['ink2'], va='top')
        sh_s = f'{100 * min(shs):.0f}–{100 * max(shs):.0f}%' if len(shs) > 1 else f'{100 * shs[0]:.0f}%'
        lens_s = f'{min(lens):,}–{max(lens):,} steps' if len(lens) > 1 else f'{lens[0]:,} steps'
        a.set_title(f'b  Zoom (linear), bands ±{band[0]}%, ±{band[1]}%: {lens_s} at {np.mean(lvs) / math.log(p):.2f}·log p\n    hold {sh_s} of the rise; test {np.median(te0):.2f} → {np.median(te1):.2f} inside', loc='left', fontsize=10)
    else:
        a.set_ylabel('% from mean CE over each run\'s flat window')
        a.text(0.01, 0.97, 'black bars: longest window with CE range ≤ %.0f%%\n(found from the loss alone)' % (100 * flat_tol), transform=a.transAxes, fontsize=7.5, color=C['ink2'], va='top')
        lens_s = f'flat windows of {min(lens):,}–{max(lens):,} steps' if len(lens) > 1 else f'a flat window of {lens[0]:,} steps'
        a.set_title(f'b  Zoom (linear), bands ±{band[0]}%, ±{band[1]}%: {lens_s}\n    at {np.mean(lvs) / math.log(p):.2f}·log p; test {np.median(te0):.2f} → {np.median(te1):.2f} inside', loc='left', fontsize=10)
    a = ax[0, 2]
    for d, L, obs, g0, g1 in runs:
        for key, col in [('train_acc', C['blue']), ('test_acc', C['orange']), ('refit_acc', C['aqua'])]:
            s_, v = series(obs, key); a.plot(s_, v, color=col, lw=1.0, alpha=0.85)
    shade(a); a.plot([], [], color=C['blue'], label='train'); a.plot([], [], color=C['orange'], label='test (network)'); a.plot([], [], color=C['aqua'], label='test, ridge refit on\nlast hidden layer (train-only)')
    a.set_ylim(-0.02, 1.02); a.set_xlabel('step'); a.legend(loc='lower right', fontsize=8); a.set_title('c  Accuracy', loc='left', fontsize=10)
    a = ax[1, 0]
    for d, L, obs, g0, g1 in runs:
        s_, v = series(obs, f'agop{r}_ho'); a.plot(s_, v, color=C['violet'], lw=1.2, alpha=0.9)
        s_, v = series(obs, f'agop{r}_ho_rand'); a.plot(s_, v, color=C['violet'], lw=1.0, alpha=0.7, ls=(0, (3, 2)))
        if D >= 2:
            s_, v = series(obs, f'agopH1_{h1_top or r}_ho'); a.plot(s_, v, color=C['aqua'], lw=1.2, alpha=0.9)
            s_, v = series(obs, f'agopH1_{h1_top or r}_ho_rand'); a.plot(s_, v, color=C['aqua'], lw=1.0, alpha=0.7, ls=(0, (3, 2)))
    shade(a); a.plot([], [], color=C['violet'], label=f'last hidden layer (AGOP = VᵀV), top-{r}')
    if D >= 2: a.plot([], [], color=C['aqua'], label=f'first hidden layer (AGOP through deeper layers), top-{h1_top or r}')
    a.plot([], [], color=C['muted'], ls=(0, (3, 2)), label='dashed: random frames of the same size (control)'); a.legend(loc='upper left', fontsize=8)
    a.set_ylim(0, ymax_agop); a.set_xlabel('step'); a.set_title('d  AGOP–Fourier alignment on held-out pairs\n    (mean squared canonical corr. with the addition-Fourier basis)', loc='left', fontsize=10)
    a = ax[1, 1]
    for d, L, obs, g0, g1 in runs:
        for l in range(D):
            s_, v = series(obs, f'AH_full_h{l + 1}'); a.plot(s_, v, color=lc[l], lw=1.2, alpha=0.9)
        s_, v = series(obs, 'AH_full_logit'); a.plot(s_, v, color=C['green'], lw=1.2, alpha=0.9)
        s_, v = series(obs, 'W1_four_top1')
        if len(v): a.plot(s_, v, color=C['yellow'], lw=1.2, alpha=0.9)
    shade(a)
    for l in range(D): a.plot([], [], color=lc[l], label=f'A_H layer {l + 1}')
    a.plot([], [], color=C['green'], label='A_H logits'); a.plot([], [], color=C['yellow'], label='layer-1 weight Fourier share')
    a.set_ylim(0, 1); a.set_xlabel('step'); a.legend(loc='upper left', fontsize=8); a.set_title('e  Fourier energy', loc='left', fontsize=10)
    a = ax[1, 2]
    for d, L, obs, g0, g1 in runs:
        hv = [o for o in obs if 'normW' in o]; s_ = [o['step'] for o in hv]
        for l in range(D): a.semilogy(s_, [o['normW'][l] for o in hv], color=lc[l], lw=1.2, alpha=0.9)
        a.semilogy(s_, [o['normV'] for o in hv], color=C['orange'], lw=1.2, alpha=0.9)
    shade(a)
    for l in range(D): a.plot([], [], color=lc[l], label=f'‖W{l + 1}‖')
    a.plot([], [], color=C['orange'], label='‖V‖ (head) → 1/λ_V'); a.legend(fontsize=8, loc='lower right'); a.set_xlabel('step'); a.set_title('f  Weight norms (Frobenius)', loc='left', fontsize=10)
    fig.suptitle(title, fontsize=10.5, x=0.01, ha='left', color=C['ink'])
    fig.tight_layout(rect=(0, 0, 1, 0.94)); fig.savefig(out, dpi=130); plt.close(fig)
def width_map(items, out, title):
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.4))
    cols = RAMP[1:1 + len(items)] if len(items) <= 4 else RAMP
    for (fn, lab), col in zip(items, cols):
        d, L, obs, st = load(fn); p = d['cfg']['p']; r = p - 1; t = np.arange(len(L) - 1)
        ax[0].plot(t, cm(L), color=col, lw=1.6, label=lab)
        s_, v = series(obs, 'test_acc'); ax[1].plot(s_, v, color=col, lw=1.6, label=lab)
        s_, v = series(obs, f'agop{r}_ho'); ax[2].plot(s_, v, color=col, lw=1.6, label=lab)
    ax[0].set_title('a  Training cross-entropy (cycle mean, linear)', loc='left', fontsize=10); ax[0].set_xlabel('step')
    ax[1].set_title('b  Test accuracy (network)', loc='left', fontsize=10); ax[1].set_xlabel('step'); ax[1].set_ylim(-0.02, 1.02)
    ax[2].set_title(f'c  AGOP–Fourier alignment, held-out (top-{r})', loc='left', fontsize=10); ax[2].set_xlabel('step'); ax[2].set_ylim(0, 0.7)
    ax[1].legend(fontsize=8, loc='lower right')
    fig.suptitle(title, fontsize=10.5, x=0.01, ha='left'); fig.tight_layout(rect=(0, 0, 1, 0.9)); fig.savefig(out, dpi=130); plt.close(fig)
if __name__ == '__main__':
    width_map([('runs/h1_p31_w128_lw0.015.json', 'width 128'), ('runs/h1_p31_w256_lw0.015.json', 'width 256'), ('runs/s2_p31_lw0.015_lv0.2.json', 'width 512')],
              'fig_width_1hl_p31.png', 'One hidden layer, p = 31, λ_W = 0.015, λ_V = 0.2, same seed: the memorization bump shrinks with width while the final level does not;\nthe plateau is flat only when the two meet (width 512).')
    print('ok')

def gain_panels(items, out, title):
    fig, ax = plt.subplots(1, len(items), figsize=(4.6 * len(items), 4.0), sharey=True)
    for a, (fn, lab) in zip(ax, items):
        d, L, obs, st = load(fn); T = len(L) - 1; g0, g1 = grok_window(obs, T)
        hv = [o for o in obs if 'normW' in o]; s = np.array([o['step'] for o in hv])
        prod = np.array([o['normV'] * np.prod(o['normW']) for o in hv]); lr = np.array([o['logit_rms'] for o in hv]); gain = lr / prod
        i0 = int(np.argmin(np.abs(s - g0)))
        a.axvspan(g0, g1, color=C['blue'], alpha=0.07, lw=0); a.axhline(1, color=C['axis'], lw=0.8)
        a.plot(s, prod / prod[i0], color=C['blue'], lw=1.6, label='product of weight norms')
        a.plot(s, gain / gain[i0], color=C['orange'], lw=1.6, label='normalized gain (logit RMS / product)')
        a.plot(s, lr / lr[i0], color=C['aqua'], lw=1.6, label='logit RMS')
        a.set_ylim(0.6, 1.6); a.set_xlabel('step'); a.set_title(lab, loc='left', fontsize=10)
        a.set_xlim(max(0, g0 - 6000), min(T, g1 + 8000))
    ax[0].set_ylabel('value / value at test accuracy 0.1'); ax[-1].legend(fontsize=8, loc='upper right')
    fig.suptitle(title, fontsize=10.5, x=0.01, ha='left'); fig.tight_layout(rect=(0, 0, 1, 0.9)); fig.savefig(out, dpi=130); plt.close(fig)

def ratio_dial(items, out, title, xmax=40000, colors=None, legend_loc='upper left'):
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.4))
    colors = colors or RAMP[:len(items)]; assert len(colors) >= len(items)
    for (fn, lab), col in zip(items, colors):
        d, L, obs, st = load(fn); t = np.arange(len(L) - 1)
        ax[0].plot(t, cm(L), color=col, lw=1.6, label=lab)
        s_, v = series(obs, 'test_acc'); ax[1].plot(s_, v, color=col, lw=1.6, label=lab)
        s_, v = series(obs, 'agopH1_4_ho'); ax[2].plot(s_, v, color=col, lw=1.6, label=lab)
    s_, v = series(obs, 'agopH1_4_ho_rand'); ax[2].plot(s_, v, color=C['muted'], lw=1.0, ls=(0, (3, 2)), label='random frames')
    ax[0].set_ylim(0, 2.2); ax[0].set_title('a  Training cross-entropy (cycle mean, linear)', loc='left', fontsize=10)
    ax[1].set_ylim(-0.02, 1.02); ax[1].set_title('b  Test accuracy (network)', loc='left', fontsize=10)
    ax[2].set_ylim(0, 0.6); ax[2].set_title('c  First-layer AGOP–Fourier alignment (top-4, held-out)', loc='left', fontsize=10)
    for a in ax: a.set_xlabel('step'); a.set_xlim(0, xmax)
    ax[1].legend(fontsize=8, loc=legend_loc)
    fig.suptitle(title, fontsize=10.5, x=0.01, ha='left'); fig.tight_layout(rect=(0, 0, 1, 0.9)); fig.savefig(out, dpi=130); plt.close(fig)
