import glob, math, os, sys
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from numpy.lib.stride_tricks import sliding_window_view
sys.path.insert(0, '/home/claude/kt/campaign_W'); import paperdefs_tf as P
sys.path.insert(0, '/home/claude/kt/campaign_L'); import endpoints_i as E
K = dict(train="#0072B2", test="#D55E00", refit="#7B3294", grid="#E6E6E6", window="#EBEBEB")
plt.rcParams.update({'font.size': 8, 'axes.spines.top': False, 'axes.spines.right': False, 'axes.edgecolor': '#555555', 'axes.linewidth': 0.7})
out = sys.argv[1]; os.makedirs(out, exist_ok=True)

def runmed(x, k=201):
    c = np.concatenate([(x[:-1] + x[1:]) / 2, x[-1:]]); h = k // 2
    return np.convolve(np.pad(c, (h, h), mode='edge'), np.ones(k) / k, mode='valid')
def first(s, v, th):
    i = np.where(v >= th)[0]; return int(s[i[0]]) if len(i) else None

def load_tf(fn):
    d = P.load(fn); key = P.te_key(d); nm = '_nm' if key == 'test_acc_nm' else ''
    s, te = P.series(d['obs'], key); sr, rf = P.series(d['obs'], 'refit_test_acc' + nm); sp, pb = P.series(d['obs'], 'probe_test_acc' + nm)
    return dict(L=np.asarray(d['loss'], float), s=s, te=te, sr=sr, rf=rf, sp=sp, pb=pb, p=P.cfg_of(d)['p'])
def load_paper(fn):
    d = E.load(fn); s, te = E.series(d['obs'], 'test_acc'); sr, rf = E.series(d['obs'], 'v1_ridge_acc_te')
    if not len(sr): sr, rf = E.series(d['obs'], 'refit_acc')
    return dict(L=np.asarray(d['loss'], float), s=s, te=te, sr=sr, rf=rf, sp=np.array([]), pb=np.array([]), p=d['cfg']['p'])

W = '/home/claude/kt/campaign_W/collected/results'
R5 = [load_tf(f'{W}/W_R5_f0.2_rms_lam_w0.01_eta_w0.1_lam_v0.275_s{i}/run.json.gz') for i in (1, 2, 3)]
R4 = [load_tf(f'{W}/W_R4_f0.2_rms_lam_w0.01_eta_w0.1_lam_v0.25_s{i}/run.json.gz') for i in (3, 4, 5)]
H = [load_paper(f) for f in sorted(glob.glob('/home/claude/v7work/data/F/H_p31_lw0.0150_s*/run.json.gz'))]
D = [load_paper(f) for f in sorted(glob.glob('/home/claude/v7work/data/F/D3_p61_r0.45_s*/run.json.gz'))]

fig, axs = plt.subplots(2, 3, figsize=(9.6, 5.0), layout='constrained')
for row, (runs, name, seeds) in enumerate(((R5, '$\\lambda_V$ 0.275', (1, 2, 3)), (R4, '$\\lambda_V$ 0.25', (3, 4, 5)))):
    for j, (r, sd) in enumerate(zip(runs, seeds)):
        ax = axs[row, j]; t10, t90 = first(r['s'], r['te'], 0.1), first(r['s'], r['te'], 0.9)
        a, b = max(0, t10 - 1500), min(len(r['L']) - 1, t90 + 1500); t = np.arange(a, b + 1); m = runmed(r['L'])
        ax.axvspan(t10 / 1e3, t90 / 1e3, color=K['window'], lw=0, zorder=0)
        ax.plot(t / 1e3, r['L'][a:b + 1], color=K['train'], lw=0.35, alpha=0.35, zorder=1)
        ax.plot(t / 1e3, m[a:b + 1], color=K['train'], lw=1.6, zorder=3)
        tt = np.arange(t10, t90 + 1); k, c0 = np.polyfit(tt, np.log(m[t10:t90 + 1]), 1)
        ax.plot(tt / 1e3, np.exp(c0 + k * tt), color='#222222', lw=0.9, ls='--', zorder=4)
        rate = (math.exp(1000 * k) - 1) * 100; tot = (m[t90] / m[t10] - 1) * 100
        ax.text(0.03, 0.05, f'{rate:+.1f}% per 1k updates\n{tot:+.1f}% over the rise ({(t90 - t10) / 1e3:.1f}k updates)', transform=ax.transAxes, fontsize=7.2)
        lo, hi = np.percentile(r['L'][a:b + 1], [0.5, 99.5]); ax.set_ylim(lo * 0.985, hi * 1.015)
        ax.set_title(f'R{5 if row == 0 else 4}, {name}, seed {sd}', fontsize=8); ax.grid(True, axis='y', color=K['grid'], lw=0.6)
        if j == 0: ax.set_ylabel('training CE (nats)')
        if row == 1: ax.set_xlabel('update (thousands)')
fig.suptitle('Transformer, RMS head, fraction 0.2 (campaign W): training CE while the held-out accuracy (no-mirror pairs) rises from 0.1 to 0.9 (grey);\n'
             'light: CE at every update; dark: cycle mean averaged over 201 updates; dashed: log-linear fit over the grey stretch', fontsize=8.5)
fig.savefig(os.path.join(out, 'zoom_runs.png'), dpi=200); fig.savefig(os.path.join(out, 'zoom_runs.pdf')); plt.close(fig)

cols = [(R5, 'Transformer, $\\lambda_V$ 0.275\n(campaign W, frac 0.2)', 4000), (R4, 'Transformer, $\\lambda_V$ 0.25\n(campaign W, frac 0.2)', 4000),
        (H, "Paper: one hidden layer, p = 31\n(frac 0.8, 8 runs)", 6000), (D, "Paper: depth 3, p = 61\n(frac 0.8, 8 runs)", 6000)]
fig, axs = plt.subplots(2, 4, figsize=(10.4, 4.9), layout='constrained', sharey='row')
for j, (runs, title, half) in enumerate(cols):
    lv, rates, tots, spans = [], [], [], []
    for r in runs:
        t10, t50, t90 = (first(r['s'], r['te'], q) for q in (0.1, 0.5, 0.9)); m = runmed(r['L'])
        a, b = max(0, t50 - half), min(len(m) - 1, t50 + half); x = (np.arange(a, b + 1) - t50) / 1e3
        axs[0, j].plot(x, (m[a:b + 1] / m[t50] - 1) * 100, color=K['train'], lw=1.0, alpha=0.8)
        axs[1, j].plot((r['s'] - t50) / 1e3, r['te'], color=K['test'], lw=0.9, alpha=0.7)
        if len(r['sr']): axs[1, j].plot((r['sr'] - t50) / 1e3, r['rf'], color=K['refit'], lw=0.9, alpha=0.7)
        tt = np.arange(t10, t90 + 1); k = np.polyfit(tt, np.log(m[t10:t90 + 1]), 1)[0]
        rates.append((math.exp(1000 * k) - 1) * 100); tots.append((m[t90] / m[t10] - 1) * 100); lv.append(m[t50]); spans.append((t10 - t50, t90 - t50))
    s0, s1 = np.median([s[0] for s in spans]) / 1e3, np.median([s[1] for s in spans]) / 1e3
    for ax in axs[:, j]:
        ax.axvspan(s0, s1, color=K['window'], lw=0, zorder=0); ax.set_xlim(-half / 1e3, half / 1e3); ax.grid(True, axis='y', color=K['grid'], lw=0.6)
    axs[0, j].axhline(0, color='#888888', lw=0.6)
    axs[0, j].set_title(f"{title}\nCE at held-out 0.5: {np.median(lv):.2f} nats", fontsize=7.6)
    axs[0, j].text(0.03, 0.04, f'over held-out 0.1->0.9 (grey):\n{np.median(rates):+.1f}% per 1k updates (median)\n{min(tots):+.1f} to {max(tots):+.1f}% in total',
                   transform=axs[0, j].transAxes, fontsize=6.8)
    axs[1, j].set_ylim(-0.02, 1.03)
axs[0, 0].set_ylabel('training CE relative to its value\nat held-out 0.5 (%)'); axs[0, 0].set_ylim(-15, 15)
axs[1, 0].set_ylabel('accuracy')
fig.supxlabel('updates from the held-out 0.5 crossing (thousands)', fontsize=8)
from matplotlib.lines import Line2D
fig.legend([Line2D([], [], color=K['train']), Line2D([], [], color=K['test']), Line2D([], [], color=K['refit'])],
           ['training CE (cycle mean averaged over 201 updates)', 'held-out accuracy (transformer: no-mirror pairs)', 'ridge refit'],
           loc='outside upper center', ncol=3, frameon=False, fontsize=7.5)
fig.savefig(os.path.join(out, 'zoom_compare.png'), dpi=200); fig.savefig(os.path.join(out, 'zoom_compare.pdf')); plt.close(fig)
print('ok')
