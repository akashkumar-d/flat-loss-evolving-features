#!/usr/bin/env python3
import math, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import ovw, win
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter, MaxNLocator

K = ovw.K
PAPER = {'font.family': 'DejaVu Sans', 'font.size': 8, 'axes.titlesize': 8, 'axes.labelsize': 8, 'xtick.labelsize': 8,
         'ytick.labelsize': 8, 'legend.fontsize': 8, 'axes.spines.top': False, 'axes.spines.right': False,
         'axes.edgecolor': '#555555', 'axes.linewidth': 0.7}
KFMT = FuncFormatter(lambda v, _: f"{v / 1000:g}")
GAIN = 0.2


def maximal_stretches(c, start, tol=1.03):
    amin = win.stretches(c, start, tol); n = len(c)
    return [(int(amin[t]), t) for t in range(start, n) if t == n - 1 or amin[t + 1] != amin[t]]


def new_window(c, knee):
    start = int(min(max(knee or 0, 0), len(c) - 1))
    S = maximal_stretches(c, start)
    return max(S, key=lambda s: (float(np.max(c[s[0]:s[1] + 1]) - c[s[0]]), s[1] - s[0]))


def inside_gain(x, y, a, b):
    m = (np.asarray(x) >= a) & (np.asarray(x) <= b)
    return 0.0 if m.sum() < 2 else float(np.asarray(y)[m][-1] - np.asarray(y)[m][0])


def record2(rid):
    d, f = ovw.find(rid); r = ovw.record(d, rid, f)
    r['tf'], r['rf'] = ovw._series(d['obs'], 'v1_ridge_acc_te'); r['refit_kind'] = 'canonical'
    if not len(r['tf']): (r['tf'], r['rf']), r['refit_kind'] = ovw._series(d['obs'], 'refit_acc'), 'legacy'
    a, b = new_window(r['cm'], r['knee'])
    r['W_old'], r['ok_old'] = r['W'], r['primary']
    r['W2'] = [a, b]
    r['g_te'] = inside_gain(r['t'], r['te'], a, b); r['g_rf'] = inside_gain(r['tf'], r['rf'], a, b)
    r['ok2'] = (r['g_te'] >= GAIN) or (r['g_rf'] >= GAIN)
    m = (r['t'] >= a) & (r['t'] <= b)
    r['fit2'] = bool(len(r['mm']) and np.all(r['mm'][(r['tm'] >= a) & (r['tm'] <= b)] > 0))
    te = lambda s: float(np.clip(r['te'][np.argmin(np.abs(r['t'] - s))], 0.1, 0.9))
    r['share2'] = (te(b) - te(a)) / 0.8
    r['rise2'] = float(np.max(r['cm'][a:b + 1]) - r['cm'][a])
    return r


def load2(ids): return [record2(i) for i in ids]


def summary(R):
    return dict(n=len(R), k=sum(r['ok2'] for r in R), k_old=sum(r['ok_old'] for r in R),
                W=[float(np.median([r['W2'][0] for r in R])), float(np.median([r['W2'][1] for r in R]))],
                g_te=float(np.median([r['g_te'] for r in R])), g_rf=float(np.median([r['g_rf'] for r in R])),
                g_te_rng=[min(r['g_te'] for r in R), max(r['g_te'] for r in R)], g_rf_rng=[min(r['g_rf'] for r in R), max(r['g_rf'] for r in R)],
                fit=sum(r['fit2'] for r in R), final_ok=sum((r['final'] or 0) >= 0.99 for r in R),
                refit=sorted({r['refit_kind'] for r in R}))


def _xmax(R):
    ends = [r['W2'][1] for r in R]; t90 = [(r['t90'] or r['W2'][1]) for r in R]
    xm = int(math.ceil((1.15 * max(np.median(ends), np.percentile(t90, 90)) + 1500) / 1000) * 1000)
    return min(xm, max(r['stop'] for r in R)) if xm > max(r['stop'] for r in R) else xm


def column(R, title, full=False):
    xmax = max(r['stop'] for r in R) if full else _xmax(R)
    g = np.arange(0, xmax + 1, max(10, xmax // 2000), dtype=float)
    S = dict(ce=np.array([ovw.hold(np.arange(len(r['cm'])), r['cm'], g) for r in R]),
             test=np.array([ovw.hold(r['t'], r['te'], g) for r in R]), train=np.array([ovw.hold(r['t'], r['tr'], g) for r in R]),
             err=np.array([ovw.hold(r['t'], (1 - r['tr']) * r['ntr'], g) for r in R]),
             refit=np.array([ovw.hold(r['tf'], r['rf'], g) for r in R]),
             ahL=np.array([ovw.hold(r['ta'], r['ahL'], g) for r in R]),
             ah1=np.array([ovw.hold(*r['ah1'], g) for r in R]) if R[0]['ah1'] is not None else None)
    band = (float(np.median([r['W2'][0] for r in R])), float(np.median([r['W2'][1] for r in R])))
    return dict(g=g, S=S, band=band, xmax=xmax, title=title, n=len(R), k=sum(r['ok2'] for r in R), ntr=max(r['ntr'] for r in R),
                legacy=any(r['refit_kind'] == 'legacy' for r in R))


def pgrid(cols, out, stem, rows=('ce', 'acc', 'err', 'ah'), height=5.2, ce_log=False, note=None, full=False):
    plt.rcParams.update(matplotlib.rcParamsDefault); plt.rcParams.update(PAPER)
    cols = [column(R, t, full) for t, R in cols]
    n = len(cols); width = max(6.75, 1.12 * n + 0.6)
    fig, axs = plt.subplots(len(rows), n, figsize=(width, height), sharey='row', layout='constrained')
    axs = np.atleast_2d(axs)
    if n == 1: axs = axs.reshape(len(rows), 1)
    ah_top = max(float(np.nanmax(c['S']['ahL'])) for c in cols) * 1.08
    err_top = max(c['ntr'] for c in cols) * 1.5
    for j, c in enumerate(cols):
        g, S = c['g'], c['S']
        for i, row in enumerate(rows):
            ax = axs[i, j]
            if c['band'][1] - c['band'][0] > c['xmax'] * 0.004: ax.axvspan(*c['band'], color=K['window'], lw=0, zorder=0)
            else: ax.axvline(c['band'][0], color='#C8C8C8', lw=2.0, zorder=0)
            ax.grid(True, axis='y', color=K['grid'], lw=0.6); ax.set_xlim(0, c['xmax'])
            ax.xaxis.set_major_formatter(KFMT); ax.xaxis.set_major_locator(MaxNLocator(2 if n > 5 else 3, integer=True))
            thin = dict(lw=0.45, alpha=0.28 if c['n'] > 2 else 0.5, zorder=1)
            if row == 'ce':
                for y in S['ce']: ax.plot(g, y, color=K['train'], **thin)
                ax.plot(g, np.median(S['ce'], 0), color=K['train'], lw=1.4, zorder=3)
                if ce_log: ax.set_yscale('log'); ax.set_ylim(1e-9, 3)
                else: ax.set_ylim(0, 1.05)
            elif row == 'acc':
                for y in S['test']: ax.plot(g, y, color=K['test'], **thin)
                ax.plot(g, np.median(S['train'], 0), color=K['train_acc'], lw=1.0, ls='--', zorder=3)
                ax.plot(g, np.median(S['refit'], 0), color=K['refit'], lw=1.4, ls=':', zorder=4)
                ax.plot(g, np.median(S['test'], 0), color=K['test'], lw=1.4, zorder=5); ax.set_ylim(-0.02, 1.04)
            elif row == 'err':
                for y in S['err']: ax.plot(g, y, color=K['train_acc'], **thin)
                ax.plot(g, np.median(S['err'], 0), color=K['train_acc'], lw=1.1, zorder=3)
                ax.set_yscale('symlog', linthresh=1); ax.set_ylim(-0.2, err_top)
                ticks = [t for t in (0, 1, 10, 100, 1000, 10000) if t <= err_top]
                ax.set_yticks(ticks); ax.set_yticklabels([{1000: '1k', 10000: '10k'}.get(t, str(t)) for t in ticks])
            elif row == 'ah':
                for y in S['ahL']: ax.plot(g, y, color=K['AH'], **thin)
                ax.plot(g, np.median(S['ahL'], 0), color=K['AH'], lw=1.4, ls='-', zorder=3)
                if S['ah1'] is not None: ax.plot(g, np.median(S['ah1'], 0), color=K['AH'], lw=1.2, ls='-.', zorder=3)
                ax.set_ylim(0, ah_top)
            if i < len(rows) - 1: ax.tick_params(labelbottom=False)
        axs[0, j].set_title(f"{c['title']}\n{c['k']}/{c['n']} pass", loc='center')
    lab = dict(ce='Training CE / log p\n(cycle mean)', acc='Accuracy', err='Misclassified\ntraining pairs', ah='Hidden Fourier\n$A_H$')
    for i, row in enumerate(rows): axs[i, 0].set_ylabel(lab[row])
    axs[-1, n // 2].set_xlabel('Updates (thousands)')
    h = [Line2D([], [], color=K['train'], lw=1.4), Line2D([], [], color=K['train_acc'], lw=1.0, ls='--'),
         Line2D([], [], color=K['test'], lw=1.4), Line2D([], [], color=K['refit'], lw=1.4, ls=':')]
    l = ['Train CE', 'Train acc.', 'Held-out', 'Refit']
    if 'err' in rows: h.append(Line2D([], [], color=K['train_acc'], lw=1.1)); l.append('Misclassified')
    if 'ah' in rows and any(c['S']['ah1'] is not None for c in cols):
        h += [Line2D([], [], color=K['neutral'], lw=1.4, ls='-'), Line2D([], [], color=K['neutral'], lw=1.2, ls='-.')]
        l += ['Last hidden', 'First hidden']
    elif 'ah' in rows: h.append(Line2D([], [], color=K['AH'], lw=1.4)); l.append('$A_H$')
    fig.legend(h, l, loc='outside lower center', ncol=len(h), frameon=False, handlelength=1.8, columnspacing=0.9, fontsize=8)
    expl = ('Grey: median window, where the training CE is not falling (never 3% below its running peak).\n'
            'A seed passes if its held-out or refit accuracy rises by at least 0.2 inside the window.')
    leg = [c['title'].replace('\n', ' ') for c in cols if c['legacy']]
    if leg: expl += (' Refit: legacy (no canonical ridge logged).' if len(leg) == n else
                     f" Refit: canonical ridge; legacy in {', '.join(leg)}.")
    fig.suptitle((note + '\n' if note else '') + expl, x=0.01, ha='left', fontsize=7.5 if n < 7 else 8, color='#444444')
    fig.savefig(os.path.join(out, f'{stem}.png'), dpi=300); plt.close(fig)
    return [dict(title=c['title'].replace('\n', ' '), k=c['k'], n=c['n'], band=c['band']) for c in cols]


def sheet2(R, title, out, stem, note=''):
    plt.rcParams.update(matplotlib.rcParamsDefault); plt.rcParams.update(ovw.STYLE)
    n = len(R); XM = _xmax(R); fmt_k = ovw.fmt_k; rng = ovw.rng
    band = (float(np.median([r['W2'][0] for r in R])), float(np.median([r['W2'][1] for r in R])))
    knee = float(np.median([r['knee'] for r in R if r['knee'] is not None])) if any(r['knee'] is not None for r in R) else None
    thin_a = 0.4 if n <= 12 else 0.22
    g1 = np.arange(0, XM + 1, 1.0 if XM <= 150000 else 2.0); g20 = np.arange(0, XM + 1, 20.0); g500 = np.arange(0, XM + 1, 500.0)
    med = lambda key, g, tk='t': np.nanmedian([ovw.hold(r[tk], r[key], g) for r in R], 0)
    ce_med = np.median([ovw.hold(np.arange(len(r['L'])), r['L'], g1) for r in R], 0)
    post = np.concatenate([r['cm'][500:min(len(r['cm']), XM)] for r in R])
    lo = max(0.0, float(np.min(post)) - 0.05); hi = min(1.05, float(np.max(post)) * 1.04 + 0.01)
    if hi - lo < 0.15: mid = (hi + lo) / 2; lo, hi = max(0.0, mid - 0.08), mid + 0.08
    above = [int(np.where(r['L'][:20000] > hi)[0].max()) if np.any(r['L'][:20000] > hi) else 0 for r in R]

    fig = plt.figure(figsize=(11.0, 10.8))
    gs = fig.add_gridspec(5, 2, width_ratios=[2.0, 1], hspace=0.30, wspace=0.2, left=0.07, right=0.985, top=0.895, bottom=0.07)
    gr = gs[:, 1].subgridspec(3, 1, height_ratios=[1, 1, 2.5], hspace=0.42)
    ax = [fig.add_subplot(gs[i, 0]) for i in range(5)]
    for a in ax[1:]: a.sharex(ax[0])
    thin = dict(lw=0.5, alpha=thin_a, zorder=2)
    for a in ax:
        a.axvspan(*band, color=K['window'], lw=0, zorder=0)
        if knee is not None: a.axvline(knee, color=K['neutral'], lw=0.8, ls=':', zorder=1)
        a.grid(True, axis='y', color=K['grid'], lw=0.6); a.set_xlim(0, XM)
        a.xaxis.set_major_formatter(KFMT); a.xaxis.set_major_locator(MaxNLocator(10, integer=True))
    for a in ax[:-1]: a.tick_params(labelbottom=False)
    ax[-1].set_xlabel('Updates (thousands)')

    a = ax[0]
    for r in R: a.plot(np.arange(len(r['L'])), r['L'], color=K['train'], **thin)
    a.plot(g1, ce_med, color=K['train'], lw=1.6, zorder=4)
    a.set_ylim(lo, hi); a.set_ylabel('Training CE / log p\n(raw, every update)')
    a.text(band[0] + XM * 0.01, lo + 0.05 * (hi - lo), 'window: CE not falling (never 3% below its running peak)', fontsize=7.5,
           color='#333333', va='bottom')
    ab = f'; above the axis until {min(above)}–{max(above)} updates' if max(above) > 0 else ''
    a.set_title(f'(a) Raw training CE, every seed (thin) and median (thick){ab}', loc='left')

    a = ax[1]
    for r in R:
        a.plot(r['t'], r['tr'], color=K['train_acc'], **thin); a.plot(r['t'], r['te'], color=K['test'], **thin)
        a.plot(r['tf'], r['rf'], color=K['refit'], lw=0.5, alpha=thin_a, zorder=2)
    a.plot(g20, med('tr', g20), color=K['train_acc'], lw=1.4, ls='--', zorder=4)
    a.plot(g500, med('rf', g500, 'tf'), color=K['refit'], lw=1.6, ls=':', zorder=5)
    a.plot(g20, med('te', g20), color=K['test'], lw=1.6, zorder=6)
    a.set_ylim(-0.02, 1.04); a.set_ylabel('Accuracy')
    a.legend([Line2D([], [], color=K['train_acc'], lw=1.4, ls='--'), Line2D([], [], color=K['test'], lw=1.6),
              Line2D([], [], color=K['refit'], lw=1.6, ls=':')],
             ['training (median; thin: each seed)', 'held-out',
              'refit (train-only ridge readout' + ('; legacy)' if any(r['refit_kind'] == 'legacy' for r in R) else ')')],
             loc='lower right', frameon=False)
    a.set_title('(b) Accuracy: training, held-out and refit (pass: held-out or refit rises ≥ 0.2 inside the window)', loc='left')

    a = ax[2]
    for r in R: a.plot(r['t'], (1 - r['tr']) * r['ntr'], color=K['train_acc'], **thin)
    a.plot(g20, (1 - med('tr', g20)) * R[0]['ntr'], color=K['train_acc'], lw=1.4, zorder=4)
    top = R[0]['ntr'] * 1.5; a.set_yscale('symlog', linthresh=1); a.set_ylim(-0.2, top)
    tk = [x for x in (0, 1, 10, 100, 1000, 10000) if x <= top]
    a.set_yticks(tk); a.set_yticklabels([{1000: '1k', 10000: '10k'}.get(x, str(x)) for x in tk])
    a.set_ylabel(f"Misclassified\ntraining pairs (of {R[0]['ntr']:,})"); a.set_title('(c) Misclassified training pairs', loc='left')

    a = ax[3]
    if all(len(r['mm']) for r in R):
        for r in R: a.plot(r['tm'], r['mm'], color=K['train_acc'], **thin)
        a.plot(g20, med('mm', g20, 'tm'), color=K['train_acc'], lw=1.4, zorder=4)
        a.axhline(0, color=K['zero'], lw=0.8, zorder=3)
        mmin = min(float(np.min(r['mm'])) for r in R); mmax = max(float(np.max(r['mm'])) for r in R)
        a.set_yscale('symlog', linthresh=0.01); a.set_ylim(min(-0.02, mmin * 1.3), max(0.05, mmax * 1.5))
        a.text(0.995, 0.06, f"every training margin in the window > 0: {sum(r['fit2'] for r in R)} of {n} (reported, not required)",
               transform=a.transAxes, ha='right', fontsize=7.5, color='#333333')
    a.set_ylabel('Smallest training\nmargin (logits)')
    a.set_title('(d) Smallest training margin (correct − largest wrong logit); below 0, some pair is misclassified', loc='left')

    a = ax[4]
    if all(len(r['rms']) for r in R):
        for r in R: a.plot(r['th'], r['rms'], color=K['neutral'], **thin)
        a.plot(g500, med('rms', g500, 'th'), color=K['neutral'], lw=1.6, zorder=4)
        allr = np.concatenate([r['rms'][1:] for r in R]); allr = allr[allr > 0]
        if allr.max() / np.median(allr) > 20: a.set_yscale('log')
        else: a.set_ylim(0, None)
    a.set_ylabel('Logit RMS'); a.set_title('(e) Size of the logits (RMS over all pairs and classes; heavy observer)', loc='left')

    zz = [float(np.median(np.abs(r['L'][1:3000] - (r['L'][:2999] + r['L'][2:3001]) / 2))) for r in R]
    r = R[int(np.argmax(zz))]; m = min(3000, len(r['cm']))
    a = fig.add_subplot(gr[0])
    a.plot(np.arange(m), r['L'][:m], color=K['train'], lw=0.35, alpha=0.8, label='raw')
    a.plot(np.arange(m), r['cm'][:m], color=K['cm'], lw=1.2, label='cycle mean')
    a.set_xlim(0, m); a.set_ylim(max(0.0, float(np.min(r['L'][:m])) - 0.05), float(np.max(r['L'][:m])) * 1.05)
    a.xaxis.set_major_formatter(KFMT); a.xaxis.set_major_locator(MaxNLocator(3, integer=True))
    a.grid(True, axis='y', color=K['grid'], lw=0.6); a.legend(loc='upper right', frameon=False)
    a.set_title(f"(f) First 3k updates, {ovw.seed_label(r['id'])}", loc='left')
    a.set_ylabel('Training CE / log p'); a.set_xlabel('Updates (thousands)', labelpad=1)

    a = fig.add_subplot(gr[1])
    for r in R:
        dv = np.abs(r['L'][:-1] - r['cm']); a.plot(np.arange(1, len(dv)), np.maximum(dv[1:], 1e-9), color=K['train'], lw=0.4, alpha=0.3)
    a.set_yscale('log'); a.set_ylim(1e-7, 1); a.set_xlim(0, XM); a.axvspan(*band, color=K['window'], lw=0, zorder=0)
    if knee is not None: a.axvline(knee, color=K['neutral'], lw=0.8, ls=':')
    a.xaxis.set_major_formatter(KFMT); a.xaxis.set_major_locator(MaxNLocator(5, integer=True))
    a.set_yticks([1e-6, 1e-4, 1e-2, 1]); a.grid(True, axis='y', color=K['grid'], lw=0.6); a.set_ylabel('|raw − cycle mean| / log p')
    bm = [float(np.abs(r['L'][r['W2'][0]:r['W2'][1]] - r['cm'][r['W2'][0]:r['W2'][1]]).max()) for r in R if r['W2'][1] > r['W2'][0]]
    if bm: a.text(0.98, 0.9, f'in the window: ≤ {max(bm):.0e} in every seed', transform=a.transAxes, ha='right', fontsize=7.5)
    a.set_title('(g) What the cycle mean removes', loc='left'); a.set_xlabel('Updates (thousands)', labelpad=1)

    a = fig.add_subplot(gr[2])
    for r in R:
        tt = r['t'].astype(int); ok = tt < len(r['cm'])
        a.plot(r['cm'][tt[ok]], r['tr'][ok], color=K['train_acc'], lw=0.5, alpha=0.45 if n <= 12 else 0.25, zorder=2)
    mk = dict(ms=4.5, zorder=4, mec='white', mew=0.5)
    for r in R:
        a0, b0 = r['W2']
        j = int(np.argmin(np.abs(r['t'] - a0))); a.plot(r['cm'][min(a0, len(r['cm']) - 1)], r['tr'][j], 's', color=K['neutral'], **mk)
        j = int(np.argmin(np.abs(r['t'] - b0))); a.plot(r['cm'][min(b0, len(r['cm']) - 1)], r['tr'][j], 'D', color=K['train'], **mk)
        if r['t90'] is not None and r['t90'] < len(r['cm']):
            j = int(np.argmin(np.abs(r['t'] - r['t90']))); a.plot(r['cm'][r['t90']], r['tr'][j], '^', color=K['test'], **mk)
    a.set_xlim(lo, hi); a.set_ylim(0, 1.04); a.grid(True, color=K['grid'], lw=0.6)
    a.set_xlabel('Training CE / log p (cycle mean)'); a.set_ylabel('Training accuracy')
    a.legend([Line2D([], [], marker=m_, ls='', color=c_, ms=5) for m_, c_ in (('s', K['neutral']), ('D', K['train']), ('^', K['test']))],
             ['window start', 'window end', 'held-out reaches 90%'], loc='best', frameon=True, framealpha=0.85, edgecolor='none')
    a.set_title('(h) Training accuracy vs training CE', loc='left')

    k = sum(r['ok2'] for r in R); k_old = sum(r['ok_old'] for r in R)
    fig.suptitle(f'{title}: {k}/{n} seeds pass (old endpoint {k_old}/{n})', x=0.07, ha='left', fontsize=10.5, y=0.992)
    past = [r for r in R if r['stop'] > XM]
    tail = ((f" All {n} seeds run to {fmt_k(past[0]['stop'])}; the axis stops at {fmt_k(XM)}." if len(past) == n and len({r['stop'] for r in past}) == 1
             else f" {len(past)} seed(s) run past {fmt_k(XM)} (to {', '.join(fmt_k(r['stop']) for r in past[:6])}{'…' if len(past) > 6 else ''}).") if past else '')
    sub = (f"Grey: median window ({fmt_k(band[0])}–{fmt_k(band[1])}); gain inside it: held-out {rng([r['g_te'] for r in R])}, refit {rng([r['g_rf'] for r in R])}."
           + (f" Dotted: median knee ({fmt_k(knee)})." if knee is not None else '')
           + '\nSeeds that stop early keep their last value in the medians.' + tail + ('\n' + note if note else ''))
    fig.text(0.07, 0.951, sub, fontsize=8, color='#444444', va='center')
    fig.savefig(os.path.join(out, f'{stem}.png'), dpi=150); plt.close(fig)
