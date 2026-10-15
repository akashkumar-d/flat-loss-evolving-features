#!/usr/bin/env python3
import glob, json, math, os, sys
from collections import deque
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter, MaxNLocator

K = dict(train="#0072B2", train_acc="#000000", test="#D55E00", refit="#7B3294", AH="#009E73", window="#EBEBEB", grid="#E6E6E6",
         null="#888888")
KFMT = FuncFormatter(lambda v, _: f"{v / 1000:g}")
plt.rcParams.update(matplotlib.rcParamsDefault)
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 8, 'axes.titlesize': 8, 'axes.labelsize': 8, 'xtick.labelsize': 8,
                     'ytick.labelsize': 8, 'legend.fontsize': 8, 'axes.spines.top': False, 'axes.spines.right': False,
                     'axes.edgecolor': '#555555', 'axes.linewidth': 0.7})
GAIN = 0.2


def stretches(c, start, tol=1.03):
    n = len(c); amin = np.full(n, -1, int); dq = deque(); a = start
    for t in range(start, n):
        while dq and c[dq[-1]] <= c[t]: dq.pop()
        dq.append(t)
        while c[dq[0]] > tol * c[t]:
            a += 1
            while dq[0] < a: dq.popleft()
        amin[t] = a
    return amin


def new_window(c, start):
    start = int(min(max(start, 0), len(c) - 1)); amin = stretches(c, start); n = len(c)
    S = [(int(amin[t]), t) for t in range(start, n) if t == n - 1 or amin[t + 1] != amin[t]]
    return max(S, key=lambda s: (float(np.max(c[s[0]:s[1] + 1]) - c[s[0]]), s[1] - s[0]))


def dip_start(c, tol=1.03):
    amin = stretches(c, 0, tol); n = len(c)
    S = [(int(amin[t]), t) for t in range(n) if t == n - 1 or amin[t + 1] != amin[t]]
    best, start = 1.0, n - 1
    for a, b in S:
        seg = c[a:b + 1]; low = np.minimum.accumulate(seg); k = int(np.argmax(seg / low)); r = float(seg[k] / low[k])
        if r > best: best, start = r, a + int(np.argmin(seg[:k + 1]))
    return start if best >= tol else n - 1


def gain(series, a, b):
    x, y = (np.asarray(v, float) for v in series); ok = np.isfinite(y) & (x >= a) & (x <= b)
    return 0.0 if ok.sum() < 2 else float(y[ok][-1] - y[ok][0])


def score(r, t_c, c, start):
    a, b = new_window(c, start)
    W = [int(t_c[a]), int(t_c[b])]; g_te, g_rf = gain(r["test"], *W), gain(r["refit"], *W)
    pk = a + int(np.argmax(c[a:b + 1]))
    r.update(W=W, primary=bool(g_te >= GAIN or g_rf >= GAIN), peak=int(t_c[pk]), start=int(t_c[min(start, len(c) - 1)]))
    return r


def onehl_run(npz):
    z = np.load(npz); j = json.loads(Path(npz[:-4] + ".json").read_text()); m = j["meta"]; cfg = m["config"]; p = int(cfg["p"])
    s = z["step"].astype(float); L = z["train_ce"].astype(float)
    ta = z["acc_step"].astype(float) if "acc_step" in z.files else s
    te, tr = z["test_accuracy"].astype(float), z["train_accuracy"].astype(float)
    to = z["obs_step"].astype(float); rf = z["obs_refit_test_accuracy"].astype(float); ah = z["obs_hidden_AH_add"].astype(float)
    assert len(s) == int(s[-1]) + 1, 'the headline runs log the CE at every update'
    c = (L[:-1] + L[1:]) / 2 / math.log(p); t_c = s[:-1]
    ok_r, ok_a = np.isfinite(rf), np.isfinite(ah)
    t90 = ta[np.argmax(te >= 0.9)] if np.any(te >= 0.9) else None
    r = dict(id=m["run_id"], p=p, ce=(t_c, c), test=(ta, te), train=(ta, tr), refit=(to[ok_r], rf[ok_r]), ahL=(to[ok_a], ah[ok_a]),
             t90=float(t90) if t90 is not None else None)
    lv = cfg.get("decay_V")
    if "V_norm" in z.files and lv:
        k = np.flatnonzero(z["V_norm"] >= 0.99 / lv)
        start = int(np.searchsorted(t_c, s[k[0]])) if len(k) else 0
    else:
        start = dip_start(c)
    r = score(r, t_c, c, start)
    tc = z["test_ce"].astype(float); r["test_ce"] = (tc[:-1] + tc[1:]) / 2 / math.log(p)
    return r


def on_grid(x, y, g):
    x = np.asarray(x, float); y = np.asarray(y, float); ok = np.isfinite(y); x, y = x[ok], y[ok]
    return np.interp(g, x, y, left=y[0], right=y[-1])


def hero(runs, out, layout, x0=1000, x1=16000):
    n = len(runs); p = runs[0]["p"]; lp = math.log(p)
    if layout == '2x2':
        W_, H_ = 3.30, 2.30; bottom, gap, top = 0.49, 0.405, 0.205; slots = [(.12, .335), (.645, .315)]
        fig = plt.figure(figsize=(W_, H_)); h = (H_ - bottom - gap - top) / 2; rows = [bottom + h + gap, bottom]
        axs = [fig.add_axes([slots[cc][0], rows[rr] / H_, slots[cc][1], h / H_]) for rr, cc in ((0, 0), (0, 1), (1, 0), (1, 1))]
        xlab, leg = (.53, .078), (.53, .010)
    else:
        W_, H_ = 6.75, 1.58; left, right, gap = 0.36, 0.06, 0.42; bottom, top = 0.57, 0.22
        w = (W_ - left - right - 3 * gap) / 4
        fig = plt.figure(figsize=(W_, H_))
        axs = [fig.add_axes([(left + i * (w + gap)) / W_, bottom / H_, w / W_, (H_ - bottom - top) / H_]) for i in range(4)]
        xlab, leg = (0.5, 0.15), (0.5, 0.005)
    g = np.arange(x0, x1 + 1, 10, dtype=float)
    band = (float(np.median([r["W"][0] for r in runs])), float(np.median([r["W"][1] for r in runs])))
    locked_window = (4500, 12000)
    thin = dict(lw=0.45, alpha=0.28, zorder=1)
    C = np.array([on_grid(*r["ce"], g) for r in runs]) * lp
    HE = np.array([on_grid(r["ce"][0], r["test_ce"], g) for r in runs]) * lp
    peak = float(np.median([r["peak"] for r in runs]))
    inb = (g >= band[0]) & (g <= band[1]); curve_peak = float(g[inb][np.argmax(np.median(C, 0)[inb])])
    for ax in axs:
        ax.axvspan(*locked_window, color=K["window"], lw=0, zorder=0); ax.grid(True, axis="y", color=K["grid"], lw=0.6)
        ax.set_xlim(x0, x1); ax.xaxis.set_major_formatter(KFMT); ax.xaxis.set_major_locator(MaxNLocator(3, integer=True))
        ax.axvline(peak, color="#444444", lw=0.8, ls=(0, (3, 2)), zorder=6)
    ax = axs[0]
    for y in HE: ax.plot(g, y, color=K["test"], **thin)
    for y in C: ax.plot(g, y, color=K["train"], **thin)
    ax.plot(g, np.median(HE, 0), color=K["test"], lw=1.4, zorder=3); ax.plot(g, np.median(C, 0), color=K["train"], lw=1.4, zorder=3)
    ax.axhline(lp, color=K["null"], lw=0.8, ls="--", zorder=1)
    ax.text(0.97, lp / 7.5 + 0.02, r"$\log p$", transform=ax.transAxes, ha="right", va="bottom", fontsize=8)
    ax.set_ylim(0, 7.5); ax.set_yticks([0, 3, 6]); ax.set_title("Cross-entropy", loc="center", fontsize=8.5)
    ax = axs[1]
    for y in C: ax.plot(g, y, color=K["train"], **thin)
    med = np.median(C, 0); ax.plot(g, med, color=K["train"], lw=1.4, zorder=3)
    lo, hi = float(np.min(med)), float(np.max(med)); pad = 0.08 * (hi - lo)
    ax.set_ylim(lo - pad, hi + 2.2 * pad)
    ax.text(peak, hi + 0.9 * pad, " peak", ha="left", va="bottom", fontsize=8)
    ax.set_title("Training CE", loc="center", fontsize=8.5)
    ax = axs[2]
    T = np.array([on_grid(*r["test"], g) for r in runs])
    for y in T: ax.plot(g, y, color=K["test"], **thin)
    ax.plot(g, np.median(np.array([on_grid(*r["train"], g) for r in runs]), 0), color=K["train_acc"], lw=1.0, ls="--", zorder=3)
    ax.plot(g, np.median(np.array([on_grid(*r["refit"], g) for r in runs]), 0), color=K["refit"], lw=1.4, zorder=4)
    ax.plot(g, np.median(T, 0), color=K["test"], lw=1.4, zorder=5)
    ax.set_ylim(-0.04, 1.06); ax.set_yticks([0, 1]); ax.set_title("Accuracy", loc="center", fontsize=8.5)
    ax = axs[3]
    A = np.array([on_grid(*r["ahL"], g) for r in runs])
    for y in A: ax.plot(g, y, color=K["AH"], **thin)
    ax.plot(g, np.median(A, 0), color=K["AH"], lw=1.4, zorder=3)
    ax.set_ylim(0, 0.13); ax.set_yticks([0, 0.06, 0.12]); ax.set_title(r"Hidden Fourier $A_H$", loc="center", fontsize=8.5)
    fig.text(*xlab, "Updates (thousands)", ha="center", va="bottom", fontsize=8)
    hh = [Line2D([], [], color=K["train"], lw=1.1), Line2D([], [], color=K["train_acc"], lw=1.0, ls="--"),
          Line2D([], [], color=K["test"], lw=1.1), Line2D([], [], color=K["refit"], lw=1.1)]
    fig.legend(hh, ["Train CE", "Train acc.", "Held-out", "Refit"], loc="lower center", bbox_to_anchor=leg, ncol=4,
               frameon=False, handlelength=1.3, handletextpad=.35, columnspacing=.7 if layout == '2x2' else 1.4, borderaxespad=0,
               borderpad=0, fontsize=8)
    stem = 'H_hero_2x2' if layout == '2x2' else 'H_hero_1x4'
    os.makedirs(out, exist_ok=True); fig.canvas.draw()
    fig.savefig(os.path.join(out, f"{stem}.pdf"), metadata={"Title": stem, "Author": "", "Creator": "hero_row.py", "CreationDate": None})
    fig.savefig(os.path.join(out, f"{stem}.png"), dpi=200); plt.close(fig)
    T90 = [r["t90"] for r in runs]
    return dict(n=n, band_median=list(band), peak_median=peak, peak_of_median_curve=curve_peak, shaded_window=list(locked_window),
                passes=int(sum(r["primary"] for r in runs)),
                peak_minus_t90_median=float(np.median([r["peak"] - t for r, t in zip(runs, T90) if t is not None])))


if __name__ == '__main__':
    src, out = sys.argv[1], sys.argv[2]
    runs = [onehl_run(f) for f in sorted(glob.glob(os.path.join(src, '*.npz')))]
    assert len(runs) == 20
    rep = {lay: hero(runs, out, lay) for lay in ('2x2', '1x4')}
    json.dump(rep, open(os.path.join(out, 'hero_row_stats.json'), 'w'), indent=1)
    print(json.dumps(rep['1x4']))
