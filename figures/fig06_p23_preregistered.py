#!/usr/bin/env python3
import argparse
import glob
import json
import os

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

C = dict(train="#0072B2", train_acc="#000000", test="#D55E00", refit="#7B3294", AH="#009E73", agop="#E69F00",
         null="#888888", ink="black", grid="#E6E6E6", phase="#EBEBEB", peak="#666666")
plt.rcParams.update({
    "font.family": "DejaVu Sans", "mathtext.fontset": "dejavusans", "font.size": 8, "axes.titlesize": 8.5,
    "axes.labelsize": 8, "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8,
    "axes.titleweight": "normal", "axes.titlepad": 4, "axes.titlelocation": "left", "text.color": "black",
    "axes.labelcolor": "black", "axes.titlecolor": "black", "xtick.color": "black", "ytick.color": "black",
    "axes.edgecolor": "#888888", "axes.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False,
    "xtick.major.size": 2.5, "ytick.major.size": 2.5, "xtick.major.pad": 2, "ytick.major.pad": 2,
    "axes.grid": True, "axes.grid.axis": "y", "grid.color": C["grid"], "grid.linewidth": 0.45,
    "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.dpi": 300,
    "legend.frameon": False})
LABEL = dict(fontsize=8, color="black")
KFMT = FuncFormatter(lambda v, _: f"{v / 1000:g}")
XLABEL = "Updates (thousands)"


def acc_legend(fig, band=None):
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    h = [Line2D([], [], color=C["train_acc"], lw=1.0, ls="--"), Line2D([], [], color=C["test"], lw=1.4),
         Line2D([], [], color=C["refit"], lw=1.4)]
    lab = ["Train acc.", "Held-out", "Refit"]
    if band:
        h = [Line2D([], [], color=C["train"], lw=1.4)] + h + [Patch(facecolor=C["phase"], edgecolor="none")]
        lab = ["Train CE"] + lab + [band]
    fig.legend(h, lab, loc="outside lower center", ncol=len(h), frameon=False,
               handlelength=2.0, columnspacing=1.4 if not band else 1.2, fontsize=8)


def rng(x):
    x = np.asarray(x, dtype=float)
    return [float(x.min()), float(np.median(x)), float(x.max())]


def first_after(a, t0, th):
    i = np.flatnonzero(a[t0 + 1:] >= th)
    return t0 + 1 + int(i[0]) if len(i) else None


def load(src, t0, t1):
    runs = []
    pattern = src if "*" in src else os.path.join(src, "*.npz")
    for f in sorted(glob.glob(pattern)):
        z = np.load(f)
        ce = z["train_ce"].astype(float)
        lb = ce.copy()
        lb[:-1] = 0.5 * (ce[:-1] + ce[1:])
        os_ = z["obs_step"].astype(int)
        full = z["obs_full_observation"] > 0.5
        acc = z["test_accuracy"].astype(float)
        runs.append(dict(id=os.path.basename(f)[:-4], ce=ce, lb=lb, acc=acc, tr=z["train_accuracy"], os=os_,
                         AH=z["obs_hidden_AH_add"].astype(float), rs=os_[full],
                         ref=z["obs_refit_test_accuracy"][full].astype(float),
                         tpk=t0 + int(np.argmax(lb[t0:t1])),
                         t10=first_after(acc, t0, 0.1), t90=first_after(acc, t0, 0.9)))
    if not runs:
        raise SystemExit(f"no .npz extracts in {src}")
    return runs


def setup_axes(axs, t0, t1, ticks):
    for ax in axs:
        ax.set_xlim(t0, t1)
        ax.set_xticks(ticks)
        ax.xaxis.set_major_formatter(KFMT)
        ax.set_xlabel(XLABEL)


def heldout_panel(ax, t, acc, grid, refg, letter, tr=None):
    for y in acc:
        ax.plot(t, y, color=C["test"], lw=0.45, alpha=0.35)
    med = np.median(acc, 0)
    ax.plot(t, med, color=C["test"], lw=1.4)
    for y in refg:
        ax.plot(grid, y, color=C["refit"], lw=0.45, alpha=0.35)
    mref = np.median(refg, 0)
    ax.plot(grid, mref, color=C["refit"], lw=1.4)
    if tr is not None:
        ax.plot(t, np.median(tr, 0), color=C["train_acc"], lw=1.0, ls="--", zorder=3)
    ax.set_title(f"({letter}) Accuracy")
    ax.set_ylim(-0.03, 1.08)
    ax.set_yticks([0, 0.5, 1.0])


def ah_panel(ax, t, AHs, letter, ymax):
    for y in AHs:
        ax.plot(t, y, color=C["AH"], lw=0.45, alpha=0.35)
    ax.plot(t, np.median(AHs, 0), color=C["AH"], lw=1.4)
    ax.set_title(f"({letter}) Hidden Fourier $A_H$")
    ax.set_ylim(0, ymax)


def save(fig, out, stem):
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(out, f"{stem}.{ext}"), metadata={"Title": stem, "Author": ""} if ext == "pdf" else None)
    plt.close(fig)


def phase(src, out, t0=4500, t1=12000):
    R = load(src, t0, t1)
    assert len(R) == 20, f"expected the 20 headline runs, found {len(R)}"
    assert all(np.all(r["tr"][t0:t1 + 1] == 1.0) for r in R), "training accuracy is not 1 throughout the window"
    t = np.arange(t0, t1 + 1)
    grid = np.arange(t0, t1 + 1, 500)
    rel = np.array([100 * (r["ce"][t0:t1 + 1] / r["ce"][t0] - 1) for r in R])
    acc = np.array([r["acc"][t0:t1 + 1] for r in R])
    refg = np.array([np.interp(grid, r["rs"], r["ref"]) for r in R])
    AHs = np.array([np.interp(t, r["os"], r["AH"]) for r in R])
    pk = np.array([r["tpk"] for r in R])

    fig, axs = plt.subplots(1, 3, figsize=(6.75, 1.95), constrained_layout=True)
    setup_axes(axs, t0, t1, [4500, 6000, 8000, 10000, 12000])
    for ax in axs:
        ax.axvspan(pk.min(), pk.max(), color=C["phase"], lw=0, zorder=0)
        ax.axvline(np.median(pk), color=C["peak"], lw=0.7, ls="--", zorder=1)
    ax = axs[0]
    for y in rel:
        ax.plot(t, y, color=C["train"], lw=0.45, alpha=0.35)
    ax.plot(t, np.median(rel, 0), color=C["train"], lw=1.4)
    ax.axhline(0, color=C["ink"], lw=0.6)
    ax.set_title("(a) Training CE vs. start (%)")
    ax.set_ylabel("change from $t_0$ (%)")
    top = float(rel.max())
    ax.set_ylim(min(-0.8, float(rel.min()) - 0.5), top + 1.4)
    ax.text(pk.max() + 150, top + 0.2, "loss peak", ha="left", va="bottom", **LABEL)
    heldout_panel(axs[1], t, acc, grid, refg, "b")
    ah_panel(axs[2], t, AHs, "c", 0.125)
    save(fig, out, "fig2_H_phase")

    up = [100 * (r["lb"][r["tpk"]] / r["lb"][t0] - 1) for r in R]
    down = [100 * (r["lb"][t1 - 1] / r["lb"][r["tpk"]] - 1) for r in R]
    end = [100 * (r["ce"][t1] / r["ce"][t0] - 1) for r in R]
    gain = [(np.interp(r["tpk"], r["os"], r["AH"]) - np.interp(t0, r["os"], r["AH"])) /
            (np.interp(t1, r["os"], r["AH"]) - np.interp(t0, r["os"], r["AH"])) for r in R]
    assert all(r["t10"] is not None and r["t90"] is not None for r in R)
    d1090 = [100 * (r["lb"][r["t90"]] / r["lb"][r["t10"]] - 1) for r in R]
    stats = dict(
        window=[t0, t1], n_runs=len(R),
        peak=rng(pk), rise_to_peak_pct=rng(up), fall_after_peak_pct=rng(down), end_vs_start_pct=rng(end),
        refit_at_4500=rng(refg[:, 0]), refit_at_6000=rng([np.interp(6000, r["rs"], r["ref"]) for r in R]),
        heldout_at_peak=rng([r["acc"][r["tpk"]] for r in R]), AH_gain_fraction_before_peak=rng(gain),
        t10=rng([r["t10"] for r in R]), t90=rng([r["t90"] for r in R]),
        cycle_mean_change_t10_to_t90_pct=rng(d1090),
        peak_before_t10_all=bool(all(r["tpk"] < r["t10"] for r in R)),
        definitions=dict(
            peak="argmax of the cycle mean Lbar on [t0, t1)",
            rise_to_peak_pct="100 (Lbar(peak) / Lbar(t0) - 1)",
            fall_after_peak_pct="100 (Lbar(t1 - 1) / Lbar(peak) - 1)",
            end_vs_start_pct="100 (L(t1) / L(t0) - 1), raw CE",
            AH_gain_fraction_before_peak="(A_H(peak) - A_H(t0)) / (A_H(t1) - A_H(t0)), linear interpolation",
            t10_t90="first update > t0 with held-out accuracy >= 0.1 / 0.9",
            cycle_mean_change_t10_to_t90_pct="100 (Lbar(t90) / Lbar(t10) - 1)"))
    json.dump(stats, open(os.path.join(out, "fig2_phase_stats.json"), "w"), indent=1)
    print(json.dumps({k: v for k, v in stats.items() if k != "definitions"}))


def p23(src, out, t0=6500, t1=16500, dot=0.9, stem="p23_confirm_cohort", phase_band=False, letters="abc"):
    R = load(src, t0, t1)
    for r in R:
        r["tdot"] = first_after(r["acc"], t0, dot)
    t = np.arange(t0, t1 + 1)
    grid = np.arange(t0, t1 + 1, 500)
    rel = np.array([100 * (r["lb"][t0:t1 + 1] / r["lb"][t0] - 1) for r in R])
    acc = np.array([r["acc"][t0:t1 + 1] for r in R])
    tra = np.array([r["tr"][t0:t1 + 1] for r in R], dtype=float)
    refg = np.array([np.interp(grid, r["rs"], r["ref"]) for r in R])
    AHs = np.array([np.interp(t, r["os"], r["AH"]) for r in R])

    fig, axs = plt.subplots(1, 3, figsize=(6.75, 2.2), constrained_layout=True)
    setup_axes(axs, t0, t1, [6500, 9000, 11500, 14000, 16500])
    if phase_band:
        pk = np.array([r["tpk"] for r in R])
        for ax in axs:
            ax.axvspan(pk.min(), pk.max(), color=C["phase"], lw=0, zorder=0)
            ax.axvline(np.median(pk), color=C["peak"], lw=0.7, ls="--", zorder=1)
    ax = axs[0]
    for y in rel:
        ax.plot(t, y, color=C["train"], lw=0.45, alpha=0.35)
    ax.plot(t, np.median(rel, 0), color=C["train"], lw=1.4)
    ax.axhline(0, color=C["ink"], lw=0.6)
    for r, y in zip(R, rel):
        if r["tdot"] is not None and r["tdot"] <= t1:
            ax.plot(r["tdot"], y[r["tdot"] - t0], "o", ms=3.0, color=C["test"], mec="white", mew=0.5, zorder=3)
    ax.set_title(f"({letters[0]}) Training CE vs. start (%)")
    ax.set_ylabel("cycle mean vs. $t_0$ (%)")
    top = float(rel.max())
    ax.set_ylim(min(-0.8, float(rel.min()) - 0.5), top + 1.0)
    ax.text(t1 - 150, 0.45, f"dots: held-out {round(100 * dot)}%", ha="right", va="bottom", **LABEL)
    heldout_panel(axs[1], t, acc, grid, refg, letters[1], tr=tra)
    ah_panel(axs[2], t, AHs, letters[2], max(0.1, float(AHs.max()) * 1.1))
    acc_legend(fig)
    save(fig, out, stem)

    rows = []
    for r in R:
        row = dict(run=r["id"], tdot=r["tdot"], t90=r["t90"], peak=r["tpk"], heldout_at_peak=float(r["acc"][r["tpk"]]),
                   rise_to_peak_pct=float(100 * (r["lb"][r["tpk"]] / r["lb"][t0] - 1)))
        if r["tdot"] is not None:
            seg = r["lb"][t0:r["tdot"] + 1]
            rm = np.maximum.accumulate(seg)
            row.update(drawdown_to_dot_pct=float(100 * np.max((rm - seg) / rm)),
                       net_to_dot_pct=float(100 * (seg[-1] / seg[0] - 1)))
        rows.append(row)
    stats = dict(window=[t0, t1], dot_threshold=dot, n_runs=len(R), runs=rows,
                 dot_reached=sum(r["tdot"] is not None for r in R),
                 dot_by_t1_minus_500=sum(r["tdot"] is not None and r["tdot"] <= t1 - 500 for r in R),
                 post_hoc=dict(peak=rng([r["tpk"] for r in R]),
                               heldout_at_peak=rng([r["acc"][r["tpk"]] for r in R]),
                               n_heldout_at_peak_ge_50=int(sum(r["acc"][r["tpk"]] >= 0.5 for r in R)),
                               rise_to_peak_pct=rng([x["rise_to_peak_pct"] for x in rows])))
    json.dump(stats, open(os.path.join(out, stem + "_stats.json"), "w"), indent=1)
    print(json.dumps({k: v for k, v in stats.items() if k != "runs"}))


def grid(cohorts, out, stem="fig2_grid", height=3.35):
    n = len(cohorts)
    fig, axs = plt.subplots(3, n, figsize=(6.75, height), sharey="row", constrained_layout=True,
                            gridspec_kw=dict(height_ratios=[1.05, 1, 1]))
    axs = np.atleast_2d(axs).reshape(3, n)
    summary = []
    for j, (title, src, t0, t1) in enumerate(cohorts):
        R = load(src, t0, t1)
        t = np.arange(t0, t1 + 1)
        grid_t = np.arange(t0, t1 + 1, 500)
        rel = np.array([100 * (r["lb"][t0:t1 + 1] / r["lb"][t0] - 1) for r in R])
        acc = np.array([r["acc"][t0:t1 + 1] for r in R])
        refg = np.array([np.interp(grid_t, r["rs"], r["ref"]) for r in R])
        AHs = np.array([np.interp(t, r["os"], r["AH"]) for r in R])
        pk = np.array([r["tpk"] for r in R])
        ticks = [t0, int(round((t0 + t1) / 2000.0)) * 1000, t1]
        for i in range(3):
            ax = axs[i, j]
            ax.set_xlim(t0, t1)
            ax.set_xticks(ticks)
            ax.xaxis.set_major_formatter(KFMT)
            if i < 2:
                ax.tick_params(labelbottom=False)
        dot = dict(marker="o", ms=2.8, color="#333333", mec="white", mew=0.45, ls="none", zorder=4)
        ax = axs[0, j]
        for y in rel:
            ax.plot(t, y, color=C["train"], lw=0.45, alpha=0.35)
        ax.plot(t, np.median(rel, 0), color=C["train"], lw=1.4)
        ax.axhline(0, color=C["ink"], lw=0.6)
        ax.plot(pk, [y[k - t0] for y, k in zip(rel, pk)], **dot)
        ax.set_title(f"({'abcdefgh'[j]}) {title}")
        ax = axs[1, j]
        for y in acc:
            ax.plot(t, y, color=C["test"], lw=0.45, alpha=0.35)
        ax.plot(t, np.median(acc, 0), color=C["test"], lw=1.4)
        ax.plot(grid_t, np.median(refg, 0), color=C["refit"], lw=1.4, ls=":")
        ax.plot(pk, [a[k - t0] for a, k in zip(acc, pk)], **dot)
        ax = axs[2, j]
        for y in AHs:
            ax.plot(t, y, color=C["AH"], lw=0.45, alpha=0.35)
        ax.plot(t, np.median(AHs, 0), color=C["AH"], lw=1.4)
        ax.plot(pk, [a[k - t0] for a, k in zip(AHs, pk)], **dot)
        ax.set_xlabel("update")
        gain = [(np.interp(r["tpk"], r["os"], r["AH"]) - np.interp(t0, r["os"], r["AH"])) /
                (np.interp(t1, r["os"], r["AH"]) - np.interp(t0, r["os"], r["AH"])) for r in R]
        summary.append(dict(title=title, window=[t0, t1], n_runs=len(R),
                            window_min_ce=rng([r["ce"][t0:t1 + 1].min() for r in R]),
                            rise_to_peak_pct=rng([100 * (r["lb"][r["tpk"]] / r["lb"][t0] - 1) for r in R]),
                            peak=rng(pk), end_vs_start_pct=rng([100 * (r["lb"][t1 - 1] / r["lb"][t0] - 1) for r in R]),
                            heldout_t0=rng([r["acc"][t0] for r in R]),
                            heldout_at_peak=rng([r["acc"][r["tpk"]] for r in R]),
                            AH_gain_fraction_before_peak=rng(gain),
                            dAH=rng([np.interp(t1, r["os"], r["AH"]) - np.interp(t0, r["os"], r["AH"]) for r in R])))
    axs[0, 0].set_ylabel("CE vs. $t_0$ (%)")
    axs[1, 0].set_ylabel("held-out acc.")
    axs[2, 0].set_ylabel("$A_H$")
    axs[1, 0].set_ylim(-0.03, 1.08)
    axs[2, 0].set_ylim(0, 0.125)
    lo = min(ax.get_ylim()[0] for ax in axs[0]); hi = max(ax.get_ylim()[1] for ax in axs[0])
    axs[0, 0].set_ylim(lo, hi + 0.8)
    axs[0, 0].text(0.03, 0.05, "\u25cf  CE peak of each run", transform=axs[0, 0].transAxes, ha="left", va="bottom", **LABEL)
    j = 1 if n > 1 else 0
    a = axs[1, j]
    a.text(0.04, 0.88, "refit", transform=a.transAxes, ha="left", va="center", **LABEL)
    a.text(0.97, 0.33, "network", transform=a.transAxes, ha="right", va="center", **LABEL)
    save(fig, out, stem)
    json.dump(summary, open(os.path.join(out, stem + "_stats.json"), "w"), indent=1)
    for srow in summary:
        print(json.dumps({k: (np.round(v, 3).tolist() if isinstance(v, list) else v) for k, v in srow.items()}))


def refit_start(r, th=0.10):
    i = np.flatnonzero(r["ref"] >= th)
    if not len(i):
        return None
    return int(r["rs"][max(int(i[0]) - 1, 0)])


def sci_small(v, _=None):
    if v >= 1e-3:
        return f"{v:g}"
    m, e = f"{v:.1e}".split("e")
    m = m.rstrip("0").rstrip(".")
    return rf"${m}\times10^{{{int(e)}}}$" if m != "1" else rf"$10^{{{int(e)}}}$"


def log_ticks(lo, hi, k=3):
    cand = sorted({m * 10.0 ** e for e in range(-4, 3) for m in (1, 1.2, 1.5, 2, 3, 4, 5, 6, 8)})
    cand = [c for c in cand if lo <= c <= hi]
    if len(cand) <= k:
        return cand
    targets = np.exp(np.linspace(np.log(cand[0]), np.log(cand[-1]), k))
    return sorted({min(cand, key=lambda c: abs(np.log(c) - np.log(tg))) for tg in targets})


def grid_ext(cohorts, out, stem="fig2_grid", height=3.35, band_label=None, compact=False):
    n = len(cohorts)
    fig, axs = plt.subplots(3, n, figsize=(6.75, height), constrained_layout=True,
                            gridspec_kw=dict(height_ratios=[1.05, 1, 1]))
    axs = np.atleast_2d(axs).reshape(3, n)
    summary = []
    for j, (title, src, t0, t1) in enumerate(cohorts):
        R = load(src, t0, t1)
        rst = [refit_start(r) for r in R]
        x0 = int(np.floor((min(x for x in rst if x is not None) - 1000) / 500.0) * 500)
        x0 = max(x0, 0)
        t = np.arange(x0, t1 + 1)
        gt = np.arange(x0, t1 + 1, 500)
        lbs = np.array([r["lb"][x0:t1 + 1] for r in R])
        acc = np.array([r["acc"][x0:t1 + 1] for r in R])
        tra = np.array([r["tr"][x0:t1 + 1] for r in R], dtype=float)
        refg = np.array([np.interp(gt, r["rs"], r["ref"]) for r in R])
        AHs = np.array([np.interp(t, r["os"], r["AH"]) for r in R])
        t90 = np.array([r["t90"] if r["t90"] is not None else np.inf for r in R], dtype=float)
        b0, b1 = float(np.median(rst)), float(min(np.median(t90), t1))
        mid = int(round((x0 + t1) / 2000.0)) * 1000
        for i in range(3):
            ax = axs[i, j]
            ax.axvspan(b0, b1, color=C["phase"], lw=0, zorder=0)
            ax.set_xlim(x0, t1)
            ax.set_xticks([x0, mid, t1])
            ax.xaxis.set_major_formatter(KFMT)
            if i < 2:
                ax.tick_params(labelbottom=False)
            if j > 0 and i > 0:
                ax.tick_params(labelleft=False)
        ax = axs[0, j]
        for y in lbs:
            ax.plot(t, y, color=C["train"], lw=0.45, alpha=0.35)
        ax.plot(t, np.median(lbs, 0), color=C["train"], lw=1.4)
        ax.set_title(f"({'abcdefgh'[j]}) {title}", loc="center")
        ax.set_yscale("log")
        lo, hi = float(np.min(lbs)), float(np.max(lbs))
        ax.set_ylim(lo / 1.12, hi * 1.12)
        ax.set_yticks(log_ticks(lo, hi))
        ax.yaxis.set_minor_locator(matplotlib.ticker.NullLocator())
        ax.yaxis.set_major_formatter(FuncFormatter(sci_small))
        ax = axs[1, j]
        for y in acc:
            ax.plot(t, y, color=C["test"], lw=0.45, alpha=0.35)
        ax.plot(t, np.median(acc, 0), color=C["test"], lw=1.4)
        ax.plot(gt, np.median(refg, 0), color=C["refit"], lw=1.4, ls=":")
        ax.plot(t, np.median(tra, 0), color=C["train_acc"], lw=1.0, ls="--", zorder=3)
        ax.set_ylim(-0.03, 1.08)
        if compact:
            ax.set_yticks([0, 0.5, 1])
        ax = axs[2, j]
        for y in AHs:
            ax.plot(t, y, color=C["AH"], lw=0.45, alpha=0.35)
        ax.plot(t, np.median(AHs, 0), color=C["AH"], lw=1.4)
        ax.set_ylim(0, 0.125)
        if compact:
            ax.set_yticks([0, 0.05, 0.1])
        ax.set_xlabel("")
        ends = [r["t90"] if r["t90"] is not None else t1 for r in R]
        net = [100 * (r["lb"][e] / r["lb"][s0] - 1) for r, s0, e in zip(R, rst, ends)]
        dd = []
        for r, s0, e in zip(R, rst, ends):
            seg = r["lb"][s0:e + 1]; rm = np.maximum.accumulate(seg); dd.append(100 * float(np.max((rm - seg) / rm)))
        summary.append(dict(title=title, locked_window=[t0, t1], shown=[x0, t1], n_runs=len(R),
                            refit_start=rng(rst), band=[b0, b1], network_t90=rng([x for x in t90 if np.isfinite(x)]),
                            t90_missing=int(np.sum(~np.isfinite(t90))),
                            ce_at_refit_start=rng([r["lb"][s0] for r, s0 in zip(R, rst)]),
                            ce_at_end=rng([r["lb"][e] for r, e in zip(R, ends)]),
                            ce_change_refit_start_to_t90_pct=rng(net), n_ce_higher_at_end=int(sum(x > 0 for x in net)),
                            max_drawdown_in_band_pct=rng(dd),
                            heldout_at_refit_start=rng([r["acc"][s0] for r, s0 in zip(R, rst)])))
    labels = (["Train CE\n(log scale)", "Accuracy", "Hidden\nFourier $A_H$"] if compact else
              ["Training CE (log)", "Accuracy", "Hidden Fourier $A_H$"])
    for i, lab in enumerate(labels):
        axs[i, 0].set_ylabel(lab)
    axs[2, n // 2].set_xlabel(XLABEL)
    acc_legend(fig, band_label)
    save(fig, out, stem)
    json.dump(summary, open(os.path.join(out, stem + "_stats.json"), "w"), indent=1)
    for srow in summary:
        print(json.dumps({k: (np.round(v, 3).tolist() if isinstance(v, list) else v) for k, v in srow.items()}))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("what", choices=["phase", "p23", "grid", "grid_ext"])
    ap.add_argument("src", help="phase/p23: extracts folder; grid: ignored (use --cohort)")
    ap.add_argument("out")
    ap.add_argument("--cohort", action="append", default=[], metavar="TITLE:DIR:T0:T1",
                    help="grid: one column per cohort, in order")
    ap.add_argument("--height", type=float, default=3.5,
                    help="grid, grid_ext: figure height in inches (Figure 6: 3.5; Figure 2 since 2026-10-02: 2.8)")
    ap.add_argument("--band-label", default=None,
                    help="grid_ext: legend name of the gray band; also adds training CE to the legend (Figure 2)")
    ap.add_argument("--compact", action="store_true",
                    help="grid_ext: two-line row labels and fixed accuracy/A_H ticks for a shorter figure (Figure 2)")
    ap.add_argument("--t0", type=int)
    ap.add_argument("--t1", type=int)
    ap.add_argument("--dot", type=float, default=0.9, help="p23: held-out level marked by dots (default 0.9)")
    ap.add_argument("--stem", default="p23_confirm_cohort", help="p23: output file stem")
    ap.add_argument("--phase-band", action="store_true", help="p23: mark the range and median of the CE peaks")
    ap.add_argument("--letters", default="abc", help="p23: panel letters (default abc)")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    if a.what in ("grid", "grid_ext"):
        cs = []
        for spec in a.cohort:
            title, src, t0, t1 = spec.rsplit(":", 3)
            cs.append((title, src, int(t0), int(t1)))
        stem = a.stem if a.stem != "p23_confirm_cohort" else "fig2_grid"
        if a.what == "grid_ext":
            grid_ext(cs, a.out, stem, a.height, a.band_label, a.compact)
        else:
            grid(cs, a.out, stem, a.height)
    elif a.what == "phase":
        phase(a.src, a.out, a.t0 or 4500, a.t1 or 12000)
    else:
        p23(a.src, a.out, a.t0 or 6500, a.t1 or 16500, a.dot, a.stem, a.phase_band, a.letters)


if __name__ == "__main__":
    main()
