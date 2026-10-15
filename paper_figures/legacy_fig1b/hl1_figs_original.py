from __future__ import annotations

import json
import math
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter, MaxNLocator

C = dict(train="#0072B2", test="#009E73", refit="#E69F00", AH="#7B4AB5", agop="#CC79A7", null="#8C8C8C",
         ink="#222222", muted="#666666", grid="#E6E6E6", window="#E3E8EE")
TEXTW = 5.5

plt.rcParams.update({
    "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7,
    "legend.fontsize": 7, "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "xtick.major.size": 2.5, "ytick.major.size": 2.5, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": C["grid"], "grid.linewidth": 0.5, "axes.titlelocation": "left",
    "axes.titleweight": "bold", "figure.dpi": 150, "savefig.dpi": 300, "pdf.fonttype": 42, "ps.fonttype": 42,
    "legend.frameon": False, "legend.handlelength": 1.8, "lines.solid_capstyle": "butt",
    "font.family": "DejaVu Sans", "mathtext.fontset": "dejavusans"})
kfmt = FuncFormatter(lambda v, _: f"{v / 1000:g}k" if v else "0")


def _f(x):
    return np.asarray(x, dtype=float)


def _test_window(ta, te):
    te = np.asarray(te, float)
    lo90 = np.flatnonzero(te < 0.9)
    j = (lo90[-1] + 1) if lo90.size else 0
    t1 = int(ta[j]) if j < len(te) else None
    end = j if t1 is not None else len(te)
    lo10 = np.flatnonzero(te[:end] < 0.1)
    i = (lo10[-1] + 1) if lo10.size else 0
    if i >= len(te):
        return None, None
    return int(ta[i]), t1


def load_compact(compact_dir, line, evidence=None):
    runs = []
    for js in sorted(Path(compact_dir).glob("*.json")):
        d = json.loads(js.read_text()); m = d["meta"]; cfg = m["config"]; s = d.get("score") or {}
        z = np.load(js.with_suffix(".npz"))
        g = lambda k: _f(z[k]) if k in z.files else None
        t = g("step"); ta = g("acc_step") if "acc_step" in z.files else t
        r = dict(id=m["run_id"], line=line, recipe=m.get("config_id") or cfg.get("config_id"), p=int(cfg["p"]),
                 evidence=evidence or m.get("evidence"), config=cfg, T=int(t[-1]), t=t, train_ce=g("train_ce"), test_ce=g("test_ce"),
                 ta=ta, train_acc=g("train_accuracy"), test_acc=g("test_accuracy"), to=g("obs_step"), AH=g("obs_hidden_AH_add"),
                 refit=g("obs_refit_test_accuracy"), Aplus=g("obs_availability_A_plus"), score=s, meta=m)
        r["tce_t"] = t
        if r["test_ce"] is None and "test_ce_logged" in z.files:
            r["test_ce"], r["tce_t"] = g("test_ce_logged"), ta
        if "obs_agop_q4_purity" in z.files:
            r.update(agop=g("obs_agop_q4_purity"), agop_valid=g("obs_agop_q4_valid"), agop_null=g("obs_agop_q4_null_median"),
                     agop_null_lo=g("obs_agop_q4_null_min"), agop_null_hi=g("obs_agop_q4_null_max"), agop_label="q = 4")
        elif "obs_agop_top4_fourier_sum_purity" in z.files:
            a4 = g("obs_agop_top4_fourier_sum_purity")
            r.update(agop=a4, agop_valid=np.where(np.isfinite(a4), 1.0, np.nan), agop_null=np.full_like(a4, np.nan),
                     agop_null_lo=np.full_like(a4, np.nan), agop_null_hi=np.full_like(a4, np.nan), agop_label="top-4")
        wins = cfg.get("windows") or []
        if wins:
            r["t0"], r["t1"], r["window_rule"] = int(wins[0]["t0"]), int(wins[0]["t1"]), "fixed"
        else:
            t10, t90 = _test_window(r["ta"], r["test_acc"])
            ts = fin(r["to"], r["AH"])[0]
            og = int(np.median(np.diff(ts))) if len(ts) > 1 else 1
            r["t10"], r["t90_stay"], r["obs_grid"] = t10, t90, og
            if t10 is None:
                r["t0"] = r["t1"] = None
            else:
                r["t0"] = int(og * math.floor(t10 / og))
                r["t1"] = int(min(r["T"], og * math.ceil((t90 if t90 is not None else r["T"]) / og)))
            r["window_rule"] = f"test-defined (held-out 10% -> stays >= 90%), snapped outward to the {og}-update observer grid"
        r["every_update"] = bool(len(t) == r["T"] + 1)
        runs.append(r)
    return add_derived(runs)


def at(t, y, step):
    i = np.searchsorted(t, step)
    return float(y[i]) if i < len(t) and t[i] == step else float("nan")


def endpoints(r):
    t0, t1 = r["t0"], r["t1"]
    out = dict(v_min=float("nan"), P10=False, P03=False, ce_min=float("nan"), CE01=False, F06=False, AGOPnull02=False,
               dAH=float("nan"), F02=False, dAGOP=float("nan"), dnull=float("nan"),
               AGOP02=False, rec90=bool(np.any(r["test_acc"] >= 0.9 - 1e-12)), fit100=None, t90=None, window_ok=t0 is not None)
    te = r["test_acc"]
    if out["rec90"]:
        out["t90"] = int(r["ta"][np.argmax(te >= 0.9 - 1e-12)])
    if t0 is None:
        return out
    sel = (r["t"] >= t0) & (r["t"] <= t1)
    L = r["train_ce"][sel]
    need = (t1 - t0 + 1) if r["every_update"] else sel.sum()
    ok = bool(sel.sum() == need and sel.sum() > 1 and np.all(np.isfinite(L)) and L.min() > 0)
    if ok:
        out["v_min"] = float((L.max() - L.min()) / L.min()); out["P10"] = out["v_min"] <= 0.10; out["P03"] = out["v_min"] <= 0.03
        out["ce_min"] = float(L.min()); out["CE01"] = out["ce_min"] >= 0.1
    ah0, ah1 = at(*fin(r["to"], r["AH"]), t0), at(*fin(r["to"], r["AH"]), t1)
    out["dAH"] = ah1 - ah0; out["F02"] = bool(out["dAH"] >= 0.02); out["F06"] = bool(out["dAH"] >= 0.06)
    a0, a1 = at(*fin(r["to"], r["agop"]), t0), at(*fin(r["to"], r["agop"]), t1)
    va = at(*fin(r["to"], r["agop_valid"]), t0) == 1 and at(*fin(r["to"], r["agop_valid"]), t1) == 1
    if va:
        out["dAGOP"] = a1 - a0; out["AGOP02"] = bool(out["dAGOP"] >= 0.02)
    out["dnull"] = at(*fin(r["to"], r["agop_null"]), t1) - at(*fin(r["to"], r["agop_null"]), t0)
    out["AGOPnull02"] = bool(np.isfinite(out["dAGOP"]) and np.isfinite(out["dnull"]) and out["dAGOP"] - out["dnull"] >= 0.02)
    sa = (r["ta"] >= t0) & (r["ta"] <= t1)
    out["fit100"] = bool(np.all(r["train_acc"][sa] >= 1.0))
    out["test_t0"], out["test_t1"] = at(r["ta"], te, t0), at(r["ta"], te, t1)
    return out


def wilson(k, n, z=1.959963984540054):
    if n == 0:
        return (float("nan"), float("nan"))
    ph = k / n; den = 1 + z * z / n
    c = (ph + z * z / (2 * n)) / den; h = z * math.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n)) / den
    return max(0.0, c - h), min(1.0, c + h)


def fin(tx, y):
    tx, y = np.asarray(tx, float), np.asarray(y, float)
    ok = np.isfinite(y)
    return tx[ok], y[ok]


def _window(ax, r):
    ax.axvspan(r["t0"], r["t1"], color=C["window"], lw=0, zorder=0)


def _xk(ax, T, nb=5):
    ax.set_xlim(0, T); ax.xaxis.set_major_locator(MaxNLocator(nb, integer=True)); ax.xaxis.set_major_formatter(kfmt)


def _save(fig, out, stem):
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / f"{stem}.pdf"); fig.savefig(out / f"{stem}.png", dpi=300)
    plt.close(fig)


def recipe_label(r):
    c = r["config"]
    return f"p = {r['p']}, c = {c['c']:g}, $\\eta_V$ = {c['eta_V']:g}, $\\lambda_W$ = {c['decay_W']:g}"


def hero(r, out, stem, dloc="upper left", logp=False, xmax=None, ce_scale="log", feat="all"):
    e = endpoints(r)
    fig, axs = plt.subplots(2, 2, figsize=(TEXTW, 3.55), constrained_layout=True)
    T = r["T"] if xmax is None else xmax; t = r["t"]
    ax = axs[0, 0]; _window(ax, r)
    ax.plot(t, r["train_ce"], color=C["train"], lw=1.0, label="training CE")
    ax.plot(r["tce_t"], r["test_ce"], color=C["test"], lw=1.0, label="held-out CE")
    ax.set_yscale(ce_scale)
    if ce_scale == "linear":
        ax.set_ylim(bottom=0)
    if logp:
        ax.axhline(math.log(r["p"]), color=C["muted"], lw=0.8, ls="--", label="log p (uniform guess)")
    _xk(ax, T); ax.set_ylabel("cross-entropy"); ax.set_title("(a) Loss over the run")
    ax.legend(loc="upper right" if logp else "center right", fontsize=6.3)
    ax = axs[0, 1]
    sel = (t >= r["t0"]) & (t <= r["t1"])
    L = r["train_ce"][sel]; base = L.min()
    ax.plot(t[sel], 100 * (L / base - 1), color=C["train"], lw=0.8)
    ax.set_xlim(r["t0"], r["t1"]); ax.xaxis.set_major_locator(MaxNLocator(5, integer=True)); ax.xaxis.set_major_formatter(kfmt)
    ax.set_ylim(0, max(13, 100 * e["v_min"] * 1.35))
    ax.axhline(10, color=C["muted"], lw=0.7, ls="--"); ax.text(r["t1"], 10.3, "10% ", fontsize=6.5, color=C["muted"], va="bottom", ha="right")
    ax.set_ylabel("CE above window min (%)")
    ax.set_title("(b) Training CE on the window")
    ax.text(0.02, 0.97, f"(max − min)/min = {100 * e['v_min']:.1f}%", transform=ax.transAxes, ha="left", va="top",
            fontsize=6.5, color=C["ink"])
    ax.text(0.98, 0.97, f"min CE = {base:.2g}", transform=ax.transAxes, ha="right", va="top", fontsize=6.5, color=C["muted"])
    ax = axs[1, 0]; _window(ax, r)
    ax.plot(r["ta"], r["train_acc"], color=C["train"], lw=1.0, label="train")
    ax.plot(r["ta"], r["test_acc"], color=C["test"], lw=1.2, label="held-out")
    ax.plot(*fin(r["to"], r["refit"]), color=C["refit"], lw=1.2, ls=(0, (1.2, 1.2)), label="held-out, ridge refit\n(train rows only)")
    _xk(ax, T); ax.set_ylim(-0.02, 1.03); ax.set_ylabel("accuracy"); ax.set_xlabel("update")
    ax.set_title("(c) Accuracy"); ax.legend(loc="lower right", fontsize=6.3)
    ax = axs[1, 1]; _window(ax, r)
    if feat == "AH":
        ta_, ya_ = fin(r["to"], r["AH"])
        ax.plot(ta_, ya_, color=C["AH"], lw=1.4)
        _xk(ax, T); ax.set_ylim(0, 1.18 * np.nanmax(ya_)); ax.set_ylabel("$A_H$"); ax.set_xlabel("update")
        ax.set_title("(d) Hidden Fourier fraction")
        a0, a1 = at(ta_, ya_, r["t0"]), at(ta_, ya_, r["t1"])
        ax.annotate("", xy=(r["t1"], a1), xytext=(r["t1"], a0), arrowprops=dict(arrowstyle="<->", color=C["ink"], lw=0.8))
        ax.text(r["t1"] + 0.015 * T, (a0 + a1) / 2, f"$\\Delta A_H$ = {a1 - a0:.3f}", fontsize=6.8, color=C["ink"], va="center")
        _save(fig, out, stem)
        return e
    tl, lo = fin(r["to"], r["agop_null_lo"]); th, hi = fin(r["to"], r["agop_null_hi"])
    if len(tl):
        ax.fill_between(tl, lo, hi, color=C["null"], alpha=0.18, lw=0)
        ax.plot(*fin(r["to"], r["agop_null"]), color=C["null"], lw=0.9, ls=(0, (2, 1.5)), label="AGOP null, 8 random frames")
    ax.plot(*fin(r["to"], np.where(r["agop_valid"] == 1, r["agop"], np.nan)), color=C["agop"], lw=1.2,
            label=f"AGOP purity ({r.get('agop_label', 'q = 4')})")
    ax.plot(*fin(r["to"], r["AH"]), color=C["AH"], lw=1.2, label="hidden Fourier $A_H$")
    _xk(ax, T); ax.set_ylim(0, 1); ax.set_ylabel("fraction"); ax.set_xlabel("update")
    ax.set_title("(d) Feature alignment"); ax.legend(loc=dloc, fontsize=6.3)
    _save(fig, out, stem)
    return e


def _grid(runs, key_t, key_y, grid):
    M = np.full((len(runs), len(grid)), np.nan)
    for i, r in enumerate(runs):
        tt, yy = fin(r[key_t], r[key_y])
        idx = np.searchsorted(tt, grid)
        ok = idx < len(tt)
        ok[ok] &= tt[idx[ok]] == grid[ok]
        M[i, ok] = yy[idx[ok]]
    return M


def cohort(groups, out, stem, col_titles=None, rows=None, ce_row=True, ce_log=True):
    rows = list(rows or [("ta", "test_acc", C["test"], "held-out accuracy", 100),
                         ("to", "AH", C["AH"], "hidden Fourier $A_H$", 100),
                         ("to", "agop_minus_null", C["agop"], "AGOP purity − null", 500)])
    if ce_row:
        T_all = max(r["T"] for g_ in groups for r in g_)
        native = int(np.median(np.diff(groups[0][0]["t"])))
        stride = native * max(1, int(math.ceil(T_all / 1500 / native)))
        rows.insert(0, ("t", "train_ce", C["train"], "training CE", stride, ce_log))
    z = 1 if ce_row else 0
    nr = 1 + len(rows)
    n = len(groups)
    if n == 1:
        fig = plt.figure(figsize=(TEXTW, 3.5), constrained_layout=True)
        top = 2 if ce_row else 1; bot = nr - top
        gs = fig.add_gridspec(2, 6)
        cells = [gs[0, k * (6 // top):(k + 1) * (6 // top)] for k in range(top)] + \
                [gs[1, k * (6 // bot):(k + 1) * (6 // bot)] for k in range(bot)]
        axs = np.array([fig.add_subplot(c) for c in cells], dtype=object).reshape(-1, 1)
    else:
        fig, axs = plt.subplots(nr, n, figsize=(TEXTW, (1.18 if nr > 4 else 1.35) * nr + 0.2), constrained_layout=True,
                                squeeze=False, sharey="row")
    for j, runs in enumerate(groups):
        r0 = runs[0]; T = max(r["T"] for r in runs); t0, t1 = r0["t0"], r0["t1"]
        shared = all((r["t0"], r["t1"]) == (t0, t1) for r in runs)
        ax = axs[z, j]
        for r in runs:
            if r["t0"] is None:
                continue
            sel = (r["t"] >= r["t0"]) & (r["t"] <= r["t1"]); L = r["train_ce"][sel]
            x = r["t"][sel] if shared else (r["t"][sel] - r["t0"]) / max(1, r["t1"] - r["t0"])
            ax.plot(x, 100 * (L / L.min() - 1), color=C["train"], lw=0.5, alpha=0.6,
                    marker="o" if not r["every_update"] else None, ms=1.2)
        ax.axhline(10, color=C["muted"], lw=0.7, ls="--")
        if shared:
            ax.set_xlim(t0, t1); ax.xaxis.set_major_locator(MaxNLocator(3, integer=True)); ax.xaxis.set_major_formatter(kfmt)
        else:
            ax.set_xlim(0, 1); ax.set_xticks([0, 0.5, 1]); ax.set_xticklabels(["$t_0$", "", "$t_1$"])
            ax.set_xlabel("own window, rescaled", fontsize=6.5, labelpad=1)
        axs[0, j].set_title(col_titles[j] if col_titles else f"p = {r0['p']}", fontsize=7.5, loc="left")
        for i, spec in enumerate(rows):
            kt, ky, col, lab, step = spec[:5]; logy = len(spec) > 5 and spec[5]
            ax = axs[i + (1 if (ce_row and i > 0) or not ce_row else 0), j]
            if shared:
                _window(ax, r0)
            else:
                ax.axvspan(np.median([r["t0"] for r in runs]), np.median([r["t1"] for r in runs]), color=C["window"], lw=0, zorder=0)
            g = np.arange(0, T + 1, step)
            M = _grid(runs, kt, ky, g)
            for row in M:
                ok = np.isfinite(row)
                ax.plot(g[ok], row[ok], color=col, lw=0.45, alpha=0.45)
            cnt = np.sum(np.isfinite(M), 0)
            med = np.where(cnt == len(runs), np.nanmedian(np.where(np.isfinite(M), M, np.nan), 0) if M.size else np.nan, np.nan)
            ax.plot(g, med, color=col, lw=1.4)
            if logy:
                ax.set_yscale("log")
            _xk(ax, T, (5 if n <= 3 else 3) if n > 1 else 3)
            if j == 0:
                ax.set_ylabel(lab)
            if n == 1:
                ax.set_xlabel("update")
        axs[-1, j].set_xlabel("update")
    axs[z, 0].set_ylabel("training CE above\nwindow min (%)")
    _save(fig, out, stem)


def add_derived(runs):
    for r in runs:
        r["agop_minus_null"] = np.where(r["agop_valid"] == 1, r["agop"], np.nan) - r["agop_null"]
    return runs


def summary(groups, out, stem, labels, strict_joint=True, short=None, xlab="p", keys=None, ref_dAH=0.02):
    short = short or labels
    E = [[endpoints(r) for r in runs] for runs in groups]
    for runs, es in zip(groups, E):
        for r, e in zip(runs, es):
            e["joint"] = bool((r.get("score") or {}).get("joint_PAG10_F02")) if r.get("score") else None
    keys = list(keys) if keys else [("P10", "CE range\n≤ 10%"), ("P03", "CE range\n≤ 3%"), ("F02", "$\\Delta A_H$\n≥ 0.02"),
            ("AGOP02", "$\\Delta$AGOP\n≥ 0.02"), ("rec90", "held-out\n≥ 90%")]
    if strict_joint and all(e["joint"] is not None for es in E for e in es):
        keys.append(("joint", "strict\njoint"))
    ng = len(groups)
    if ng == 1:
        fig, axs = plt.subplots(1, 4, figsize=(TEXTW, 2.05), constrained_layout=True, gridspec_kw={"width_ratios": [2.1, 1, 1, 1]})
    else:
        fig = plt.figure(figsize=(TEXTW, 3.6 if ng <= 3 else 3.9), constrained_layout=True)
        gs = fig.add_gridspec(2, 3, height_ratios=[1, 1.05])
        axs = [fig.add_subplot(gs[0, :]), fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1]), fig.add_subplot(gs[1, 2])]
    ax = axs[0]; ax.set_label("categorical")
    off = np.linspace(-0.08 * (ng - 1), 0.08 * (ng - 1), ng) if ng > 1 else [0.0]
    marks = ["o", "s", "D", "^", "v", "P"]
    for gi, es in enumerate(E):
        n = len(es)
        for ki, (k, lab) in enumerate(keys):
            kk = sum(bool(e[k]) for e in es); lo, hi = wilson(kk, n)
            x = ki + off[gi]
            ax.plot([x, x], [lo, hi], color=C["ink"], lw=0.7)
            ax.plot(x, kk / n, marker=marks[gi % len(marks)], color=C["ink"], ms=3.2, mfc="white" if gi % 2 else C["ink"], ls="none",
                    label=labels[gi] if ki == 0 else None)
    ax.set_xticks(range(len(keys)))
    if ng > 1:
        ax.set_xticklabels([lab for _, lab in keys], fontsize=6.5)
    else:
        ax.set_xticklabels([lab.replace("\n", " ") for _, lab in keys], fontsize=6, rotation=30, ha="right", rotation_mode="anchor")
    ax.set_xlim(-0.5, len(keys) - 0.5)
    ax.set_ylim(-0.03, 1.03); ax.set_yticks([0, 0.5, 1]); ax.set_ylabel("fraction of runs"); ax.grid(axis="x", visible=False)
    ax.set_title("(a) Endpoints (Wilson 95%)", fontsize=7.5)
    if ng > 1:
        ax.legend(loc="upper left", bbox_to_anchor=(1.005, 1.0), ncol=1, fontsize=6.3, handletextpad=0.2, borderaxespad=0)
    else:
        ax.set_title(f"(a) Endpoints ({len(E[0])} runs)", fontsize=7.5)
    rng = np.random.default_rng(0)
    panels = ((axs[1], "v_min", 100, "(max − min)/min (%)", 10, "(b) CE range on window", C["train"]),
              (axs[2], "dAH", 1, "gain on window", ref_dAH, "(c) $\\Delta A_H$", C["AH"]),
              (axs[3], "dAGOP", 1, "gain on window", 0.02, "(d) $\\Delta$AGOP", C["agop"]))
    for ax, key, mul, lab, ref, title, col in panels:
        for gi, es in enumerate(E):
            v = np.array([e[key] for e in es], float) * mul
            x = gi + rng.uniform(-0.17, 0.17, len(v))
            ax.plot(x, v, "o", ms=2.3, color=col, alpha=0.85, mec="none")
            if np.isfinite(v).any():
                ax.plot([gi - 0.28, gi + 0.28], [np.nanmedian(v)] * 2, color=C["ink"], lw=1.0)
            if key == "dAGOP":
                nv = np.array([e["dnull"] for e in es], float)
                ax.plot(gi + rng.uniform(-0.17, 0.17, len(nv)), nv, "o", ms=2.1, color=C["null"], alpha=0.8, mec="none")
        ax.axhline(ref, color=C["muted"], lw=0.7, ls="--")
        ax.set_xticks(range(ng)); ax.set_xticklabels(short, fontsize=6.5); ax.set_xlim(-0.6, ng - 0.4); ax.set_xlabel(xlab, fontsize=7, labelpad=1)
        ax.set_ylabel(lab, fontsize=6.5); ax.set_title(title if ng > 1 else title.replace(" on window", ""), fontsize=7.5)
        ax.grid(axis="x", visible=False)
        if key != "dAGOP":
            ax.set_ylim(bottom=0)
    _save(fig, out, stem)
    return E
