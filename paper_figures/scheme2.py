#!/usr/bin/env python3
import argparse, glob, json, math, os, sys, tarfile, tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig_scheme2")
sys.dont_write_bytecode = True
sys.path.insert(0, str(HERE))
import _paths as P
import numpy as np
import median_figs as MF
import win
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator, NullLocator

REFIT = "#7B3294"
K, KFMT, E = dict(MF.K, refit=REFIT), MF.KFMT, MF.E
LAST, FIRST = MF.LAST, MF.FIRST
V2 = P.WINDOW_RESULTS
GAIN = 0.2


def new_window(c, start):
    start = int(min(max(start, 0), len(c) - 1)); amin = win.stretches(c, start); n = len(c)
    S = [(int(amin[t]), t) for t in range(start, n) if t == n - 1 or amin[t + 1] != amin[t]]
    return max(S, key=lambda s: (float(np.max(c[s[0]:s[1] + 1]) - c[s[0]]), s[1] - s[0]))


def dip_start(c, tol=1.03):
    amin = win.stretches(c, 0, tol); n = len(c)
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
    r.update(W=W, primary=bool(g_te >= GAIN or g_rf >= GAIN), g_te=g_te, g_rf=g_rf, peak=int(t_c[pk]),
             level=float(c[pk]), start=int(t_c[min(start, len(c) - 1)]))
    return r


def v2_verdicts():
    out = {}
    for f in sorted(glob.glob(str(V2 / "*" / "settings.jsonl"))):
        for line in open(f):
            for run in json.loads(line)["runs"]:
                out[run["id"]] = (run["W2"], run["ok2"], run["g_te"], run["g_rf"])
    return out


def deep_run(d, rid, verdict):
    r = MF.run_series(d); L = r["L"]; c = (L[:-1] + L[1:]) / 2 / math.log(r["p"]); t_c = np.arange(len(c))
    W, ok, g_te, g_rf = verdict
    pk = W[0] + int(np.argmax(c[W[0]:W[1] + 1]))
    r.update(ce=(t_c, c), W=list(W), primary=bool(ok), g_te=g_te, g_rf=g_rf, peak=int(pk), level=float(c[pk]), id=rid)
    return r


def dev_run(d, rid):
    r = MF.run_series(d); L = r["L"]; c = (L[:-1] + L[1:]) / 2 / math.log(r["p"]); t_c = np.arange(len(c))
    r.update(ce=(t_c, c), id=rid)
    return score(r, t_c, c, E.run_fields(d)["knee"] or 0)


def onehl_run(npz):
    z = np.load(npz); j = json.loads(Path(npz[:-4] + ".json").read_text()); m = j["meta"]; cfg = m["config"]; p = int(cfg["p"])
    s = z["step"].astype(float); L = z["train_ce"].astype(float)
    ta = z["acc_step"].astype(float) if "acc_step" in z.files else s
    te, tr = z["test_accuracy"].astype(float), z["train_accuracy"].astype(float)
    to = z["obs_step"].astype(float); rf = z["obs_refit_test_accuracy"].astype(float); ah = z["obs_hidden_AH_add"].astype(float)
    every = len(s) == int(s[-1]) + 1
    if every: c = (L[:-1] + L[1:]) / 2 / math.log(p); t_c = s[:-1]
    else: c = L / math.log(p); t_c = s
    ntr = int(m.get("n") or cfg.get("n_train") or round(cfg.get("train_fraction", 0.8) * p * p))
    ok_r, ok_a = np.isfinite(rf), np.isfinite(ah)
    t90 = ta[np.argmax(te >= 0.9)] if np.any(te >= 0.9) else None
    r = dict(id=m["run_id"], recipe=m.get("config_id") or cfg.get("config_id"), p=p, depth=1, cfg=cfg, ce=(t_c, c),
             test=(ta, te), train=(ta, tr), err=(ta, (1 - tr) * ntr), refit=(to[ok_r], rf[ok_r]), ahL=(to[ok_a], ah[ok_a]),
             ah1=None, agop={"first": None, "last": None}, mse=None, refit_kind="canonical", share=0.0, why=None,
             t90=float(t90) if t90 is not None else None, ntr=ntr, T=int(s[-1]))
    lv = cfg.get("decay_V")
    if "V_norm" in z.files and lv:
        k = np.flatnonzero(z["V_norm"] >= 0.99 / lv)
        start = int(np.searchsorted(t_c, s[k[0]])) if len(k) else 0; r["start_rule"] = "knee"
    else:
        start = dip_start(c); r["start_rule"] = "CE minimum before the largest climb"
    return score(r, t_c, c, start)


def onehl_groups(runs, key):
    d = {}
    for r in runs: d.setdefault(r[key], []).append(r)
    return [sorted(d[k], key=lambda r: r["id"]) for k in sorted(d, key=lambda k: (d[k][0]["p"], str(k)))]


def column2(runs, title, ids):
    t90 = [r["t90"] or r["W"][1] for r in runs]
    xmax = int(math.ceil((1.15 * max(np.median([r["W"][1] for r in runs]), np.percentile(t90, 90)) + 1500) / 1000) * 1000)
    xmax = min(xmax, int(max(r["ce"][0][-1] for r in runs)))
    g = np.arange(0, xmax + 1, max(10, xmax // 2000), dtype=float)
    on = MF.on_grid
    S = dict(ce=np.array([on(*r["ce"], g) for r in runs]),
             test=np.array([on(*r["test"], g) for r in runs]), train=np.array([on(*r["train"], g) for r in runs]),
             err=np.array([on(*r["err"], g) for r in runs]), refit=np.array([on(*r["refit"], g) for r in runs]),
             ahL=np.array([on(*r["ahL"], g) for r in runs]),
             ah1=np.array([on(*r["ah1"], g) for r in runs]) if runs[0]["ah1"] is not None else None)
    for name in ("first", "last"):
        ok = all(r["agop"][name] is not None for r in runs)
        S[f"ag_{name}"] = np.array([on(*r["agop"][name][:2], g) for r in runs]) if ok else None
        S[f"agr_{name}"] = np.array([on(*r["agop"][name][2:], g) for r in runs]) if ok else None
    S["mse"] = np.array([on(*r["mse"], g) for r in runs]) if all(r["mse"] is not None for r in runs) else None
    band = (float(np.median([r["W"][0] for r in runs])), float(np.median([r["W"][1] for r in runs])))
    stats = dict(title=title, n=len(runs), passes=int(sum(r["primary"] for r in runs)), band_median=band,
                 peak_median=float(np.median([r["peak"] for r in runs])),
                 runs=[dict(id=r["id"], W=r["W"], primary=r["primary"], g_te=round(r["g_te"], 4), g_rf=round(r["g_rf"], 4),
                            peak=r["peak"], start=r.get("start"), start_rule=r.get("start_rule", "knee"),
                            verdict=r.get("verdict", "scored here")) for r in runs],
                 refit=sorted({r["refit_kind"] for r in runs}))
    return dict(g=g, S=S, band=band, xmax=xmax, title=title, stats=stats, ntr=max(r["ntr"] for r in runs))


def ce_log_lim(cols):
    pos = np.concatenate([np.median(c["S"]["ce"], 0)[c["g"] > 0.05 * c["xmax"]] for c in cols])
    return (10 ** np.floor(np.log10(max(pos[pos > 0].min() * 0.5, 1e-12))), 1.5)


def draw2(cols, out, stem, rows=("ce", "acc", "err", "ah"), height=5.2, ce_log=False, ce_zoom=False,
          ce_title="Training CE / log p (cycle mean)", ce_lim=None, ah_top=None, align_titles=False, legend=True):
    n = len(cols); single = n == 1
    if single:
        fig, a1 = plt.subplots(1, len(rows), figsize=(6.75, height), layout="constrained")
        axs = np.asarray(a1).reshape(len(rows), 1)
    else:
        fig, axs = plt.subplots(len(rows), n, figsize=(6.75, height), sharey=False if ce_zoom else "row", layout="constrained")
        axs = np.atleast_2d(axs)
        if ce_zoom:
            for i, row in enumerate(rows):
                for j in range(1, n):
                    if row != "ce": axs[i, j].sharey(axs[i, 0]); axs[i, j].tick_params(labelleft=False)
    if ah_top is None: ah_top = max(float(np.nanmax(c["S"]["ahL"])) for c in cols) * 1.08
    err_top = max(c["ntr"] for c in cols) * 1.5
    agop_top = max([float(np.nanmax(c["S"][k])) for c in cols for k in ("ag_first", "ag_last", "agr_first", "agr_last")
                    if c["S"].get(k) is not None] or [1.0]) * 1.08
    mse_top = max([float(np.nanmax(np.percentile(c["S"]["mse"], 90, axis=0))) for c in cols if c["S"].get("mse") is not None]
                  or [1.0]) * 1.15
    if ce_log and ce_lim is None: ce_lim = ce_log_lim(cols)
    for j, c in enumerate(cols):
        g, S = c["g"], c["S"]
        for i, row in enumerate(rows):
            ax = axs[i, j]
            if c["band"][1] - c["band"][0] > 0.004 * c["xmax"]: ax.axvspan(*c["band"], color=K["window"], lw=0, zorder=0)
            ax.grid(True, axis="y", color=K["grid"], lw=0.6); ax.set_xlim(0, c["xmax"])
            ax.xaxis.set_major_formatter(KFMT); ax.xaxis.set_major_locator(MaxNLocator(2 if n > 5 else 3, integer=True))
            thin = dict(lw=0.45, alpha=0.28, zorder=1)
            if row == "ce":
                for y in S["ce"]: ax.plot(g, y, color=K["train"], **thin)
                ax.plot(g, np.median(S["ce"], 0), color=K["train"], lw=1.4, zorder=3)
                if ce_zoom:
                    m = np.median(S["ce"], 0); sel = g >= c["band"][0]
                    lo, hi = float(np.min(m[sel])), float(np.max(m[sel])); pad = 0.12 * (hi - lo)
                    ax.set_ylim(lo - pad, hi + 1.6 * pad); ax.yaxis.set_major_locator(MaxNLocator(3))
                    ax.ticklabel_format(axis="y", style="sci", scilimits=(-3, 3), useMathText=True)
                elif ce_log:
                    ax.set_yscale("log"); ax.set_ylim(*ce_lim); ax.yaxis.set_minor_locator(NullLocator())
                    pw = np.arange(np.ceil(np.log10(ce_lim[0])), 1); st = max(1, int(np.ceil(len(pw) / 4)))
                    ax.set_yticks(10.0 ** pw[::-st][::-1])
                else: ax.set_ylim(0, 1.05)
            elif row == "acc":
                for y in S["test"]: ax.plot(g, y, color=K["test"], **thin)
                ax.plot(g, np.median(S["train"], 0), color=K["train_acc"], lw=1.0, ls="--", zorder=3)
                ax.plot(g, np.median(S["refit"], 0), color=K["refit"], lw=1.4, zorder=4)
                ax.plot(g, np.median(S["test"], 0), color=K["test"], lw=1.4, zorder=5); ax.set_ylim(-0.02, 1.04)
            elif row == "err":
                for y in S["err"]: ax.plot(g, y, color=K["train_acc"], **thin)
                ax.plot(g, np.median(S["err"], 0), color=K["train_acc"], lw=1.1, zorder=3)
                ax.set_yscale("symlog", linthresh=1); ax.set_ylim(-0.2, err_top)
                ticks = [t for t in (0, 1, 10, 100, 1000, 10000) if t <= err_top]
                ax.set_yticks(ticks); ax.set_yticklabels([{1000: "1k", 10000: "10k"}.get(t, str(t)) for t in ticks])
            elif row == "agop":
                if S["ag_last"] is not None:
                    for y in S["ag_last"]: ax.plot(g, y, color=K["agop"], **thin)
                    ax.plot(g, np.median(S["agr_last"], 0), color=K["null"], lw=0.9, ls=":", zorder=2)
                    ax.plot(g, np.median(S["ag_last"], 0), color=K["agop"], lw=1.4, ls=LAST, zorder=3)
                if S["ag_first"] is not None:
                    ax.plot(g, np.median(S["agr_first"], 0), color=K["null"], lw=0.9, ls="--", zorder=2)
                    ax.plot(g, np.median(S["ag_first"], 0), color=K["agop"], lw=1.2, ls=FIRST, zorder=3)
                ax.set_ylim(0, agop_top)
            elif row == "mse":
                for y in S["mse"]: ax.plot(g, y, color=K["refit"], **thin)
                ax.plot(g, np.median(S["mse"], 0), color=K["refit"], lw=1.4, zorder=3); ax.set_ylim(0, mse_top)
            elif row == "ah":
                for y in S["ahL"]: ax.plot(g, y, color=K["AH"], **thin)
                ax.plot(g, np.median(S["ahL"], 0), color=K["AH"], lw=1.4, ls=LAST, zorder=3)
                if S["ah1"] is not None: ax.plot(g, np.median(S["ah1"], 0), color=K["AH"], lw=1.2, ls=FIRST, zorder=3)
                ax.set_ylim(0, ah_top)
            if i < len(rows) - 1 and not single: ax.tick_params(labelbottom=False)
        st = c["stats"]
        if single: fig.suptitle(c["title"], fontsize=8)
        else: axs[0, j].set_title(c["title"], loc="center")
    lab = dict(ce="Training CE / log p\n(cycle mean)", acc="Accuracy", err="Misclassified\ntraining pairs", ah="Hidden Fourier\n$A_H$",
               agop="Hidden AGOP purity\n(class-centered, q = 4)", mse="Refit held-out\nMSE")
    if height < 4:
        lab.update(ce="Train CE\n/ log p", ah="Hidden\nFourier $A_H$")
    if single:
        ttl = dict(ce=ce_title, acc="Accuracy", ah="Hidden Fourier $A_H$", err="Misclassified training pairs")
        for i, row in enumerate(rows): axs[i, 0].set_title(ttl[row], loc="left")
        axs[len(rows) // 2, 0].set_xlabel("Updates (thousands)")
    else:
        for i, row in enumerate(rows): axs[i, 0].set_ylabel(lab[row])
        for j in ((0, 1) if n == 2 else (n // 2,)): axs[-1, j].set_xlabel("Updates (thousands)")
    h = [Line2D([], [], color=K["train"], lw=1.4), Line2D([], [], color=K["train_acc"], lw=1.0, ls="--"),
         Line2D([], [], color=K["test"], lw=1.4), Line2D([], [], color=K["refit"], lw=1.4)]
    l = ["Train CE", "Train acc.", "Held-out", "Refit"]
    if "err" in rows: h.append(Line2D([], [], color=K["train_acc"], lw=1.1)); l.append("Misclassified")
    if "agop" in rows:
        h += [Line2D([], [], color=K["agop"], lw=1.4), Line2D([], [], color=K["null"], lw=0.9, ls="--")]
        l += ["AGOP purity", "Random frames"]
    if "ah" in rows and any(c["S"]["ah1"] is not None for c in cols):
        h += [Line2D([], [], color=K["layer"], lw=1.4, ls=LAST), Line2D([], [], color=K["layer"], lw=1.2, ls=FIRST)]
        l += ["Last hidden", "First hidden"]
    elif "ah" in rows: h.append(Line2D([], [], color=K["AH"], lw=1.4)); l.append("$A_H$")
    if legend: fig.legend(h, l, loc="outside lower center", ncol=len(h) if len(h) <= 7 else (len(h) + 1) // 2, frameon=False,
               handlelength=1.8, columnspacing=0.9, fontsize=8)
    if align_titles:
        fig.canvas.draw(); top = max(ax.title.get_position()[1] for ax in axs[0])
        for ax in axs[0]: ax.set_title(ax.get_title(loc="center"), loc="center", y=top)
    save(fig, out, stem)


def save(fig, out, stem):
    out.mkdir(parents=True, exist_ok=True)
    fig.canvas.draw()
    fig.savefig(out / f"{stem}.pdf", metadata={"Title": stem, "Author": "", "Creator": "scheme2.py", "CreationDate": None})
    fig.savefig(out / f"{stem}.png", dpi=200); plt.close(fig)


HEAD_W, HEAD_H = 3.30, 2.30
HEAD = dict(bottom=0.49, gap=0.405, top=0.205)
SLOTS = [(.12, .335), (.645, .315)]


def hero2(runs, out, stem="H_hero_2x2", x0=1000, x1=16000):
    n = len(runs); p = runs[0]["p"]; lp = math.log(p)
    fig = plt.figure(figsize=(HEAD_W, HEAD_H))
    h = (HEAD_H - HEAD["bottom"] - HEAD["gap"] - HEAD["top"]) / 2
    rows = [HEAD["bottom"] + h + HEAD["gap"], HEAD["bottom"]]
    axs = [fig.add_axes([SLOTS[c][0], rows[r] / HEAD_H, SLOTS[c][1], h / HEAD_H]) for r, c in ((0, 0), (0, 1), (1, 0), (1, 1))]
    g = np.arange(x0, x1 + 1, 10, dtype=float); on = MF.on_grid
    band = (float(np.median([r["W"][0] for r in runs])), float(np.median([r["W"][1] for r in runs])))
    locked_window = (4500, 12000)
    thin = dict(lw=0.45, alpha=0.28, zorder=1)
    C = np.array([on(*r["ce"], g) for r in runs]) * lp
    HE = np.array([on(r["ce"][0], r["test_ce"], g) for r in runs]) * lp
    runs_peak = float(np.median([r["peak"] for r in runs]))
    inb = (g >= band[0]) & (g <= band[1]); curve_peak = float(g[inb][np.argmax(np.median(C, 0)[inb])])
    peak = runs_peak
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
    T = np.array([on(*r["test"], g) for r in runs])
    for y in T: ax.plot(g, y, color=K["test"], **thin)
    ax.plot(g, np.median(np.array([on(*r["train"], g) for r in runs]), 0), color=K["train_acc"], lw=1.0, ls="--", zorder=3)
    ax.plot(g, np.median(np.array([on(*r["refit"], g) for r in runs]), 0), color=K["refit"], lw=1.4, zorder=4)
    ax.plot(g, np.median(T, 0), color=K["test"], lw=1.4, zorder=5)
    ax.set_ylim(-0.04, 1.06); ax.set_yticks([0, 1]); ax.set_title("Accuracy", loc="center", fontsize=8.5)
    ax = axs[3]
    A = np.array([on(*r["ahL"], g) for r in runs])
    for y in A: ax.plot(g, y, color=K["AH"], **thin)
    ax.plot(g, np.median(A, 0), color=K["AH"], lw=1.4, zorder=3)
    ax.set_ylim(0, 0.13); ax.set_yticks([0, 0.06, 0.12]); ax.set_title(r"Hidden Fourier $A_H$", loc="center", fontsize=8.5)
    fig.text(.53, .078, "Updates (thousands)", ha="center", va="bottom", fontsize=8)
    hh = [Line2D([], [], color=K["train"], lw=1.1), Line2D([], [], color=K["train_acc"], lw=1.0, ls="--"),
          Line2D([], [], color=K["test"], lw=1.1), Line2D([], [], color=K["refit"], lw=1.1)]
    fig.legend(hh, ["Train CE", "Train acc.", "Held-out", "Refit"], loc="lower center", bbox_to_anchor=(.53, .010), ncol=4,
               frameon=False, handlelength=1.3, handletextpad=.35, columnspacing=.7, borderaxespad=0, borderpad=0, fontsize=8)
    save(fig, out, stem)
    T90 = [r["t90"] for r in runs]
    return dict(n=n, band_median=band, peak_median=peak, peak_of_median_curve=curve_peak, peaks=[r["peak"] for r in runs],
                shaded_window=list(locked_window), shaded_window_kind="locked_evaluation",
                windows=[r["W"] for r in runs], starts=[r["start"] for r in runs], t90=T90, passes=int(sum(r["primary"] for r in runs)),
                peak_minus_t90_median=float(np.median([r["peak"] - t for r, t in zip(runs, T90) if t is not None])))


ONEHL = ["H_hero_2x2", "fig2_grid", "H_cohort", "l1_phase_grid", "L1_cohort_fullCE", "L2_cohort_p11_p13_p17", "L2_cohort_p19_p23_p29",
         "L3_cohort", "L4_cohort", "locked_cohorts_1hl", "L2_cohort_all"]
P23B = P.P23B
DEEP = ["fig3_depths_p31", "campaign_medians_D345", "acrossp_1hl", "acrossp_depth2", "acrossp_depth3", "acrossp_depth4",
        "fig2_acrossp_1hl", "fig3_depths_2to5", "acrossp_depth3_3row", "acrossp_depth4_3row"]
DEV = ["deep_development_overview"]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("names", nargs="*")
    a = ap.parse_args(); out = a.out.resolve(); want = a.names or ONEHL + DEEP + DEV
    report = {}
    if set(want) & set(ONEHL):
        src = {"H": P.H, "L1": P.L1, "L2": P.L2, "L3": P.L3, "L4": P.L4}
        need = {"H_hero_2x2": "H", "fig2_grid": "P23B L1 H", "H_cohort": "H", "l1_phase_grid": "L1", "L1_cohort_fullCE": "L1",
                "L2_cohort_p11_p13_p17": "L2", "L2_cohort_p19_p23_p29": "L2", "L3_cohort": "L3", "L4_cohort": "L4",
                "locked_cohorts_1hl": "P23B L1", "L2_cohort_all": "L2"}
        keys = sorted({k for w in want if w in need for k in need[w].split()})
        with tempfile.TemporaryDirectory() as tmp:
            if "P23B" in keys:
                with tarfile.open(P23B) as t:
                    try: t.extractall(tmp, filter="data")
                    except TypeError: t.extractall(tmp)
                src["P23B"] = Path(tmp) / "compactB"
            R = {k: [onehl_run(f) for f in sorted(glob.glob(str(src[k] / "*.npz")))] for k in keys}
        for r in R.get("H", []):
            z = np.load([f for f in glob.glob(str(src["H"] / "*.npz")) if Path(f).stem == r["id"]][0])
            tc = z["test_ce"].astype(float); r["test_ce"] = (tc[:-1] + tc[1:]) / 2 / math.log(r["p"])
            assert len(r["test_ce"]) == len(r["ce"][0])
        if "H_hero_2x2" in want: report["H_hero_2x2"] = hero2(R["H"], out)
        if "fig2_grid" in want:
            spec2 = [("(a) p = 23", R["P23B"]), ("(b) p = 29", [r for r in R["L1"] if r["p"] == 29]),
                     ("(c) p = 31", R["H"])]
            cols = [column2(rs, t, [r["id"] for r in rs]) for t, rs in spec2]
            draw2(cols, out, "fig2_grid", rows=("ce", "acc", "ah"), height=2.8, ce_zoom=True)
            report["fig2_grid"] = [c["stats"] for c in cols]
        if "locked_cohorts_1hl" in want:
            spec4 = [("(a) p = 23", R["P23B"])] + [(f"({'bcd'[i]}) p = {gp[0]['p']}", gp)
                                                              for i, gp in enumerate(onehl_groups(R["L1"], "recipe"))]
            cols = [column2(rs, t, [r["id"] for r in rs]) for t, rs in spec4]
            draw2(cols, out, "locked_cohorts_1hl", rows=("ce", "acc", "ah"), height=2.6, ce_zoom=True, align_titles=True)
            report["locked_cohorts_1hl"] = [c["stats"] for c in cols]
        if "l1_phase_grid" in want:
            cols = [column2(gp, f"p = {gp[0]['p']}", [r["id"] for r in gp]) for gp in onehl_groups(R["L1"], "recipe")]
            draw2(cols, out, "l1_phase_grid", rows=("ce", "acc", "ah"), height=3.0, ce_zoom=True)
            report["l1_phase_grid"] = [c["stats"] for c in cols]
        if "H_cohort" in want:
            c = column2(R["H"], "p = 31", [r["id"] for r in R["H"]])
            draw2([c], out, "H_cohort", rows=("ce", "acc", "ah"), height=2.2); report["H_cohort"] = [c["stats"]]
        if "L1_cohort_fullCE" in want:
            cols = [column2(gp, f"p = {gp[0]['p']}", [r["id"] for r in gp]) for gp in onehl_groups(R["L1"], "recipe")]
            draw2(cols, out, "L1_cohort_fullCE", rows=("ce", "acc", "ah"), height=3.4, ce_log=True); report["L1_cohort_fullCE"] = [c["stats"] for c in cols]
        g2 = onehl_groups(R["L2"], "p") if "L2" in R else []
        l2 = [(s0, stem) for s0, stem in ((0, "L2_cohort_p11_p13_p17"), (3, "L2_cohort_p19_p23_p29")) if stem in want]
        if l2 or "L2_cohort_all" in want:
            c6 = [column2(gp, f"p = {gp[0]['p']}", [r["id"] for r in gp]) for gp in g2]
            lim6, ah6 = ce_log_lim(c6), max(float(np.nanmax(c["S"]["ahL"])) for c in c6) * 1.08
            for s0, stem in l2:
                cols = c6[s0:s0 + 3]
                draw2(cols, out, stem, rows=("ce", "acc", "ah"), height=3.4, ce_log=True, ce_lim=lim6, ah_top=ah6)
                report[stem] = [c["stats"] for c in cols]
            if "L2_cohort_all" in want:
                draw2(c6, out, "L2_cohort_all", rows=("ce", "acc", "ah"), height=2.8, ce_log=True, ce_lim=lim6, ah_top=ah6)
                report["L2_cohort_all"] = [c["stats"] for c in c6]
        if "L3_cohort" in want:
            cols = [column2(gp, f"p = {gp[0]['p']}", [r["id"] for r in gp]) for gp in onehl_groups(R["L3"], "recipe")]
            draw2(cols, out, "L3_cohort", rows=("ce", "acc", "ah"), height=3.4, ce_log=True); report["L3_cohort"] = [c["stats"] for c in cols]
        if "L4_cohort" in want:
            c = column2(R["L4"], "p = 13", [r["id"] for r in R["L4"]])
            draw2([c], out, "L4_cohort", rows=("ce", "acc", "ah"), height=2.2, ce_log=True, ce_title="Training CE / log p")
            report["L4_cohort"] = [c["stats"]]
    if set(want) & set(DEEP):
        stores = [MF.Runs(str(P.CAMPAIGN["C"]), "C"), MF.Runs(str(P.CAMPAIGN["D"]), "D"), MF.Runs(str(P.CAMPAIGN["F"]), "F"),
                  MF.Runs(str(P.CAMPAIGN["G"]), "G"), MF.Runs(str(P.DEV_RUNS), "dev")]
        V = v2_verdicts(); cache = {}
        def load(rid):
            if rid not in cache:
                hits = [s for s in stores if s.has(rid)]; assert len(hits) == 1, (rid, len(hits))
                d = hits[0].load(rid)
                if rid in V: cache[rid] = deep_run(d, rid, V[rid]); cache[rid]["verdict"] = "v2 settings.jsonl"
                else: cache[rid] = dev_run(d, rid); cache[rid]["verdict"] = "scored here"
            return cache[rid]
        spec = MF.specs()
        for fid in [w for w in want if w in DEEP]:
            cols_spec, rows, height, *kw = spec[fid]; cols = []
            for title, ids in cols_spec:
                if isinstance(ids, tuple): ids = sorted(k for k in stores[0].idx if k.startswith(ids))
                ids = [i for i in ids if any(s.has(i) for s in stores)]
                cols.append(column2([load(i) for i in ids], title, ids))
            draw2(cols, out, fid, rows=rows, height=height, **(kw[0] if kw else {})); report[fid] = [c["stats"] for c in cols]
            print(fid, [(c["title"], c["stats"]["passes"], c["stats"]["n"]) for c in cols], flush=True)
    if "deep_development_overview" in want:
        import scheme_deepdev as SD
        M = SD.manifest(); names = {t: SD.runs_of(pre, M) for t, pre in SD.CONFIGS}
        got = SD.read_all([x for v in names.values() for x in v], P.DEV_RUNS, M, [P.DEV_EXTRA])
        cols = []
        for t, _ in SD.CONFIGS:
            runs = [dev_run(got[x]["d"], x) for x in names[t]]
            cols.append(column2(runs, t, names[t]))
        draw2(cols, out, "deep_development_overview", rows=("ce", "acc", "ah"), height=3.6)
        report["deep_development_overview"] = [c["stats"] for c in cols]
    path = out / "scheme2_stats.json"; old = json.loads(path.read_text()) if path.exists() else {}
    old.update(report); path.write_text(json.dumps(old, indent=1) + "\n")
    for k, v in report.items():
        if isinstance(v, list): print(k, [(c["title"].replace("\n", " "), c["passes"], c["n"], [round(x) for x in c["band_median"]]) for c in v], flush=True)
        else: print(k, {x: v[x] for x in ("n", "band_median", "peak_median", "peak_of_median_curve", "peak_minus_t90_median", "passes")},
                    flush=True)


if __name__ == "__main__":
    main()
