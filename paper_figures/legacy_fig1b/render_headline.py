from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import platform
import sys

HERE = Path(__file__).resolve().parent
WORK = HERE.parent
DRAFT = WORK.parent.parent
OUT = DRAFT / "figures/20260929_headline_v1"
BUNDLE = Path(os.environ.get("ONEHL_BUNDLE", "onehl_bundle"))
os.environ.setdefault("MPLCONFIGDIR", str(WORK / "matplotlib_cache"))
sys.dont_write_bytecode = True

import numpy as np
import hl1_figs_original as F
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

C = F.C
plt.rcParams.update({
    "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 7.5,
    "axes.titlepad": 9, "axes.labelcolor": "black", "text.color": "black",
    "xtick.color": "black", "ytick.color": "black", "axes.edgecolor": "black",
    "axes.titlecolor": "black", "axes.linewidth": .7, "lines.linewidth": 1.5,
    "path.simplify": False, "pdf.fonttype": 42, "savefig.dpi": 240,
})


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def native(v):
    if isinstance(v, dict):
        return {k: native(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [native(x) for x in v]
    if isinstance(v, np.generic):
        return v.item()
    return v


def compare(a, b, path="root"):
    if isinstance(b, dict):
        assert a.keys() == b.keys(), path
        for k in b:
            compare(a[k], b[k], path + "." + k)
    elif isinstance(b, list):
        assert len(a) == len(b), path
        for i, (x, y) in enumerate(zip(a, b)):
            compare(x, y, f"{path}[{i}]")
    elif isinstance(b, (int, float)) and not isinstance(b, bool):
        assert np.isclose(a, b, rtol=1e-12, atol=1e-14), (path, a, b)
    else:
        assert a == b, (path, a, b)


def load_validate():
    folder = WORK / "data/out/confirm_compact"
    runs = sorted(F.load_compact(folder, "H"), key=lambda r: r["id"])
    assert len(runs) == 20
    assert [r["meta"]["confirmation_pair"] for r in runs] == list(range(1, 21))
    assert len({(r["meta"]["init_seed"], r["meta"]["split_seed"]) for r in runs}) == 20
    source = json.loads((BUNDLE / "figures/H_endpoints.json").read_text())
    rows = []
    for r in runs:
        assert r["id"].endswith(f'_confirm{r["meta"]["confirmation_pair"]:02d}')
        assert all(r["config"][k] == v for k, v in source["config"].items())
        assert r["config"]["activation"] == "relu"
        assert r["config"]["polar_cutoff"] == {"mode": "relative", "value": 1e-10}
        assert r["window_rule"] == "fixed" and [r["t0"], r["t1"]] == [4500, 12000]
        assert r["T"] == 40000 and r["meta"]["terminal_status"]["status"] == "complete"
        assert np.array_equal(r["t"], np.arange(40001))
        assert np.array_equal(r["ta"], np.arange(40001))
        for k in ("train_ce", "test_ce", "train_acc", "test_acc"):
            assert r[k].shape == r["t"].shape and np.isfinite(r[k]).all(), k
        assert np.all(r["train_ce"] > 0)
        assert np.array_equal(r["to"], np.arange(0, 40001, 100))
        assert np.all(np.diff(r["to"]) > 0)
        e = F.endpoints(r)
        e["joint"] = bool(r["score"]["joint_PAG10_F02"])
        e["run"] = r["id"]
        e["pair"] = r["meta"]["confirmation_pair"]
        e["dAGOP_corrected"] = e["dAGOP"] - e["dnull"]
        e["first_fit"] = r["score"]["T_fit"]
        e["fit_persists_from_first_fit"] = r["score"]["fit_persists_from_first_fit"]
        e["protocol_flags"] = {k: r["score"][k] for k in
            ("fit_strict", "bad_test_delay", "delayed_acquisition", "full_transition_covered", "representation_first")}
        tt, yy = F.fin(r["to"], r["refit"])
        assert np.array_equal(tt, np.arange(0, 40001, 500))
        hit = np.flatnonzero(yy >= .9 - 1e-12)
        assert len(hit)
        e["refit_first_observed90"] = int(tt[hit[0]])
        e["refit_previous_observation"] = int(tt[hit[0] - 1]) if hit[0] else None
        assert e["refit_first_observed90"] == r["score"]["T_refit"]
        assert e["t90"] == r["score"]["T90"]
        for time in (4500, 12000):
            assert F.at(r["to"], r["agop_valid"], time) == 1
        win = (r["t"] >= 4500) & (r["t"] <= 12000)
        assert int(win.sum()) == 7501
        assert e["fit100"] and np.all(r["train_acc"][win] == 1)
        assert e["protocol_flags"]["fit_strict"]
        assert e["test_t0"] <= .2 and 4500 < e["t90"] <= 12000
        assert e["refit_first_observed90"] < e["t90"]
        assert not e["fit_persists_from_first_fit"]
        assert np.any(r["train_acc"][e["first_fit"]:4500] < 1)
        rows.append(e)
    def med(k): return float(np.median([e[k] for e in rows]))
    def ran(k): return [float(min(e[k] for e in rows)), float(max(e[k] for e in rows))]
    hero_e = F.endpoints(runs[0])
    actual = dict(n=20, window=[4500, 12000], config=source["config"],
        passes={k: sum(bool(e[k]) for e in rows) for k in source["passes"]},
        median={k: med(k) for k in source["median"]},
        range={k: ran(k) for k in source["range"]},
        hero=dict(run=runs[0]["id"], rule="confirmation pair 1",
                  **{k: round(v, 5) if isinstance(v, float) else v for k, v in hero_e.items()}))
    compare(native(actual), source)
    assert all(e["refit_first_observed90"] == 6500 for e in rows)
    return runs, rows, actual


def save(fig, stem):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / (stem + ".pdf"), metadata={"Title": stem, "Author": "",
                "Subject": "Saved 1HL confirmation data; presentation revision only"})
    fig.savefig(OUT / (stem + ".png"), dpi=240)
    plt.close(fig)


def time_axis(ax, end=40000, start=0):
    ax.set_xlim(start, end)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x / 1000:g}"))
    ax.set_xlabel("Training updates (thousands)")


def shade(ax):
    ax.axvspan(4500, 12000, color=C["window"], zorder=0, lw=0)


def make_hero(r, e):
    fig, axs = plt.subplots(2, 2, figsize=(7.3, 5.15))
    fig.subplots_adjust(left=.095, right=.985, bottom=.15, top=.90, wspace=.29, hspace=.57)
    fig.suptitle("1HL ReLU  |  p = 31  |  width 512  |  preselected confirmation pair 1", fontsize=10, y=.98)
    ax = axs[0, 0]
    shade(ax)
    ax.plot(r["t"], r["train_ce"], color=C["train"], label="Training")
    ax.plot(r["tce_t"], r["test_ce"], color=C["test"], label="Held-out")
    ax.axhline(np.log(31), color=C["null"], ls="--", lw=.8, label=r"Uniform prediction: $\log p$")
    ax.set_ylim(0, 7.2)
    ax.set_xticks([0, 10000, 20000, 30000, 40000])
    ax.set_ylabel("Cross-entropy")
    ax.set_title("(a) Full training run", loc="left")
    time_axis(ax)
    ax.legend(loc="upper right", fontsize=7)

    ax = axs[0, 1]
    sel = (r["t"] >= 4500) & (r["t"] <= 12000)
    loss = r["train_ce"][sel]
    ax.plot(r["t"][sel], 100 * (loss / loss.min() - 1), color=C["train"])
    ax.axhline(10, color=C["null"], ls="--", lw=.8)
    ax.text(.98, .78, "10% criterion", ha="right", transform=ax.transAxes, fontsize=7)
    ax.text(.03, .94, f"Range = {100 * e['v_min']:.2f}%\nMinimum CE = {loss.min():.4f}",
            va="top", transform=ax.transAxes, fontsize=8)
    ax.set_ylim(0, 13.4)
    ax.set_xticks([4500, 6000, 8000, 10000, 12000])
    ax.set_ylabel("CE above window minimum (%)")
    ax.set_title("(b) Fixed-window loss zoom", loc="left")
    time_axis(ax, 12000, 4500)

    ax = axs[1, 0]
    shade(ax)
    ax.plot(r["ta"], r["train_acc"], color=C["train"], ls="--", lw=1.2, label="Training network")
    ax.plot(r["ta"], r["test_acc"], color=C["test"], lw=1.6, label="Held-out network")
    tt, yy = F.fin(r["to"], r["refit"])
    ax.plot(tt, yy, color=C["refit"], ls=":", lw=1.8, marker=".", ms=2.5,
            label="Held-out, train-only refit")
    ax.axhline(.9, color=C["null"], lw=.6, ls=(0, (4, 4)))
    ax.axvline(e["refit_first_observed90"], color=C["refit"], lw=.7, ls=":")
    ax.axvline(e["t90"], color=C["test"], lw=.7, ls=":")
    ax.set_ylim(-.025, 1.075)
    ax.set_ylabel("Accuracy")
    ax.set_xticks([0, 4000, 8000, 12000, 16000])
    ax.set_title("(c) Prediction recovery: transition zoom", loc="left")
    ax.legend(loc="lower right", fontsize=6.9)
    time_axis(ax, 16000)

    ax = axs[1, 1]
    shade(ax)
    tt, yy = F.fin(r["to"], r["AH"])
    ax.plot(tt, yy, color=C["AH"], lw=1.8)
    a0, a1 = F.at(tt, yy, 4500), F.at(tt, yy, 12000)
    ax.scatter([4500, 12000], [a0, a1], color=C["AH"], s=14, zorder=4)
    ax.annotate("", xy=(12500, a1), xytext=(12500, a0),
                arrowprops={"arrowstyle": "<->", "color": "black", "lw": .9})
    ax.text(.47, .35, rf"$\Delta A_H = {e['dAH']:.4f}$", transform=ax.transAxes, fontsize=9)
    ax.set_ylim(0, .13)
    ax.set_ylabel(r"Hidden Fourier fraction $A_H$")
    ax.set_title("(d) Hidden representation", loc="left")
    ax.set_xticks([0, 4000, 8000, 12000, 16000])
    time_axis(ax, 16000)
    fig.text(.095, .066, "Shading: fixed window 4.5k-12k. No smoothing or adaptive loss control; head decay = 0.2.", fontsize=7.3)
    fig.text(.095, .037, "First observed 90%: refit 6.5k (500-update grid); network 10.775k (every update).", fontsize=7.3)
    fig.text(.095, .009, "Window training accuracy = 100% at every update.", fontsize=7.0)
    save(fig, "H_hero_paper")


def make_summary(rows):
    fig, axs = plt.subplots(2, 2, figsize=(7.3, 5.45))
    fig.subplots_adjust(left=.095, right=.985, bottom=.16, top=.875, wspace=.35, hspace=.64)
    p = axs[0, 0].get_position()
    axs[0, 0].set_position([p.x0 + .20, p.y0, p.width - .20, p.height])
    fig.suptitle("Locked confirmation: all 20 initialization / split pairs", fontsize=11, y=.98)
    fig.text(.5, .935, "p = 31  |  width 512  |  random 80% training split  |  fixed window 4.5k-12k", ha="center", fontsize=8)
    ax = axs[0, 0]
    keys = [("P10", r"CE range $\leq 10\%$"), ("CE01", r"Minimum CE $\geq 0.1$"),
            ("F06", r"$\Delta A_H \geq 0.06$"), ("fit100", "Window train accuracy = 100%"),
            ("rec90", "Held-out reaches 90%"), ("AGOPnull02", r"AGOP gain - null $\geq 0.02$")]
    for i, (key, _) in enumerate(keys):
        count = sum(bool(e[key]) for e in rows)
        lo, hi = F.wilson(count, 20)
        ax.plot([lo, hi], [i, i], color="black", lw=1)
        ax.plot(count / 20, i, "o", color="black", ms=3.5)
        ax.text(1.17, i, f"{count}/20", va="center", ha="left", fontsize=7.5)
    ax.set_yticks(range(len(keys)))
    ax.set_yticklabels([lab for _, lab in keys], fontsize=7.0)
    ax.set_ylim(len(keys) - .5, -.65)
    ax.set_xlim(-.055, 1.60)
    ax.set_xticks([0, .5, 1])
    ax.set_xlabel("Fraction passing (95% Wilson CI)", fontsize=7.7)
    ax.set_title("(a) Declared endpoints", loc="left")
    ax.grid(axis="y", visible=False)

    pairs = np.arange(1, 21)
    specs = [(axs[0, 1], "v_min", 100, C["train"], 10, (0, 11.5),
              "(b) Training-CE variation", "Relative range (%)"),
             (axs[1, 0], "dAH", 1, C["AH"], .06, (0, .115),
              "(c) Hidden Fourier gain", r"$\Delta A_H$"),
             (axs[1, 1], "dAGOP_corrected", 1, C["agop"], .02, (-.025, .215),
              "(d) AGOP gain minus null", r"$\Delta\mathrm{AGOP}-\Delta\mathrm{null}$")]
    for ax, key, scale, col, threshold, limits, title, ylabel in specs:
        values = np.array([e[key] for e in rows]) * scale
        ax.scatter(pairs, values, color=col, s=17, zorder=3, edgecolors="none")
        ax.scatter(pairs[:1], values[:1], s=53, facecolors="none", edgecolors="black", linewidths=.85, zorder=4)
        ax.axhline(threshold, color=C["null"], ls="--", lw=.8)
        ax.set_ylim(*limits)
        ax.set_xlim(.3, 20.7)
        ax.set_xticks([1, 5, 10, 15, 20])
        ax.set_xlabel("Confirmation pair")
        ax.set_ylabel(ylabel)
        ax.set_title(title, loc="left")
        if key == "dAGOP_corrected":
            ax.axhline(0, color=C["null"], lw=.5)
        ax.grid(axis="x", visible=False)
    fig.text(.095, .050, "Dashed lines: declared thresholds. Outlined point: preselected hero (pair 1). All runs included.", fontsize=7.1)
    fig.text(.095, .020, "AGOP null = median of 8 random frames. Corrected gain passes in 16/20; the hero fails.", fontsize=7.1)
    save(fig, "H_summary_paper")


def main():
    runs, rows, checked = load_validate()
    make_hero(runs[0], rows[0])
    make_summary(rows)
    files = [p for p in (WORK / "data").rglob("*") if p.is_file()]
    sources = {str(p.relative_to(WORK)): sha(p) for p in sorted(files)}
    sources.update({str(p): sha(p) for p in [
        BUNDLE / "figures/H_endpoints.json", BUNDLE / "code/hl1_figs.py",
        BUNDLE / "code/render_hero.py", BUNDLE / "lightning/hero_search/hero_confirm_compact.tar.gz"]})
    assert sha(HERE / "hl1_figs_original.py") == sha(BUNDLE / "code/hl1_figs.py")
    validation = dict(status="PASS", source_summary_reproduced=True, n=20,
        checks=["all expected pairs, common recipe and fixed window", "all 40001 dense CE/accuracy steps",
                "all 7501 inclusive window CE steps", "valid AGOP ranks at both endpoints",
                "all original H_endpoints.json fields reproduced", "refit cadence and first-observed crossings",
                "window fit preserved and earlier fit loss checked", "unaltered original loader and endpoints"],
        hero_rule="confirmation pair 1, unchanged", data_changed=False, training_run=False,
        display_changes=["larger fonts and explicit panel labels", "transition panels zoom to 0-16k; full CE stays 0-40k",
                         "direct per-run AGOP gain minus null gain", "horizontal Wilson endpoint panel; window-fit row added",
                         "2026-09-29: strict-joint row and footnote removed from the summary; hero footnote no longer points to it; 0/20 kept in per_run rows and README"],
        plotting=dict(python=platform.python_version(), numpy=np.__version__, matplotlib=matplotlib.__version__),
        original_summary=checked, per_run=rows,
        sources=sources, renderer_sha256=sha(__file__),
        outputs={p.name: sha(p) for p in sorted(OUT.glob("H_*_paper.*"))})
    (WORK / "VALIDATION.json").write_text(json.dumps(native(validation), indent=2) + "\n")
    print(json.dumps({"status": "PASS", "figures": list(validation["outputs"]),
                      "passes": checked["passes"], "hero_refit90": rows[0]["refit_first_observed90"],
                      "hero_network90": rows[0]["t90"]}, indent=2))


if __name__ == "__main__":
    main()
