from __future__ import annotations

import datetime
import hashlib
import json
import os
from pathlib import Path
import platform
import sys

HERE = Path(__file__).resolve().parent
DRAFT = HERE.parents[3]
ORIGINAL = DRAFT / "experiments/headline_figures_20260929"
os.environ["MPLCONFIGDIR"] = str(HERE / "matplotlib_cache")
sys.dont_write_bytecode = True
sys.path.insert(0, str(ORIGINAL / "code"))
import render_headline as H
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

F = H.F
COLORS = dict(train="#0072B2", test="#D55E00", AH="#009E73", null="#888888")
WIDTH, HEIGHT = 3.30, 2.30
GENERATION_COMMAND = "PYTHONDONTWRITEBYTECODE=1 python render_summary.py"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def style():
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8,
        "axes.titlesize": 8.5, "axes.labelsize": 8, "xtick.labelsize": 8,
        "ytick.labelsize": 8, "legend.fontsize": 8, "axes.titleweight": "normal",
        "axes.titlepad": 4, "axes.titlelocation": "left", "axes.labelcolor": "black",
        "text.color": "black", "axes.titlecolor": "black", "xtick.color": "black",
        "ytick.color": "black", "axes.edgecolor": "#888888", "axes.linewidth": .6,
        "axes.spines.top": False, "axes.spines.right": False,
        "grid.color": "#E6E6E6", "grid.linewidth": .45, "pdf.fonttype": 42,
        "ps.fonttype": 42, "path.simplify": False, "savefig.dpi": 300})


def draw(rows):
    style()
    fig = plt.figure(figsize=(WIDTH, HEIGHT))
    bottom, gap, top, drop = .49, .405, .205, .09
    height = (HEIGHT - bottom - gap - top) / 2
    upper = bottom + height + gap
    forest = fig.add_axes([.414, (upper - drop) / HEIGHT, .100, (height + drop) / HEIGHT])
    keys = [("P10", r"CE range $\leq$10%"), ("CE01", r"Min CE $\geq$.1"),
            ("F06", r"$\Delta A_H\geq$.06"), ("fit100", "Train = 100%"),
            ("rec90", "Held-out 90%"), ("fit_persists_from_first_fit", "First-fit persists")]
    rates = []
    for i, (key, label) in enumerate(keys):
        count = sum(bool(row[key]) for row in rows)
        lo, hi = F.wilson(count, 20)
        forest.plot([lo, hi], [i, i], color="black", lw=.7)
        forest.plot(count / 20, i, "o", color="black", ms=2.7)
        forest.text(1.23, i, str(count), va="center", ha="left")
        rates.append(dict(key=key, label=label, count=count, denominator=20,
                          wilson95=[lo, hi]))
    forest.set_yticks(range(6), [label for _, label in keys])
    forest.set_ylim(5.5, -.5)
    forest.set_xlim(-.055, 1.10)
    forest.set_xticks([0, 1], ["0", "1"])
    forest.grid(False)
    forest.tick_params(axis="y", length=0, pad=3)
    forest.tick_params(axis="x", pad=1, length=2)
    fig.text(.25, 1 - (2.50 - .947 * 2.50) / HEIGHT,
             "Passes / 20 (95% CI)", fontsize=8.3, ha="center", va="bottom")

    pairs = np.arange(1, 21)
    specs = [(.645, upper, .315, "v_min", 100, COLORS["train"], 10,
              (0, 11.5), [0, 5, 10], "CE range (%)"),
             (.12, bottom, .335, "dAH", 1, COLORS["AH"], .06,
              (0, .115), [0, .05, .1], r"Fourier gain $\Delta A_H$"),
             (.645, bottom, .315, "test_t1", 1, COLORS["test"], .9,
              (0, 1.05), [0, .5, 1], "Held-out at 12k")]
    plotted = {}
    for left, lower, width, key, scale, color, threshold, limits, ticks, title in specs:
        ax = fig.add_axes([left, lower / HEIGHT, width, height / HEIGHT])
        values = np.asarray([row[key] for row in rows]) * scale
        assert np.all((values >= limits[0]) & (values <= limits[1]))
        ax.scatter(pairs, values, color=color, s=9, zorder=3, edgecolors="none")
        ax.scatter(pairs[:1], values[:1], s=31, facecolors="none", edgecolors="black",
                   linewidths=.7, zorder=4)
        ax.axhline(threshold, color=COLORS["null"], ls="--", lw=.6)
        ax.set_ylim(*limits)
        ax.set_yticks(ticks)
        ax.set_xlim(.1, 20.9)
        ax.set_xticks([1, 10, 20])
        ax.set_title(title, pad=3)
        ax.tick_params(axis="both", pad=2, length=2.5)
        ax.grid(True, axis="y", color="#E6E6E6", lw=.45)
        ax.grid(False, axis="x")
        plotted[key] = values.tolist()
    fig.text(.53, .080 * 2.50 / HEIGHT, "Initialization / split pair", ha="center", va="center")
    fig.text(.53, .026 * 2.50 / HEIGHT, "Dashed: criterion; outlined: pair 1", ha="center", va="center")
    return fig, rates, plotted


def main():
    inputs = sorted((ORIGINAL / "data").rglob("*.json")) + sorted((ORIGINAL / "data").rglob("*.npz"))
    inputs += [ORIGINAL / "code/render_headline.py", ORIGINAL / "code/hl1_figs_original.py",
               DRAFT / "experiments/compact_headline_20260929/render_compact.py",
               DRAFT / "experiments/submission_figures_20260930/render_onehl.py",
               H.BUNDLE / "figures/H_endpoints.json",
               DRAFT / "figures/20260930_submission/H_summary_2x2.pdf",
               DRAFT / "figures/20260930_submission/H_summary_2x2.png"]
    before = {str(path): sha(path) for path in inputs}
    runs, rows, original_summary = H.load_validate()
    per_pair = []
    for run, row in zip(runs, rows):
        score = run["score"]
        assert [run["t0"], run["t1"]] == [4500, 12000]
        assert row["test_t1"] == score["test_acc_start_end"][1]
        assert row["test_t1"] == float(run["test_acc"][12000])
        assert row["v_min"] == score["v_min"]
        assert row["dAH"] == score["delta_AH"]
        lost_steps = run["ta"][(run["ta"] > row["first_fit"]) &
                               (run["ta"] < 4500) & (run["train_acc"] < 1)]
        assert len(lost_steps) > 0 and row["fit_persists_from_first_fit"] is False
        per_pair.append(dict(pair=row["pair"], run=row["run"],
            window=[4500, 12000], v_min=row["v_min"], delta_AH=row["dAH"],
            held_out_accuracy_at_12000=row["test_t1"],
            first_fit=row["first_fit"], first_lost_fit_after_first_fit=int(lost_steps[0]),
            fit_persists_from_first_fit=False, original_strict_joint=bool(row["joint"])))
    fig, rates, plotted = draw(rows)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    texts = [text for text in fig.findobj(plt.Text) if text.get_visible() and text.get_text()]
    outside = []
    for text in texts:
        box = text.get_window_extent(renderer)
        if box.x0 < -.6 or box.y0 < -.6 or box.x1 > fig.bbox.width + .6 or box.y1 > fig.bbox.height + .6:
            outside.append(text.get_text())
    assert not outside, outside
    assert min(text.get_fontsize() for text in texts) >= 8
    assert all("AGOP" not in text.get_text().upper() for text in texts)
    count_boxes = [text.get_window_extent(renderer) for text in fig.axes[0].texts]
    tick_boxes = [text.get_window_extent(renderer) for text in fig.axes[1].get_yticklabels()]
    separation = (min(box.x0 for box in tick_boxes) - max(box.x1 for box in count_boxes)) * 72 / fig.dpi
    assert separation >= 3, separation
    stem = HERE / "H_summary_2x2"
    fig.savefig(stem.with_suffix(".pdf"), metadata={"Title": "H confirmation summary",
        "Author": "", "Subject": "All 20 saved confirmation pairs; fixed-window endpoint accuracy",
        "CreationDate": datetime.datetime(2026, 10, 5, tzinfo=datetime.timezone.utc)})
    fig.savefig(stem.with_suffix(".png"), dpi=300)
    plt.close(fig)
    assert before == {str(path): sha(path) for path in inputs}
    receipt = dict(status="PASS", generation_command=GENERATION_COMMAND,
        command_working_directory=str(DRAFT.parent), renderer_sha256=sha(__file__),
        original_loader_complete_validation_repeated=True, original_summary=original_summary,
        training=False, new_observers=False, smoothing=False, excluded_runs=0,
        source_files_unchanged=True, source_sha256=before,
        scope="New candidate-local summary only; all original assets and saved outcomes preserved",
        size_inches=[WIDTH, HEIGHT], minimum_text_pt=8, all_text_inside_canvas=True,
        count_to_neighbor_tick_gap_pt=separation, rates=rates, per_pair=per_pair,
        scatter_values=plotted, endpoint_accuracy_field="score.test_acc_start_end[1]",
        endpoint_accuracy_raw_crosscheck="NPZ test_accuracy at acc_step=12000",
        endpoint_accuracy_summary=dict(min=min(plotted["test_t1"]),
            median=float(np.median(plotted["test_t1"])), max=max(plotted["test_t1"])),
        original_strict_joint_count=sum(row["joint"] for row in rows),
        caption_note="Bottom-right: directly observed held-out accuracy at the fixed window end, t=12000. "
                     "The first-fit-persistence rate is 0/20 because every run loses perfect training "
                     "accuracy after its first fit and before the fixed window. Original strict joint: 0/20.",
        software=dict(python=platform.python_version(), numpy=np.__version__, matplotlib=matplotlib.__version__),
        outputs={path.name: sha(path) for path in (stem.with_suffix(".pdf"), stem.with_suffix(".png"))})
    (HERE / "FIGURE_RECEIPT.json").write_text(json.dumps(H.native(receipt), indent=2) + "\n")
    print(json.dumps({"status": "PASS", "rates": rates,
        "endpoint_accuracy_summary": receipt["endpoint_accuracy_summary"], "outputs": receipt["outputs"]}, indent=2))


if __name__ == "__main__":
    main()
