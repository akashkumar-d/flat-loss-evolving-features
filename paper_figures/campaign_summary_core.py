from pathlib import Path
import argparse
import ast
import hashlib
import json
import os
import platform

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig_campaign_summary_core")
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.text import Text

HERE = Path(__file__).resolve().parent
GROUPS = ["D3", "D4", "D5", "D4p23", "D4w512", "D3ngd"]
LABELS = ["D3", "D4", "D5", "D4\np23", "D4\nw512", "D3*"]
SPECS = [
    ("share", "a) Recovery share", "Share of 0.1 to 0.9 rise", .75, (-.02, 1.14), "#444444"),
    ("dAH_last", "b) Last-hidden Fourier gain", r"$\Delta A_H$", .02, (0, .105), "#009E73"),
    ("final_test", "c) Final held-out accuracy", "Accuracy", .99, (.90, 1.012), "#D55E00"),
]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def cohort_data(analysis):
    rows = list(analysis["rows"].values())
    output = {}
    for group in GROUPS:
        runs = sorted([r for r in rows if r["cohort"] == group], key=lambda r: (r["seed"], r["stage"]))
        offsets = np.linspace(-.20, .20, len(runs))
        offsets = offsets[np.argsort(np.argsort([r["id"] for r in runs]))]
        values = {key: np.array([r["main"][key] for r in runs]) for key, *_ in SPECS}
        output[group] = dict(
            ids=[r["id"] for r in runs], n=len(runs), k=sum(r["main"]["primary"] for r in runs),
            offsets=offsets.tolist(),
            primary_failures={key: sum(not r["main"]["primary_checks"][key] for r in runs)
                              for key in ("share", "F02", "final", "fit")},
            values={key: val.tolist() for key, val in values.items()},
            statistics={key: dict(min=float(val.min()), median=float(np.median(val)), max=float(val.max()))
                        for key, val in values.items()},
        )
    assert sum(c["n"] for c in output.values()) == len(rows) == 164
    assert len({rid for c in output.values() for rid in c["ids"]}) == 164
    return output


def reference_check(analysis, cohorts, source):
    tree = ast.parse(source.read_text())
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "campaign_summary")
    captured = {}
    receipt = {"derived": {}}
    scope = dict(np=np, plt=plt, RECEIPT=receipt, BLUE="#0072B2", GREEN="#009E73", PURPLE="#7851A9",
                 ORANGE="#D55E00", GRAY="#888888", LAYER="#444444", AGOP_C="#E69F00",
                 wilson=lambda k, n: (0, 0), save=lambda fig, name: captured.update(fig=fig))
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(source), "exec"), scope)
    scope["campaign_summary"](analysis)
    fig = captured["fig"]
    old = receipt["derived"]["campaign_summary"]
    for gi, group in enumerate(GROUPS):
        cohort = cohorts[group]
        assert old[group]["n"] == cohort["n"] and old[group]["k"] == cohort["k"]
        assert old[group]["primary_failures"] == cohort["primary_failures"]
        for ai, (key, *_rest) in enumerate(SPECS):
            expected = np.column_stack((gi + np.asarray(cohort["offsets"]), cohort["values"][key]))
            assert np.array_equal(fig.axes[ai].collections[gi].get_offsets(), expected)
            assert old[group][key] == cohort["statistics"][key]
            median_line = fig.axes[ai].lines[gi]
            assert np.array_equal(median_line.get_xdata(), [gi - .25, gi + .25])
            assert np.array_equal(median_line.get_ydata(), [cohort["statistics"][key]["median"]] * 2)
        assert fig.axes[0].texts[gi].get_text() == f'{cohort["k"]}/{cohort["n"]}'
    plt.close(fig)
    return dict(exact_scatter_coordinates=True, exact_median_bars=True, exact_pass_counts=True,
                exact_min_median_max=True, exact_primary_failure_counts=True, runs=164,
                method="Executed preserved campaign_summary function; intercepted save; compared first three axes.",
                wilson_note="Wilson intervals are not plotted or part of this comparison.")


def validate(fig, counts, cohorts):
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    all_ticks, drawn_ticks = set(), set()
    for ax in fig.axes:
        for axis in (ax.xaxis, ax.yaxis):
            for tick in axis.get_major_ticks() + axis.get_minor_ticks():
                all_ticks.update((tick.label1, tick.label2))
            for tick in axis._update_ticks():
                drawn_ticks.update((tick.label1, tick.label2))
    texts = [t for t in fig.findobj(Text) if t.get_visible() and t.get_text().strip()
             and (t not in all_ticks or t in drawn_ticks)]
    outside = [t.get_text() for t in texts if not fig.bbox.contains(*t.get_window_extent(renderer).p0)
               or not fig.bbox.contains(*t.get_window_extent(renderer).p1)]
    assert not outside, outside
    boxes = [t.get_window_extent(renderer) for t in counts]
    gaps = [b.x0 - a.x1 for a, b in zip(boxes, boxes[1:])]
    assert min(gaps) > 1, gaps
    assert max(b.y1 for b in boxes) < fig.axes[0].title.get_window_extent(renderer).y0
    for ax, (key, *_rest) in zip(fig.axes, SPECS):
        lo, hi = ax.get_ylim()
        assert all(lo <= min(c["values"][key]) <= max(c["values"][key]) <= hi for c in cohorts.values())
        labels = [t.get_window_extent(renderer) for t in ax.get_xticklabels()]
        assert all(b.x0 - a.x1 > 1 for a, b in zip(labels, labels[1:])), "overlapping cohort labels"
    return dict(all_text_inside_canvas=True, all_values_inside_axes=True,
                minimum_font_points=min(t.get_fontsize() for t in texts),
                count_label_minimum_gap_pixels=float(min(gaps)), count_labels_clear_of_title=True,
                width_inches=fig.get_figwidth(), height_inches=fig.get_figheight(), centered_titles=True)


def draw(cohorts):
    fig, axes = plt.subplots(1, 3, figsize=(6.75, 2.5), gridspec_kw={"width_ratios": [1.22, 1, 1]})
    counts = []
    for gi, group in enumerate(GROUPS):
        cohort = cohorts[group]
        for ai, (key, title, ylabel, threshold, ylim, color) in enumerate(SPECS):
            ax = axes[ai]
            vals = cohort["values"][key]
            ax.scatter(gi + np.array(cohort["offsets"]), vals, s=11, color=color, alpha=.65, linewidths=0, zorder=3)
            median = cohort["statistics"][key]["median"]
            ax.plot([gi - .25, gi + .25], [median, median], color="black", lw=1.5, zorder=4)
            if ai == 0:
                counts.append(ax.text(gi, 1.055, f'{cohort["k"]}/{cohort["n"]}', ha="center", va="bottom", fontsize=8))
    for ax, (key, title, ylabel, threshold, ylim, color) in zip(axes, SPECS):
        ax.set_ylim(*ylim)
        ax.set_xlim(-.5, 5.5)
        ax.set_xticks(range(6), LABELS, fontsize=8)
        ax.set_ylabel(ylabel, labelpad=3)
        ax.axhline(threshold, color="#555555", ls="--", lw=.9)
        ax.axvline(2.5, color="#AAAAAA", lw=.7)
        ax.grid(True, axis="y")
        ax.set_title(title, loc="center", fontsize=8.5, pad=8)
    axes[0].set_yticks([0, .25, .5, .75, 1], ["0", ".25", ".5", ".75", "1"])
    axes[1].set_yticks([0, .025, .05, .075, .1], ["0", ".025", ".05", ".075", ".10"])
    axes[2].set_yticks([.90, .95, 1], [".90", ".95", "1"])
    fig.subplots_adjust(left=.068, right=.994, top=.84, bottom=.30, wspace=.25)
    fig.text(.53, .080, "Counts: joint passes. p23: p = 23; w512: width 512; D3*: normalized GD.",
             ha="center", va="bottom", fontsize=8)
    fig.text(.53, .020, "Other settings: Polar. Black bars: medians. Dashed lines: primary thresholds.",
             ha="center", va="bottom", fontsize=8)
    return fig, validate(fig, counts, cohorts)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis", type=Path, default=HERE.parent / "deep/campaign_C/results/analysis.json")
    parser.add_argument("--reference-renderer", type=Path, default=HERE / "deep_campaign_summary/build_figures.py")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    plt.rcParams.update(matplotlib.rcParamsDefault)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8.5, "axes.labelsize": 8.5,
                        "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": "#AAAAAA",
                        "axes.linewidth": .6, "grid.color": "#DDDDDD", "grid.linewidth": .45,
                        "xtick.labelsize": 8, "ytick.labelsize": 8, "pdf.fonttype": 42, "ps.fonttype": 42,
                        "savefig.facecolor": "white"})
    data_hash = sha(args.analysis)
    analysis = json.loads(args.analysis.read_text())
    cohorts = cohort_data(analysis)
    check = reference_check(analysis, cohorts, args.reference_renderer)
    fig, validation = draw(cohorts)
    args.out.mkdir(parents=True, exist_ok=True)
    outputs = {}
    for extension in ("pdf", "png"):
        target = args.out / f"deep_campaign_summary.{extension}"
        options = dict(metadata={"Title": "Campaign summary: primary metrics", "Creator": "campaign_summary_core.py",
                                 "CreationDate": None}) if extension == "pdf" else {}
        fig.savefig(target, dpi=240, **options)
        outputs[target.name] = sha(target)
    plt.close(fig)
    assert sha(args.analysis) == data_hash
    receipt = dict(source_sha256=data_hash, renderer_sha256=sha(__file__),
                   reference_renderer_sha256=sha(args.reference_renderer),
                   runtime=dict(python=platform.python_version(), numpy=np.__version__, matplotlib=matplotlib.__version__),
                   comparison=check, validation=validation, cohorts=cohorts, output_sha256=outputs)
    (args.out / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(dict(runs=164, joint_counts={g: f'{c["k"]}/{c["n"]}' for g, c in cohorts.items()},
                          comparison="PASS", validation=validation, output_sha256=outputs), indent=2))


if __name__ == "__main__":
    main()
