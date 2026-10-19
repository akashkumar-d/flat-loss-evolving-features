from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np


HERE = Path(__file__).resolve().parent
BLUE = "#1976b7"
GREEN = "#008c78"
ORANGE = "#d48300"
PURPLE = "#7851a9"
GRAY = "#687582"
RED = "#b64d57"
P_COLORS = {13: BLUE, 17: PURPLE, 23: GREEN}

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 10,
    "axes.titlesize": 12, "axes.titleweight": "bold",
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": .16, "legend.frameon": False,
    "savefig.dpi": 165, "svg.fonttype": "none",
})


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def exact_grid_summary(series):
    if not series:
        return {k: np.array([]) for k in ("step", "median", "q25", "q75", "n")}
    steps = np.unique(np.concatenate([np.asarray(t) for t, _ in series]))
    values = np.full((len(series), len(steps)), np.nan)
    for i, (ts, ys) in enumerate(series):
        ts = np.asarray(ts)
        ys = np.asarray(ys, dtype=float)
        if len(np.unique(ts)) != len(ts):
            raise ValueError("Duplicate observation step")
        if not np.all(np.isfinite(ys)):
            raise ValueError("Non-finite value in a plotted series")
        values[i, np.searchsorted(steps, ts)] = ys
    complete_grid = np.all(np.isfinite(values))
    median = np.median if complete_grid else np.nanmedian
    quantile = np.quantile if complete_grid else np.nanquantile
    return {
        "step": steps,
        "median": median(values, axis=0),
        "q25": quantile(values, .25, axis=0),
        "q75": quantile(values, .75, axis=0),
        "n": np.sum(np.isfinite(values), axis=0),
    }


def collect_series(runs, table, field, denominator=None):
    answer = []
    for run in runs:
        z = run.get(table)
        if z is None or field not in z or not len(z["step"]):
            continue
        y = np.asarray(z[field], dtype=float)
        if denominator is not None:
            y = y / run[denominator]
        answer.append((np.asarray(z["step"]), y))
    return answer


def hidden_energy_axis_limits(runs):
    observed = collect_series(runs, "signed", "fourier_energy")
    maximum = 0.
    for _, values in observed:
        if not np.all(np.isfinite(values)):
            raise ValueError("Non-finite hidden energy cannot determine an axis range")
        maximum = max(maximum, float(np.max(values)))
    return 0., max(.15, 1.08 * maximum)


def plot_series(ax, runs, table, field, color, label, ls="-", denominator=None):
    series = collect_series(runs, table, field, denominator)
    for t, y in series:
        ax.plot(t / 1000, y, color=color, alpha=.24 if len(runs) > 1 else 0,
                linewidth=.9, linestyle=ls)
    z = exact_grid_summary(series)
    if len(z["step"]):
        t = z["step"] / 1000
        if len(runs) > 1:
            ax.fill_between(t, z["q25"], z["q75"], color=color, alpha=.12, linewidth=0)
        ax.plot(t, z["median"], color=color, linewidth=2, linestyle=ls, label=label)
    return z


def shared_time_axis(ax, zero=False):
    ax.set(xlim=(0, 40), xticks=[0, 10, 20, 30, 40],
           xlabel="optimization step (thousands)")
    ax.axvspan(15, 20, color=BLUE, alpha=.045, lw=0)
    if zero:
        ax.axhline(0, color=RED, lw=.8)


def fixed_window_text(run):
    z = run.get("trajectory")
    if z is None:
        return "No trajectory observed."
    take = (z["step"] >= 15000) & (z["step"] <= 20000)
    if not np.any(take) or not {15000, 20000}.issubset(set(z["step"].tolist())):
        return "Fixed 15–20k window: not fully observed."
    ce = z["train_ce"][take]
    count = z["test_strict_count"][take]
    if ce.mean() == 0:
        variation = "15–20k relative CE variation undefined\n(zero mean; no plateau claim)."
    else:
        relative = 100 * (ce.max() - ce.min()) / ce.mean()
        variation = f"15–20k CE range / mean: {relative:.3g}%"
    return (f"{variation}\n"
            f"Test correct: {int(count[0])} → {int(count[-1])} / {run['n_test']}")


def cross_panel(ax, runs):
    raw = []
    for run in runs:
        if run.get("cross") is not None:
            counts = np.asarray(run["cross"]["test_correct"])
            if counts.shape != (2, 2):
                raise ValueError("Cross-output matrix must be 2 by 2")
            raw.append((run, counts))
    ax.grid(False)
    ax.set(title="F. Crossed output components (observer only)",
           xticks=[0, 1], xticklabels=["10k", "20k"],
           yticks=[0, 1], yticklabels=["10k", "20k"],
           xlabel="complement Z checkpoint", ylabel="addition A checkpoint")
    if not raw:
        ax.text(.5, .5, "No complete 10k / 20k pair", ha="center", va="center",
                transform=ax.transAxes)
        return []
    proportions = np.stack([n / r["n_test"] for r, n in raw])
    med = np.median(proportions, axis=0)
    ax.imshow(med, origin="upper", vmin=0, vmax=1, cmap="YlGnBu", aspect="auto")
    for i in range(2):
        for j in range(2):
            details = ", ".join(f"{int(n[i, j])}/{r['n_test']}" for r, n in raw)
            prefix = "" if len(runs) == 1 else "median "
            ax.text(j, i, f"{prefix}{100 * med[i, j]:.1f}%\n{details}",
                    ha="center", va="center", fontsize=9.5,
                    color="white" if med[i, j] > .58 else "#193442")
    return [dict(run_id=r["run_id"], n_test=r["n_test"],
                 test_correct=n.tolist()) for r, n in raw]


def six_panel(runs, title, subtitle):
    fig, axes = plt.subplots(2, 3, figsize=(16.2, 10.2))
    a = axes.ravel()
    fig.suptitle(title, fontsize=15, y=.975)
    fig.text(.5, .934, subtitle, ha="center", fontsize=10, color=GRAY)
    plot_series(a[0], runs, "trajectory", "train_ce", BLUE, "actual training CE")
    plot_series(a[0], runs, "trajectory", "test_ce", GREEN, "actual held-out CE")
    a[0].set(yscale="log", ylabel="raw cross-entropy", title="A. Losses (no imposed CE floor)")
    a[0].legend(fontsize=8)
    if len(runs) == 1:
        a[0].text(.035, .075, fixed_window_text(runs[0]), transform=a[0].transAxes,
                  fontsize=8.5, bbox=dict(facecolor="white", edgecolor="none", alpha=.9))
    else:
        a[0].text(.035, .06, "Fixed 15–20k reference window;\nnot selected for success in these runs.",
                  transform=a[0].transAxes, fontsize=8.5,
                  bbox=dict(facecolor="white", edgecolor="none", alpha=.9))
    plot_series(a[1], runs, "trajectory", "train_strict_count", BLUE, "actual training", "--", "n_train")
    plot_series(a[1], runs, "trajectory", "test_strict_count", GREEN, "actual held-out", "-", "n_test")
    plot_series(a[1], runs, "signed", "separated_certified", PURPLE,
                "evaluated signed accuracy lower bound", ":", "n_test")
    a[1].set(ylim=(-.02, 1.04), ylabel="strict accuracy / lower bound",
             title="B. Actual predictions and signed bound")
    a[1].yaxis.set_major_formatter(PercentFormatter(1))
    a[1].legend(fontsize=8, loc="lower right")
    plot_series(a[2], runs, "signed", "rule_min", PURPLE, "minimum signed rule margin")
    plot_series(a[2], runs, "signed", "harmful_max", ORANGE, "largest harmful complement")
    a[2].set(ylabel="actual-logit contrast", title="C. Signed addition signal vs interference")
    a[2].legend(fontsize=8)
    plot_series(a[3], runs, "signed", "test_min_margin", GREEN, "actual minimum test margin")
    plot_series(a[3], runs, "signed", "row_lower_min", PURPLE, "minimum separated row bound", "--")
    plot_series(a[3], runs, "signed", "global_lower", GRAY, "coarser global bound", ":")
    a[3].set(ylabel="correct − rival margin / lower bound",
             title="D. Conservative signed-margin bounds")
    a[3].legend(fontsize=8)
    plot_series(a[4], runs, "signed", "logit_fourier_energy", PURPLE, "actual-logit addition energy")
    plot_series(a[4], runs, "signed", "fourier_energy", ORANGE, "hidden addition energy")
    a[4].set(ylim=(0, 1), ylabel="fraction (different denominators)",
             title="E. Output and hidden Fourier energies")
    a[4].legend(fontsize=8)
    crossed = cross_panel(a[5], runs)
    for i, ax in enumerate(a[:5]):
        shared_time_axis(ax, zero=i in [2, 3])
        if not ax.lines:
            ax.text(.5, .5, "No observed data", ha="center", transform=ax.transAxes)
    available = sum(r.get("trajectory") is not None for r in runs)
    completed = sum(r.get("status") == "complete" for r in runs)
    fig.subplots_adjust(left=.063, right=.986, top=.856, bottom=.147,
                        wspace=.30, hspace=.37)
    aggregation = ("Thick curves: pointwise medians; bands: IQR; faint curves: each observed run. "
                   if len(runs) > 1 else "Single declared run; no temporal smoothing. ")
    fig.text(.5, .025,
             aggregation + f"{len(runs)} declared, {available} with data, {completed} complete.\n"
             "No missing/failed run is removed from the roster; medians use only exact observed steps, without extrapolation.\n"
             "A is the nonconstant addition-Fourier projection of actual logits; Z retains the constant and other modes. "
             "Panel F swaps saved output components, not networks; raw counts follow replicate order.\n"
             "CE / accuracy use every training step; Fourier and signed quantities are observed every 100 steps.\n"
             "Full-grid projections are rule-aware observers. Energy is not AGOP purity. These are empirical replications, not an iid grokking theorem.",
             ha="center", va="bottom", fontsize=8.25, color=GRAY)
    return fig, crossed


def overview(runs):
    fig, axes = plt.subplots(2, 3, figsize=(16.2, 9.7))
    fig.suptitle("Dense ReLU replication: compare moduli without selecting successful seeds", fontsize=15, y=.976)
    fig.text(.5, .933, "Locked width 512, c24 initialization and original Polar / normalized-head update · no adaptive gating or balancing",
             ha="center", fontsize=10, color=GRAY)
    specs = [
        ("trajectory", "train_ce", None, "A. Training cross-entropy", "raw CE", True),
        ("trajectory", "test_strict_count", "n_test", "B. Actual held-out accuracy", "strict accuracy", False),
        ("signed", "separated_certified", "n_test", "C. Signed accuracy lower bound", "evaluated lower bound", False),
        ("signed", "fourier_energy", None, "D. Hidden addition-Fourier energy", "actual hidden-energy fraction", False),
        ("signed", "logit_fourier_energy", None, "E. Actual-logit addition-Fourier energy", "energy fraction", False),
        ("signed", "row_lower_min", None, "F. Minimum separated row bound", "signed margin lower bound", False),
    ]
    summaries = {}
    for p in sorted({r["p"] for r in runs}):
        subset = [r for r in runs if r["p"] == p]
        label = f"p = {p} ({len(subset)} declared runs)"
        for ax, (table, field, den, title, ylabel, log) in zip(axes.ravel(), specs):
            data = plot_series(ax, subset, table, field, P_COLORS.get(p, GRAY), label,
                               denominator=den)
            summaries[f"p{p}_{field}"] = {k: v.tolist() for k, v in data.items()}
            ax.set(title=title, ylabel=ylabel)
            if log:
                ax.set_yscale("log")
            elif field == "fourier_energy":
                ax.set_ylim(*hidden_energy_axis_limits(runs))
            elif field != "row_lower_min":
                ax.set_ylim(-.02, 1.04)
            if den is not None:
                ax.yaxis.set_major_formatter(PercentFormatter(1))
            shared_time_axis(ax, zero=field == "row_lower_min")
    for ax in axes.ravel():
        ax.legend(fontsize=8)
    fig.subplots_adjust(left=.063, right=.986, top=.854, bottom=.125, wspace=.30, hspace=.36)
    fig.text(.5, .03,
             "Thick curves: per-modulus medians; bands: IQR; faint curves: every observed seed. "
             "No interpolation, extrapolation, or success-based exclusion.\n"
             "Shading is the fixed 15–20k reference window, not a claim that each run plateaus or groks there. "
             "All Fourier projections / bounds are evaluated observers.\n"
             "Input splits are sampled without replacement on each finite grid; three repetitions per modulus do not establish a high-probability theorem.",
             ha="center", fontsize=8.5, color=GRAY)
    return fig, summaries


def load_npz(path):
    if not path.exists():
        return None
    with np.load(path, allow_pickle=False) as f:
        return {k: f[k] for k in f.files}


def load_sweep(directory):
    root = json.loads((directory / "RESULTS.json").read_text())
    checks = json.loads((directory / "CHECKS.json").read_text())
    if checks["status"] != "complete":
        raise ValueError("Independent saved-array checks are not complete")
    if "results_sha256" in checks:
        assert checks["results_sha256"] == sha(directory / "RESULTS.json")
    inputs = {
        str(directory / "RESULTS.json"): sha(directory / "RESULTS.json"),
        str(directory / "CHECKS.json"): sha(directory / "CHECKS.json"),
    }

    def validate_hashes(record, base):
        for name, expected in record.get("output_sha256", {}).items():
            path = Path(name)
            if not path.is_absolute():
                path = base / path
            assert sha(path) == expected, f"Hash mismatch: {path}"
            inputs[str(path)] = expected

    validate_hashes(root, directory)
    runs = []
    root_runs = {r["config"]["name"]: r for r in root["runs"]}
    for cfg in root["roster"]:
        name = cfg["name"]
        case = directory / name
        result_path = case / "RESULTS.json"
        if result_path.exists():
            record = json.loads(result_path.read_text())
            inputs[str(result_path)] = sha(result_path)
        else:
            record = root_runs[name]
        assert record["config"]["name"] == name
        validate_hashes(record, case)
        tables = {"trajectory": case / "trajectory.npz",
                  "signed": case / "diagnostics.npz",
                  "cross": case / "crossed_components.npz"}
        run = {
            "run_id": name, "p": cfg["p"], "config": cfg,
            "status": record["status"], "n_train": record["n_train"],
            "n_test": record["n_test"], "record": record,
        }
        for key, path in tables.items():
            run[key] = load_npz(path)
            if path.exists():
                inputs[str(path)] = sha(path)
        if run["cross"] is not None:
            np.testing.assert_array_equal(run["cross"]["step"], [10000, 20000])
        for table in ["trajectory", "signed"]:
            z = run[table]
            if z is not None:
                assert np.all(np.diff(z["step"]) > 0)
                assert len(z["step"]) and z["step"][-1] <= cfg["steps"]
                assert run["n_train"] > 0 and run["n_test"] > 0
        runs.append(run)
    assert len(runs) == len(root["roster"])
    return root, runs, inputs


def main():
    root, runs, inputs = load_sweep(HERE)
    with (HERE / "PLOT_LAUNCH.json").open("x") as f:
        json.dump({"source_sha256": sha(__file__), "input_sha256": inputs,
                   "roster": [r["run_id"] for r in runs],
                   "scientific_changes": "none; saved-array presentation only"}, f, indent=2)
    outputs = {}

    def save(fig, name):
        for ext in ["png", "svg"]:
            path = HERE / f"{name}.{ext}"
            assert not path.exists(), f"Refusing to overwrite {path}"
            fig.savefig(path)
            outputs[str(path)] = sha(path)
        plt.close(fig)

    crosses = {}
    for run in runs:
        cfg = run["config"]
        title = f"p = {run['p']} · replicate {cfg['replicate']}: loss, Fourier alignment, and test recovery"
        subtitle = (f"Dense ReLU width {cfg['width']} · {run['n_train']} train / {run['n_test']} test · "
                    f"init seed {cfg['init_seed']}, split seed {cfg['split_seed']} · status: {run['status']}")
        fig, raw = six_panel([run], title, subtitle)
        save(fig, f"{run['run_id']}_signed_recovery")
        crosses[run["run_id"]] = raw
    for p in sorted({r["p"] for r in runs}):
        subset = [r for r in runs if r["p"] == p]
        title = f"p = {p}: all {len(subset)} declared dense-ReLU replications"
        subtitle = ("Same width 512, c24 initialization scale and original Polar / normalized-head update · "
                    "independent initialization and split seeds · no loss balancing")
        fig, raw = six_panel(subset, title, subtitle)
        save(fig, f"p{p}_aggregate_signed_recovery")
        crosses[f"p{p}"] = raw
    fig, summaries = overview(runs)
    save(fig, "sweep_overview")
    data_path = HERE / "PLOT_DATA.json"
    with data_path.open("x") as f:
        json.dump({"roster": [dict(run_id=r["run_id"], p=r["p"], status=r["status"],
                                   n_train=r["n_train"], n_test=r["n_test"])
                              for r in runs],
                   "crossed_raw_counts_in_display_order": crosses,
                   "overview_exact_step_summaries_and_contributor_counts": summaries}, f)
    outputs[str(data_path)] = sha(data_path)
    with (HERE / "FIGURES.json").open("x") as f:
        json.dump({"status": "complete", "input_sha256": inputs,
                   "source_sha256": sha(__file__), "outputs": outputs,
                   "number_of_declared_runs": len(runs),
                   "claims": "Empirical replications only; no causal optimizer or iid theorem."},
                  f, indent=2)
    print(json.dumps({"status": "complete", "figures": len(outputs) - 1,
                      "declared_runs_retained": len(runs)}), flush=True)


if __name__ == "__main__":
    main()
