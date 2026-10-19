from pathlib import Path
import hashlib
import json
import time
import traceback

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np

from batch_config import HERE, PRIOR, ROSTER, MODULI, REPLICATES, STEPS, THREAD_VARS

WORK_SECONDS = 780
TOTAL_SECONDS = 840
BLUE, GREEN, PURPLE, ORANGE, GRAY, RED = "#1976b7", "#008c78", "#7851a9", "#d48300", "#687582", "#b64d57"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9,
    "axes.titlesize": 11, "axes.titleweight": "bold", "axes.spines.top": False,
    "axes.spines.right": False, "axes.grid": True, "grid.alpha": .15,
    "legend.frameon": False, "savefig.dpi": 165, "svg.fonttype": "none"})


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(path, value):
    with Path(path).open("x") as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write("\n")


def load_npz(path):
    if not path.exists():
        return None
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def wilson(k, n):
    assert 0 <= k <= n and n > 0
    z = 1.959963984540054
    phat = k / n
    denominator = 1 + z * z / n
    center = (phat + z * z / (2 * n)) / denominator
    radius = z * np.sqrt(phat * (1 - phat) / n + z * z / (4 * n * n)) / denominator
    return max(0., float(center - radius)), min(1., float(center + radius))


def binary_summary(outcomes):
    n = len(outcomes)
    positive = sum(value is not None and bool(value) for value in outcomes)
    negative = sum(value is not None and not bool(value) for value in outcomes)
    unresolved = n - positive - negative
    low = wilson(positive, n)[0]
    high = wilson(positive + unresolved, n)[1]
    return dict(declared=n, known_positive=positive, known_negative=negative,
                unresolved=unresolved, observed=positive + negative,
                interval_low=low, interval_high=high,
                interval_kind="ordinary nominal Wilson" if not unresolved else
                              "Wilson envelope across possible completions; not a measured-rate CI")


def exact_summary(series, regular_interval=None):
    selected = []
    for steps, values in series:
        steps, values = np.asarray(steps), np.asarray(values, dtype=float)
        assert len(steps) == len(values) and len(np.unique(steps)) == len(steps)
        assert np.isfinite(values).all()
        use = np.ones(len(steps), dtype=bool) if regular_interval is None else steps % regular_interval == 0
        if np.any(use):
            selected.append((steps[use], values[use]))
    if not selected:
        return {key: np.array([]) for key in ["step", "median", "q25", "q75", "n"]}
    grid = np.unique(np.concatenate([steps for steps, _ in selected]))
    matrix = np.full((len(selected), len(grid)), np.nan)
    for row, (steps, values) in enumerate(selected):
        matrix[row, np.searchsorted(grid, steps)] = values
    present = np.isfinite(matrix)
    patterns, inverse = np.unique(present.T, axis=0, return_inverse=True)
    quantiles = np.empty((3, len(grid)))
    for group, mask in enumerate(patterns):
        columns = inverse == group
        quantiles[:, columns] = np.quantile(matrix[np.ix_(mask, columns)], [.25, .5, .75], axis=0)
    return dict(step=grid, q25=quantiles[0], median=quantiles[1], q75=quantiles[2], n=present.sum(0))


def series_for(runs, table, field, denominator=None):
    answer = []
    for run in runs:
        data = run.get(table)
        if data is None or field not in data or not len(data["step"]):
            continue
        values = np.asarray(data[field], dtype=float)
        if denominator:
            values = values / run[denominator]
        answer.append((run, np.asarray(data["step"]), values))
    return answer


def ce_scale(series):
    values = [np.asarray(y) for _, y in series]
    assert all(np.isfinite(y).all() and np.all(y >= 0) for y in values)
    if not values or all(np.all(y > 0) for y in values):
        return "log", None
    positive = np.concatenate([y[y > 0] for y in values])
    return "symlog", max(np.finfo(float).tiny, float(positive.min() / 10)) if len(positive) else 1e-12


def window(run, left):
    matches = [w for w in run["record"].get("windows", []) if w["start"] == left and w["end"] == 20000]
    assert len(matches) <= 1
    return matches[0] if matches else dict(status="unavailable", joint_window_pass=None)


def accuracy_at(run, step):
    data = run.get("trajectory")
    if data is None:
        return None
    hit = np.flatnonzero(data["step"] == step)
    if not len(hit):
        return None
    assert len(hit) == 1
    return int(data["test_strict_count"][hit[0]]) / run["n_test"]


def run_events(run):
    at20 = accuracy_at(run, 20000)
    complete = run["status"] == "complete"
    final = run["record"].get("final", {})
    result = dict(test90_at20k=None if at20 is None else at20 >= .9,
                  test100_at20k=None if at20 is None else at20 == 1.,
                  test100_at40k=None if not complete else final["test_correct"] == run["n_test"],
                  delayed_recovery=None if not complete else bool(final["delayed_recovery"]))
    for left in [10000, 15000]:
        w = window(run, left)
        result[f"joint_{left}_20000"] = bool(w["joint_window_pass"]) if w["status"] == "available" else None
    return result


class Presenter:
    def __init__(self, runs, deadline):
        self.runs, self.deadline = runs, deadline
        self.cache, self.outputs = {}, {}

    def tick(self):
        if time.monotonic() >= self.deadline:
            raise TimeoutError("Saved-array postprocessing work limit reached")

    def plot(self, ax, runs, table, field, color, label, denominator=None, linestyle="-"):
        rows = series_for(runs, table, field, denominator)
        for run, t, y in rows:
            ax.plot(t / 1000, y, color=color, alpha=.12, linewidth=.7, linestyle=linestyle)
            if run["status"] != "complete":
                ax.plot(t[-1] / 1000, y[-1], marker="o", markersize=2.8, markerfacecolor="white",
                        markeredgecolor=color, markeredgewidth=.65, linestyle="none", alpha=.7)
        key = (tuple(r["name"] for r in runs), table, field, denominator)
        if key not in self.cache:
            interval = 100 if table in ["diagnostics", "window_observer"] else None
            self.cache[key] = exact_summary([(t, y) for _, t, y in rows], interval)
        z = self.cache[key]
        if len(z["step"]):
            ax.fill_between(z["step"] / 1000, z["q25"], z["q75"], color=color, alpha=.15, linewidth=0)
            ax.plot(z["step"] / 1000, z["median"], color=color, linewidth=2, linestyle=linestyle, label=label)
        return z

    def finish(self, fig, name):
        self.tick()
        for extension in ["png", "svg"]:
            path = HERE / "figures" / f"{name}.{extension}"
            assert not path.exists()
            fig.savefig(path)
            self.outputs[str(path)] = sha(path)
        plt.close(fig)
        print(json.dumps(dict(figure=name, status="saved")), flush=True)


def status_note(runs):
    complete = sum(r["status"] == "complete" for r in runs)
    observed = sum(r.get("trajectory") is not None for r in runs)
    ends = [int(r["trajectory"]["step"][-1]) for r in runs if r.get("trajectory") is not None]
    return f"{len(runs)} declared · {complete} complete · {observed} observed · common horizon among observed runs: {min(ends):,}" if ends else f"{len(runs)} declared; no observed trajectory"


def time_axis(ax, runs, fixed=False):
    ax.set(xlim=(15, 20) if fixed else (0, STEPS / 1000), xlabel="optimization step (thousands)")
    if not fixed:
        ax.axvspan(15, 20, color=BLUE, alpha=.04, linewidth=0)
        partial = [r for r in runs if r["status"] != "complete" and r.get("trajectory") is not None]
        if partial:
            start = min(r["trajectory"]["step"][-1] for r in partial) / 1000
            ax.axvspan(start, STEPS / 1000, color=GRAY, alpha=.065, linewidth=0)


def loss_axis(ax, runs):
    series = [(t, y) for field in ["train_ce", "test_ce"] for _, t, y in series_for(runs, "trajectory", field)]
    mode, threshold = ce_scale(series)
    ax.set_yscale(mode, **({"linthresh": threshold} if mode == "symlog" else {}))
    ax.set_ylabel("raw CE (log)" if mode == "log" else "raw CE (symlog; zeros retained)")


def cross_panel(ax, runs):
    available = [r for r in runs if r.get("cross") is not None]
    ax.set(title="F. Crossed outputs: median test accuracy", xticks=[0, 1], yticks=[0, 1],
           xticklabels=["10k", "20k"], yticklabels=["10k", "20k"],
           xlabel="complement Z checkpoint", ylabel="addition A checkpoint")
    ax.grid(False)
    if not available:
        ax.text(.5, .5, "No complete checkpoint pair", ha="center", transform=ax.transAxes)
        return
    matrix = np.stack([r["cross"]["test_correct"] / r["n_test"] for r in available])
    q = np.quantile(matrix, [.25, .5, .75], axis=0)
    ax.imshow(q[1], vmin=0, vmax=1, cmap="YlGnBu", origin="upper", aspect="auto")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"median {q[1,i,j]:.1%}\nIQR {q[0,i,j]:.1%}–{q[2,i,j]:.1%}\nn = {len(available)}/{len(runs)}",
                    ha="center", va="center", fontsize=10, color="white" if q[1,i,j] > .58 else "#193442")


def cohort_figure(view, runs, p):
    fig, grid = plt.subplots(2, 3, figsize=(16, 10))
    a = grid.ravel()
    fig.suptitle(f"p = {p}: all {REPLICATES} fresh dense-ReLU initialization/split pairs", fontsize=15, y=.974)
    fig.text(.5, .934, status_note(runs) + " · no adaptive balancing or retuning", ha="center", color=GRAY)
    for field, color, label in [("train_ce", BLUE, "training CE"), ("test_ce", GREEN, "held-out CE")]:
        view.plot(a[0], runs, "trajectory", field, color, label)
    loss_axis(a[0], runs)
    a[0].set_title("A. Actual losses; no imposed floor")
    view.plot(a[1], runs, "trajectory", "train_strict_count", BLUE, "training", "n_train", "--")
    view.plot(a[1], runs, "trajectory", "test_strict_count", GREEN, "held-out", "n_test")
    view.plot(a[1], runs, "diagnostics", "separated_certified", PURPLE, "evaluated signed\naccuracy lower bound", "n_test", ":")
    a[1].set(title="B. Actual accuracy and signed bound", ylim=(-.02, 1.04), ylabel="strict accuracy / lower bound")
    a[1].yaxis.set_major_formatter(PercentFormatter(1))
    view.plot(a[2], runs, "diagnostics", "rule_min", PURPLE, "minimum signed rule margin")
    view.plot(a[2], runs, "diagnostics", "harmful_max", ORANGE, "largest harmful complement")
    a[2].set(title="C. Signed signal versus interference", ylabel="actual-logit contrast")
    view.plot(a[3], runs, "diagnostics", "test_min_margin", GREEN, "actual minimum test margin")
    view.plot(a[3], runs, "diagnostics", "row_lower_min", PURPLE, "minimum separated row bound", linestyle="--")
    view.plot(a[3], runs, "diagnostics", "global_lower", GRAY, "coarser global bound", linestyle=":")
    a[3].axhline(0, color=RED, linewidth=.8)
    a[3].set(title="D. Conservative signed margin bounds", ylabel="margin / sufficient lower bound")
    view.plot(a[4], runs, "diagnostics", "fourier_energy", ORANGE, "hidden addition energy")
    view.plot(a[4], runs, "diagnostics", "logit_fourier_energy", PURPLE, "actual-logit addition energy")
    a[4].set(title="E. Hidden and output Fourier energies", ylim=(0, 1), ylabel="fractions (different denominators)")
    cross_panel(a[5], runs)
    for ax in a[:5]:
        time_axis(ax, runs)
        ax.legend(fontsize=8, loc="best")
    fig.subplots_adjust(left=.066, right=.987, top=.861, bottom=.135, hspace=.36, wspace=.30)
    fig.text(.5, .027, "Thick: medians; bands: IQR; faint: every observed individual, with no outcome-based exclusions. Hollow dots: resource-censored endpoints.\n"
             "CE / accuracy retain every observed step. Signed/Fourier medians use observed regular100-step entries; individual off-grid endpoints remain visible.\n"
             "Gray tails have changing contributors; counts are exported. Blue shading is the fixed15–20k reference, not a claim that every run plateaus there.\n"
             "A is the nonconstant addition-Fourier component of actual logits; Z is the complement. Panel F is an observer-only output swap, not retraining.\n"
             "No AGOP/ridge, scalar-energy causality, optimizer-causality, or initialization theorem is claimed. Raw per-seed crossed counts are retained in SUMMARY.json.",
             ha="center", color=GRAY, fontsize=8)
    view.finish(fig, f"p{p}_aggregate")


def raw_loss_figure(view, cohorts):
    fig, axes = plt.subplots(2, 3, figsize=(16, 9.7))
    fig.suptitle("Raw cross-entropy across all moduli and all twenty-pair cohorts", fontsize=15, y=.975)
    for ax, (p, runs) in zip(axes.ravel(), cohorts):
        view.plot(ax, runs, "trajectory", "train_ce", BLUE, "training CE")
        view.plot(ax, runs, "trajectory", "test_ce", GREEN, "held-out CE")
        loss_axis(ax, runs)
        ax.set_title(f"p = {p}; {sum(r['status']=='complete' for r in runs)}/{len(runs)} complete")
        time_axis(ax, runs)
        ax.legend(fontsize=8)
    fig.subplots_adjust(left=.066, right=.985, top=.89, bottom=.12, hspace=.37, wspace=.3)
    fig.text(.5, .025, "Actual CE, not normalized losses. Faint curves retain every observed seed; medians/IQR use exact observed steps.\n"
             "Blue: fixed15–20k window. Gray: changing-contributor censored tails. No extrapolation, successful-run filtering, or imposed CE floor.",
             ha="center", color=GRAY, fontsize=9)
    view.finish(fig, "raw_loss_all_moduli")


def window_limits(runs, table, field):
    values = [y[(t >= 15000) & (t <= 20000)] for _, t, y in series_for(runs, table, field)]
    values = [y for y in values if len(y)]
    if not values:
        return None
    joined = np.concatenate(values)
    lo, hi = float(joined.min()), float(joined.max())
    pad = max(.08 * (hi - lo), .01 * max(abs(lo), abs(hi)), 1e-14)
    return lo - pad, hi + pad


def fixed_window_figure(view, cohorts):
    fig, axes = plt.subplots(len(cohorts), 3, figsize=(15, 3.1 * len(cohorts) + 1.4))
    fig.suptitle("Locked15k–20k window: raw training CE, continuing Fourier change, and test accuracy", fontsize=15, y=.982)
    all_delta = []
    for p, runs in cohorts:
        for run in runs:
            data = run.get("diagnostics")
            if data is None:
                continue
            where = np.flatnonzero(data["step"] == 15000)
            if len(where):
                run["window_observer"] = dict(step=data["step"],
                    hidden_change=data["fourier_energy"] - data["fourier_energy"][where[0]])
                t = data["step"]
                all_delta.extend(run["window_observer"]["hidden_change"][(t >= 15000) & (t <= 20000)].tolist())
    common_delta = None
    if all_delta:
        lo, hi = min(all_delta), max(all_delta)
        pad = max(.08 * (hi - lo), 1e-5)
        common_delta = (min(0., lo - pad), hi + pad)
    for row, (p, runs) in enumerate(cohorts):
        a = axes[row]
        view.plot(a[0], runs, "trajectory", "train_ce", BLUE, "training CE")
        a[0].set(title=f"p = {p}: raw training CE", ylabel="raw CE (linear)")
        limits = window_limits(runs, "trajectory", "train_ce")
        if limits:
            a[0].set_ylim(max(0., limits[0]), limits[1])
        a[0].ticklabel_format(axis="y", style="sci", scilimits=(-2, 2), useOffset=False)
        view.plot(a[1], runs, "window_observer", "hidden_change", PURPLE, "hidden energy change from15k")
        a[1].set(title="Fourier change after the fixed start", ylabel="addition-energy change")
        if common_delta:
            a[1].set_ylim(*common_delta)
        a[1].axhline(0, color=GRAY, linewidth=.7)
        view.plot(a[2], runs, "trajectory", "test_strict_count", GREEN, "held-out accuracy", "n_test")
        a[2].set(title="Actual held-out accuracy", ylabel="strict accuracy", ylim=(-.02, 1.04))
        a[2].yaxis.set_major_formatter(PercentFormatter(1))
        for ax in a:
            time_axis(ax, runs, fixed=True)
            ax.legend(fontsize=7.5)
    fig.subplots_adjust(left=.09, right=.984, top=.94, bottom=.058, hspace=.53, wspace=.33)
    fig.text(.5, .018, "Every observed seed retained; missing observations are never extrapolated. CE uses raw units and panel-specific linear y-ranges (compare relative variation in REPORT.md).\n"
             "Fourier changes have common y-limits acrossp, with baseline fixed at15k; accuracy uses a common0–100%scale. This is not a post-selected window or proof of grokking.",
             ha="center", color=GRAY, fontsize=8.5)
    view.finish(fig, "fixed_window_all_moduli")


def interval_text(event):
    suffix = " (possible-completion envelope)" if event["unresolved"] else " (nominal Wilson)"
    return f"{100*event['interval_low']:.1f}%–{100*event['interval_high']:.1f}%" + suffix


def make_summary(runs):
    cohorts = []
    for p in MODULI:
        subset = [r for r in runs if r["p"] == p]
        events = [run_events(r) for r in subset]
        cohorts.append(dict(p=p, declared=len(subset),
            statuses={status:sum(r["status"] == status for r in subset) for status in ["complete", "partial", "failed", "unrun"]},
            events={key:binary_summary([event[key] for event in events]) for key in events[0]}))
    rows = []
    for run in runs:
        rows.append(dict(name=run["name"], p=run["p"], config=run["config"], status=run["status"],
            n_train=run["n_train"], n_test=run["n_test"], completed_steps=run["record"].get("completed_steps"),
            observed_states=run["record"].get("observed_states", 0), test_accuracy_at20k=accuracy_at(run, 20000),
            final=run["record"].get("final", {}), windows=run["record"].get("windows", []), events=run_events(run),
            error=run["record"].get("error"), resource_stop_reason=run["record"].get("resource_stop_reason"),
            crossed_test_correct=None if run.get("cross") is None else run["cross"]["test_correct"].astype(int).tolist()))
    return dict(declared_runs=len(runs), by_p=cohorts, runs=rows,
                statistical_note="Nominal Wilson summaries; with unresolved outcomes the interval spans all possible completions, not a measured-rate CI or formal coverage guarantee.",
                new_training_updates=0, initialization_theorem=False, causal_optimizer_theorem=False,
                earlier_runs_added_to_denominators=False)


def report_text(summary, png_paths, training_status):
    lines = ["# Twenty fresh pairs per modulus: replication report", "",
        f"Training status: {training_status}. All {len(ROSTER)} declared initialization/split pairs are retained; "
        f"there are {REPLICATES} new pairs for each p in {MODULI}. The earlier selected run and nine-run replication are not added to these denominators.", "",
        "## Setup", "", "Biasless dense1HL ReLU, width512; input[e_a;e_b], target(a+b) modp. Uniform fixed-size80%entry split without replacement. "
        "Independent seeded NumPy Gaussian initialization: W entries N(0,24²/p), V entries N(0,1/(24²·512)). "
        "This is transfer/replication of the previously selected c24 configuration, not a new hyperparameter search or dimension-invariant guarantee.", "",
        "The unchanged simultaneous full-batch training-CE updates are:", "", "```text",
        "W_next = W + .03*(supportPolar(-grad_W CE)/sqrt(retained rank) - .005*W)",
        "V_next = V + .03*(unitF(-grad_V CE) - .2*V)", "```", "",
        "The Polar support cutoff is eps64·max(matrix shape)·sigma_max; zero force gives zero normalized direction. "
        "There is no momentum Muon, row-sphere constraint, adaptive loss balancing, optimizer-gating intervention, head replacement, schedule, or Fourier feedback. "
        "ReLU gates may change naturally. Each complete run has40,000updates. Three single-thread workers have a1h/arm and12h/global cooperative resource envelope; "
        "a resource-censored run is not a measured failure-to-grok.", "",
        "## Completion and split sizes", "",
        "| p | Train / held-out | Complete | Partial | Failed | Unrun | Declared |",
        "|---:|---:|---:|---:|---:|---:|---:|"]
    for group in summary["by_p"]:
        p, s = group["p"], group["statuses"]
        n = int(.8 * p * p)
        lines.append(f"|{p}|{n}/{p*p-n}|{s['complete']}|{s['partial']}|{s['failed']}|{s['unrun']}|{group['declared']}|")
    lines += ["", "## Locked binary outcomes and uncertainty summaries", "",
        "Known positive/negative/unresolved counts always sum to20 for each modulus. A common20k outcome or a fully observed window can be known even if a later tail is censored. "
        "The40k and delayed-recovery outcomes are unresolved unless the run completed. No unresolved outcome is silently imputed as a measured failure.", "",
        "With no unresolved trials, the table gives ordinary nominal95%Wilson intervals. Withu unresolved trials andk known positives, it gives "
        "[WilsonLower(k,20), WilsonUpper(k+u,20)]: an interval spanning possible completions, **not a measured-rate confidence interval or a formal coverage guarantee**. "
        "These are descriptive small-sample summaries, not a high-probability initialization theorem.", "",
        "| p | Event | Known+ | Known− | Unresolved | Known+ / declared |95%nominal interval / completion envelope|",
        "|---:|---|---:|---:|---:|---:|---|"]
    for group in summary["by_p"]:
        for name, event in group["events"].items():
            lines.append(f"|{group['p']}|{name}|{event['known_positive']}|{event['known_negative']}|{event['unresolved']}|"
                         f"{event['known_positive']}/{event['declared']}|{interval_text(event)}|")
    lines += ["", "Joint-window criteria, locked before this batch: every training row strictly correct throughout; rawCE(max−min)/mean≤10%; "
        "hidden addition-energy gain≥.01; test accuracy≤50%at start and≥90%at end; signed accuracy lower bound≥90%at end. "
        "Windows are10k–20k and15k–20k. Zero-mean CE has undefined relative variation and does not pass this flag. "
        "A run that learned before the window, or recovered later, can fail this specific flag while displaying a different recovery pattern. "
        "No favorable window is selected after seeing outcomes.", "",
        "Delayed recovery requires a complete run, a1,000step fitted-training/≤10%test interval beginning at the first persistently fitted training state, "
        "and subsequent test≥90%through40k. Persistence is finite-horizon only. This stringent flag is not every possible definition of grokking.", "",
        "## All120 declared runs", "",
        "Rawloss levels, all component window criteria, exact seed/split configurations, errors/statuses, and all four raw crossed-cell counts are retained in SUMMARY.json.", "",
        "|Run|Status; last observed step|Train CE(last observed)|Test accuracy20k|Test correct40k|Joint10–20k|Joint15–20k|",
        "|---|---|---:|---:|---:|---|---|"]
    for row in summary["runs"]:
        final = row["final"]
        ce = "unavailable" if final.get("train_ce") is None else f"{final['train_ce']:.7g}"
        at20 = "unresolved" if row["test_accuracy_at20k"] is None else f"{row['test_accuracy_at20k']:.1%}"
        at40 = f"{final['test_correct']}/{row['n_test']}" if row["status"] == "complete" else "unresolved"
        flags = ["unresolved" if row["events"][key] is None else str(row["events"][key]) for key in ["joint_10000_20000", "joint_15000_20000"]]
        last = row["completed_steps"] if row["observed_states"] else "none"
        lines.append(f"|{row['name']}|{row['status']}; {last}|{ce}|{at20}|{at40}|{flags[0]}|{flags[1]}|")
    lines += ["", "## Figures", "",
        "All observed individual trajectories are retained. Thick curves are per-p medians with IQR. "
        "Signed/Fourier aggregate statistics use exact observed regular100-step points; off-grid terminal observations remain visible on individuals. "
        "Counts vary after resource censoring and are exported; changes in aggregate composition are not optimizer jumps. No interpolation or extrapolation is performed.", ""]
    lines += [f"- [{Path(path).stem.replace('_', ' ')}](<{path}>)" for path in png_paths]
    lines += ["", "## Interpretation limits", "",
        "The signed actual-head inequality splits actual logits into their nonconstant addition-Fourier componentA and remainderZ. "
        "The weakest signed rule contrast minus the largest harmful remainder contrast lower-bounds each row's actual worst-rival margin. "
        "This static bound is valid independently of whether it becomes useful on a run. Its evaluation uses the full-grid rule as an observer. "
        "Crossed outputsA(time1)+Z(time2) are hypothetical output tables, not networks trained under interventions. "
        "NoAGOP purity, ridge refit, scalar-energy causality, optimizer-causality, infinite-time recovery or initialization theorem follows from these plots. "
        "Ordinary floating-point checks are not interval-arithmetic certificates.", ""]
    return "\n".join(lines)


def main():
    import os
    assert all(os.environ.get(name) == "1" for name in THREAD_VARS)
    started = time.monotonic()
    result_hash = sha(HERE / "RESULTS.json")
    checks = json.loads((HERE / "CHECKS.json").read_text())
    assert checks["status"] == "complete" and checks["results_sha256"] == result_hash
    sources = {str(HERE / name):sha(HERE / name) for name in ["postprocess.py", "test_postprocess.py", "batch_config.py", "SCOPE.txt"]}
    dump(HERE / "POSTPROCESS_LAUNCH.json", dict(source_sha256=sources, results_sha256=result_hash,
         checks_sha256=sha(HERE / "CHECKS.json"), work_seconds=WORK_SECONDS, total_seconds=TOTAL_SECONDS, new_training_updates=0))
    outputs, inputs = {}, {str(HERE / "RESULTS.json"):result_hash, str(HERE / "CHECKS.json"):sha(HERE / "CHECKS.json")}
    try:
        root = json.loads((HERE / "RESULTS.json").read_text())
        assert root["roster"] == ROSTER and len(root["runs"]) == len(ROSTER)
        closed = json.loads((HERE / "CLOSED.json").read_text())
        assert closed["results_sha256"] == result_hash and closed["status"] == root["status"]
        inputs[str(HERE / "CLOSED.json")] = sha(HERE / "CLOSED.json")
        for mapping in [root.get("source_sha256", {}), root.get("output_sha256", {})]:
            for path, expected in mapping.items():
                assert sha(path) == expected
                inputs[path] = expected
        runs = []
        for cfg, record in zip(ROSTER, root["runs"]):
            assert record["config"] == cfg
            folder = HERE / cfg["name"]
            run = dict(name=cfg["name"], p=cfg["p"], config=cfg, record=record,
                       status=record["status"], n_train=record["n_train"], n_test=record["n_test"])
            for key, filename in [("trajectory", "trajectory.npz"), ("diagnostics", "diagnostics.npz"), ("cross", "crossed_components.npz")]:
                path = folder / filename
                if path.exists():
                    expected = record["output_sha256"][str(path)]
                    assert sha(path) == expected
                    inputs[str(path)] = expected
                    run[key] = load_npz(path)
                else:
                    run[key] = None
            if run["cross"] is not None:
                np.testing.assert_array_equal(run["cross"]["step"], [10000, 20000])
                assert run["cross"]["test_correct"].shape == (2, 2)
            runs.append(run)
        (HERE / "figures").mkdir()
        view = Presenter(runs, started + WORK_SECONDS)
        cohorts = [(p, [run for run in runs if run["p"] == p]) for p in MODULI]
        for p, subset in cohorts:
            assert len(subset) == REPLICATES
            cohort_figure(view, subset, p)
        raw_loss_figure(view, cohorts)
        fixed_window_figure(view, cohorts)
        outputs.update(view.outputs)
        summary = make_summary(runs)
        summary.update(input_sha256=inputs, source_sha256=sources, training_status=root["status"], prior_configuration_source=str(PRIOR))
        dump(HERE / "SUMMARY.json", summary)
        outputs[str(HERE / "SUMMARY.json")] = sha(HERE / "SUMMARY.json")
        pngs = [path for path in outputs if Path(path).suffix == ".png"]
        with (HERE / "REPORT.md").open("x") as handle:
            handle.write(report_text(summary, pngs, root["status"]))
        outputs[str(HERE / "REPORT.md")] = sha(HERE / "REPORT.md")
        aggregate_arrays = {}
        contributor_notes = []
        for key, z in view.cache.items():
            names, table, field, denominator = key
            tag = f"{names[0].split('_')[0]}__{table}__{field}"
            for metric, values in z.items():
                aggregate_arrays[f"{tag}__{metric}"] = values
            contributor_notes.append(dict(cohort=list(names), table=table, field=field, denominator=denominator,
                observed_step_count=len(z["step"]), min_contributors=int(z["n"].min()) if len(z["n"]) else 0,
                max_contributors=int(z["n"].max()) if len(z["n"]) else 0))
        with (HERE / "AGGREGATES.npz").open("xb") as handle:
            np.savez_compressed(handle, **aggregate_arrays)
        outputs[str(HERE / "AGGREGATES.npz")] = sha(HERE / "AGGREGATES.npz")
        for path, expected in {**inputs, **sources}.items():
            view.tick()
            assert sha(path) == expected
        elapsed = time.monotonic() - started
        assert elapsed <= TOTAL_SECONDS
        dump(HERE / "FIGURES.json", dict(status="complete", outputs=outputs, input_sha256=inputs, source_sha256=sources,
             contributor_notes=contributor_notes, declared_runs=len(ROSTER), new_training_updates=0, elapsed_seconds=elapsed))
        dump(HERE / "POSTPROCESS_CLOSED.json", dict(status="complete", elapsed_seconds=elapsed,
             figures_sha256=sha(HERE / "FIGURES.json"), summary_sha256=sha(HERE / "SUMMARY.json"), report_sha256=sha(HERE / "REPORT.md")))
        print(json.dumps(dict(status="complete", declared_runs=len(runs), pngs=len(pngs), elapsed_seconds=elapsed)), flush=True)
    except Exception as exc:
        dump(HERE / "POSTPROCESS_FAILED.json", dict(status="failed", error=str(exc), traceback=traceback.format_exc(), outputs=outputs))
        raise


if __name__ == "__main__":
    main()
