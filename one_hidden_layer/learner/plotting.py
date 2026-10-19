from __future__ import annotations

from pathlib import Path
import textwrap

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


PALETTE = {
    "actual_train": ("#0072B2", "--"),
    "actual_test": ("#009E73", "-"),
    "refit_test": ("#E69F00", ":"),
    "refit_train": ("#0072B2", ":"),
    "fourier_addition": ("#7B4AB5", "-"),
    "fourier_availability": ("#56B4E9", "-."),
    "agop_current": ("#CC79A7", "-"),
    "agop_fixed": ("#CC79A7", "--"),
    "difference_or_residual": ("#D55E00", "--"),
    "oracle_projection_error": ("#666666", ":"),
    "total_or_neutral": ("#333333", "-"),
    "random_or_threshold": ("#999999", ":"),
}
_MISSING = object()


def _number(value):
    try:
        result = float(value)
        return result if np.isfinite(result) else np.nan
    except (ValueError, TypeError):
        return np.nan


def _fmt(value, digits=4):
    value = _number(value)
    return f"{value:.{digits}g}" if np.isfinite(value) else "unresolved"


def _columns(data):
    if data is None:
        return {}
    if isinstance(data, dict):
        result = {key: np.asarray(value) for key, value in data.items()
                  if isinstance(value, (list, tuple, np.ndarray))}
        if "step" not in result:
            return {}
        size = len(result["step"])
        if any(value.ndim != 1 or len(value) != size for value in result.values()):
            raise ValueError("Every trajectory/observer column must be a 1-D step-aligned array")
    else:
        records = list(data)
        keys = set().union(*(row.keys() for row in records)) if records else set()
        result = {key: np.asarray([row.get(key, _MISSING) for row in records], dtype=object)
                  for key in keys}
    if "step" not in result:
        return {}
    step = np.asarray([_number(x) for x in result["step"]], dtype=float)
    if not np.all(np.isfinite(step)) or np.any(np.diff(step) <= 0):
        raise ValueError("Global update steps must be finite and strictly increasing")
    result["step"] = step
    return result


def _series(data, key):
    if key not in data or "step" not in data:
        return np.empty(0), np.empty(0)
    recorded = np.asarray([x is not _MISSING for x in data[key]], dtype=bool)
    return data["step"][recorded], np.asarray([_number(x) for x in data[key][recorded]], dtype=float)


def _has(data, key):
    return np.any(np.isfinite(_series(data, key)[1]))


def _style(name, **kwargs):
    color, linestyle = PALETTE[name]
    return dict(color=color, linestyle=linestyle, linewidth=2.2, **kwargs)


def _line(ax, data, key, role, label, *, positive=False, **kwargs):
    x, y = _series(data, key)
    if not np.any(np.isfinite(y)):
        return False
    if positive:
        y = np.where(y > 0, y, np.nan)
    if not np.any(np.isfinite(y)):
        return False
    ax.plot(x, y, label=label, **_style(role, **kwargs))
    return True


def _valid_agop_line(ax, data, key, validity_key, role, label):
    x, y = _series(data, key)
    vx, vy = _series(data, validity_key)
    validity = dict(zip(vx, vy))
    y = np.asarray([value if validity.get(step) == 1 else np.nan
                    for step, value in zip(x, y)])
    if not np.any(np.isfinite(y)):
        return False
    ax.plot(x, y, label=label, **_style(role))
    return True


def _legend(ax, **kwargs):
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(handles, labels, fontsize=9, framealpha=0.94,
                  borderpad=0.35, handlelength=2.5, labelspacing=0.35, **kwargs)


def _missing(ax, reason="Observer was not recorded in this trajectory."):
    ax.text(0.5, 0.5, "Not measured\n" + textwrap.fill(reason, 48),
            ha="center", va="center", transform=ax.transAxes, fontsize=10,
            color="#555555", bbox=dict(facecolor="white", edgecolor="none", alpha=.92))


def _note(ax, text, *, location="bottom", color="#444444"):
    y = .025 if location == "bottom" else .975
    ax.text(.025, y, text, transform=ax.transAxes, va="bottom" if location == "bottom" else "top",
            fontsize=9, color=color, bbox=dict(facecolor="white", edgecolor="none", alpha=.9),
            linespacing=1.25)


def _window(score):
    t0, t1 = _number(score.get("t0")), _number(score.get("t1"))
    return (t0, t1) if np.isfinite(t0 + t1) and t1 > t0 else None


def _base_axis(ax, title, ylabel, extent, window, *, fraction=False):
    ax.set_title(title, loc="left", fontsize=11, weight="bold", pad=8)
    ax.set_xlabel("Global optimization updates", fontsize=9)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.set_xlim(extent)
    ax.grid(True, color="#E5E7EB", linewidth=.65, alpha=.9)
    ax.set_axisbelow(True)
    ax.tick_params(labelsize=9)
    if fraction:
        ax.set_ylim(0, 1)
    if window:
        ax.axvspan(*window, color="#DDE3E8", alpha=.3, zorder=-1)
    if extent[1] >= 10000:
        ax.ticklabel_format(style="sci", axis="x", scilimits=(3, 3), useMathText=True)


def _exact(data, key, step):
    x, y = _series(data, key)
    indices = np.flatnonzero(x == step)
    return y[indices[0]] if len(indices) else np.nan


def _delta(data, key, window):
    return (_exact(data, key, window[1]) - _exact(data, key, window[0])) if window else np.nan


def _event(ax, score, field, label, role, y, bracket_field=None):
    t = _number(score.get(field))
    bracket = score.get(bracket_field or (field + "_bracket"))
    if isinstance(bracket, dict):
        bracket = [bracket.get("lower"), bracket.get("upper")]
    if isinstance(bracket, (list, tuple)) and len(bracket) == 2:
        low, high = map(_number, bracket)
        if np.isfinite(low + high):
            ax.axvspan(low, high, color=PALETTE[role][0], alpha=.13)
            t = high if not np.isfinite(t) else t
    if np.isfinite(t):
        linestyle = {"T_fit": "--", "T_refit": ":", "T_grok": "-."}.get(field, ":")
        ax.axvline(t, color=PALETTE[role][0], linestyle=linestyle, linewidth=1.1)
        ax.text(t, y, " " + label, transform=ax.get_xaxis_transform(), rotation=90,
                va="top", ha="left", fontsize=9, color=PALETTE[role][0],
                bbox=dict(facecolor="white", edgecolor="none", alpha=.8, pad=.2))


def _footer(config, score, sheet, obs=None):
    rule = config.get("window_rule", score.get("window_rule", "not specified"))
    if isinstance(rule, dict):
        rule = rule.get("kind", str(rule))
    source = str(config.get("source_label", config.get("source", "source recorded in run manifest")))
    track = config.get("evidence_label", "descriptive run; not an initialization theorem")
    text = f"Protocol v1 | {track} | Window rule: {rule}. Unsmoothed; shaded interval is not a success certificate."
    if sheet == "mechanism":
        text += " Signed bounds are pointwise before aggregation; CE linearization is not a Taylor certificate."
        text += "\nMargin bands: example-wise 5th–95th percentiles. Head discrepancy is a class-centered infinity norm. Oracle error is not train-only refit risk."
    if sheet == "trajectories":
        window = _window(score)
        ranks = ("/".join(_fmt(_exact(obs, "agop_q4_mapped_rank", t), 2) for t in window)
                 if window and obs is not None else "unresolved")
        text += (f"\nAGOP: class-centered logits; Gram=train; mapping=held-out; requested q=4; mapped ranks at endpoints={ranks}. "
                 "Ridge: raw hidden features; no intercept; fixed alpha=0.001 in mean-loss convention.")
    if sheet == "controls" and obs is not None and _window(score):
        window = _window(score)
        rank_info = []
        for label, key in [("positive", "positive_rank"), ("selected", "selected_rank"), ("mapped", "mapped_rank"), ("target", "target_rank")]:
            rank_info.append(label + "=" + "/".join(_fmt(_exact(obs, "agop_q4_"+key, t), 3) for t in window))
        gaps = "/".join(_fmt(_exact(obs, "agop_q4_relative_eigengap", t), 3) for t in window)
        text += "\nAGOP endpoint ranks: " + "; ".join(rank_info) + "; relative boundary gaps=" + gaps + ". Primary validity requires a gap >1e−8."
    return "\n".join(textwrap.fill(line, 190) for line in text.splitlines()) + "\n" + textwrap.fill(f"Source: {source}", 185)


def _make_sheet(config, score, subtitle):
    fig, axes = plt.subplots(2, 3, figsize=(15, 8.5))
    fig.subplots_adjust(left=.065, right=.965, bottom=.17, top=.87, wspace=.31, hspace=.5)
    p = config.get("p", "?")
    width = config.get("width", config.get("m", config.get("widths", "?")))
    run = config.get("run_id", "single trajectory")
    activation = config.get("activation", "ReLU")
    fig.suptitle(f"{run}  |  p={p}, width={width}, {activation}  |  {subtitle}", fontsize=14, y=.965)
    fields = ["P10_raw", "F02", "joint_PAG10_F02", "AGOP_supported_joint", "representation_first"]
    labels = []
    for key in fields:
        value = score.get(key)
        labels.append(f"{key}: " + ("pass" if value is True else "fail" if value is False else "unresolved"))
    fig.text(.065, .918, "  |  ".join(labels), fontsize=9, color="#444444")
    return fig, axes.ravel()


def _save(fig, outdir, basename, footer):
    fig.text(.065, .033, footer, va="bottom", ha="left", fontsize=9, color="#555555", linespacing=1.35)
    paths = {}
    for suffix in ("png", "svg", "pdf"):
        path = outdir / f"{basename}.{suffix}"
        fig.savefig(path, dpi=200, facecolor="white", metadata={"Creator": "1HL protocol v1 plotting"})
        paths[suffix] = str(path.resolve())
    plt.close(fig)
    return paths


def _trajectory_sheet(trace, obs, score, config, extent, window):
    fig, axes = _make_sheet(config, score, "Original objective, predictions and distinct feature metrics")
    p = float(config["p"])
    ax = axes[0]
    _base_axis(ax, "A. Original cross-entropy", "Original mean CE (log scale)", extent, window)
    any_loss = _line(ax, trace, "train_ce", "actual_train", "Actual train CE", positive=True)
    any_loss |= _line(ax, trace, "test_ce", "actual_test", "Actual held-out CE", positive=True)
    if any_loss:
        ax.set_yscale("log")
        ax.axhline(np.log(p), **_style("random_or_threshold", label="Uniform logits: log(p)"))
        _legend(ax, loc="upper right")
        all_ce = np.concatenate([_series(trace, k)[1] for k in ("train_ce", "test_ce")])
        if np.any(np.isfinite(all_ce) & (all_ce <= 0)):
            _note(ax, "Nonpositive CE omitted on log axis; no floor applied.")
    else:
        _missing(ax, "No positive raw CE values available; no artificial floor is used.")
    ax = axes[1]
    _base_axis(ax, "B. Raw training CE: common-window zoom", "Original training CE (linear)", window or extent, window)
    if window:
        x, y = _series(trace, "train_ce")
        mask = (x >= window[0]) & (x <= window[1])
        x, y = x[mask], y[mask]
        valid = len(x) and np.all(np.isfinite(y))
        if valid:
            ax.plot(x, y, **_style("actual_train", label="Raw train CE"))
            first = _exact(trace, "train_ce", window[0])
            if np.isfinite(first) and first > 0:
                secondary = ax.secondary_yaxis("right", functions=(lambda z: 100 * (z / first - 1), lambda z: first * (1 + z / 100)))
                secondary.set_ylabel("Change from CE(t0), %", fontsize=9)
                secondary.tick_params(labelsize=9)
            else:
                _note(ax, "Relative change unresolved: CE(t0) missing/nonpositive.")
            minimum, mean, maximum = np.min(y), np.mean(y), np.max(y)
            vmin = (maximum - minimum) / minimum if minimum > 0 else np.nan
            vmean = (maximum - minimum) / mean if mean > 0 else np.nan
            tx, acc = _series(trace, "train_acc")
            acc = acc[(tx >= window[0]) & (tx <= window[1])]
            minacc = np.nanmin(acc) if np.any(np.isfinite(acc)) else np.nan
            text = (f"min / mean / max: {_fmt(minimum)} / {_fmt(mean)} / {_fmt(maximum)}\n"
                    f"v_min={_fmt(100*vmin,3)}%; v_mean={_fmt(100*vmean,3)}%; mean/log(p)={_fmt(mean/np.log(p),3)}\n"
                    f"[{window[0]:g}, {window[1]:g}]; min train accuracy={_fmt(minacc,3)}")
            _note(ax, text, location="top")
            ax.margins(y=.5)
        else:
            _missing(ax, "Original CE observations are absent or nonfinite in the scored window.")
    else:
        _missing(ax, "No valid analysis window was selected; no favorable interval substituted.")
    ax = axes[2]
    _base_axis(ax, "C. Actual and train-only-refit accuracy", "Accuracy", extent, window, fraction=True)
    present = _line(ax, trace, "train_acc", "actual_train", "Actual train")
    present |= _line(ax, trace, "test_acc", "actual_test", "Actual held-out")
    refit = _line(ax, obs, "refit_test_accuracy", "refit_test", "Train-only ridge: held-out")
    ax.axhline(1/p, **_style("random_or_threshold", label="Chance: 1/p"))
    _event(ax, score, "T_fit", "Train fit", "random_or_threshold", .98)
    if score.get("useful_refit") is True:
        _event(ax, score, "T_refit", "Useful refit", "refit_test", .98)
    _event(ax, score, "T_grok", "Actual recovery", "actual_test", .98)
    _legend(ax, loc="lower right")
    if not refit:
        _note(ax, "Train-only ridge accuracy: not measured.", location="top")
    if not present and not refit:
        _missing(ax)
    ax = axes[3]
    _base_axis(ax, "D. Centered hidden Fourier fractions", "Fraction of centered full-grid energy", extent, window, fraction=True)
    hidden = _line(ax, obs, "hidden_AH_add", "fourier_addition", f"Addition A_H (gain {_fmt(_delta(obs, 'hidden_AH_add', window), 3)})")
    hidden |= _line(ax, obs, "hidden_AH_diff", "difference_or_residual", f"Difference fraction (gain {_fmt(_delta(obs, 'hidden_AH_diff', window), 3)})")
    if hidden:
        _legend(ax, loc="upper left")
    else:
        _missing(ax, "Canonical centered hidden Fourier fractions unavailable.")
    ax = axes[4]
    _base_axis(ax, "E. Canonical mapped AGOP–Fourier", "Effective purity (held-out mapping)", extent, window, fraction=True)
    current = _valid_agop_line(ax, obs, "agop_q4_purity", "agop_q4_valid", "agop_current", "Current q=4 axes (valid only)")
    fixed = _valid_agop_line(ax, obs, "agop_q4_fixed_purity", "agop_q4_fixed_valid", "agop_fixed", "Window-start fixed axes (valid only)")
    null = _line(ax, obs, "agop_q4_null_median", "random_or_threshold", "8 fixed random frames: median")
    x, lo = _series(obs, "agop_q4_null_min")
    _, hi = _series(obs, "agop_q4_null_max")
    if len(x) and len(lo) == len(hi):
        ax.fill_between(x, lo, hi, color="#999999", alpha=.18, label="Random-frame range")
    if current or fixed or null:
        _legend(ax, loc="upper left")
        missing = []
        if not current:
            missing.append("valid current axes")
        if not fixed:
            missing.append("fixed axes")
        if not null:
            missing.append("random controls")
        if missing:
            _note(ax, "Unavailable / unresolved: " + ", ".join(missing))
    else:
        _missing(ax, "Canonical q=4 class-centered-logit AGOP was not measured; no legacy metric substituted.")
    ax = axes[5]
    _base_axis(ax, "F. Head-fit risk: train-only ridge", "Squared error per example to centered labels", extent, window)
    risk = _line(ax, obs, "refit_test_mse", "refit_test", "Train-only ridge: held-out MSE")
    risk |= _line(ax, obs, "refit_train_mse", "refit_train", "Train-only ridge: train MSE")
    if risk:
        ax.set_ylim(bottom=0)
        _legend(ax, loc="upper right")
    else:
        _missing(ax, "Train-only ridge refit MSE was not measured. Oracle projection error is not a substitute.")
    return fig


def _mechanism_sheet(trace, obs, score, config, extent, window):
    fig, axes = _make_sheet(config, score, "Mechanism observers — not causal guarantees")
    ax = axes[0]
    _base_axis(ax, "A. Hidden-energy decomposition", "Squared feature energy per grid example", extent, window)
    any_energy = False
    for key, role, label in [
        ("hidden_E_add", "fourier_addition", "Addition energy E+"),
        ("hidden_E_residual", "difference_or_residual", "Non-rule centered energy"),
        ("hidden_E_total", "total_or_neutral", "Total centered energy"),
        ("hidden_E_DC", "random_or_threshold", "Feature mean/DC energy")]:
        any_energy |= _line(ax, obs, key, role, label)
    if any_energy:
        ax.set_ylim(bottom=0)
        _legend(ax, loc="upper right")
    else:
        _missing(ax)
    ax = axes[1]
    _base_axis(ax, "B. Hidden-span availability (oracle)", "Full-grid subspace overlap / oracle error", extent, window, fraction=True)
    available = _line(ax, obs, "availability_A_plus", "fourier_availability", "Hidden-span availability A+")
    available |= _line(ax, obs, "availability_oracle_error", "oracle_projection_error", "Full-grid oracle error 1−A+")
    if available:
        _, rank = _series(obs, "availability_rank")
        rank = rank[np.isfinite(rank)]
        rank_label = f"Hidden numerical rank: {np.min(rank):g}–{np.max(rank):g}" if len(rank) else "Numerical rank: not measured"
        _legend(ax, loc="center left", title=rank_label, title_fontsize=9)
    else:
        _missing(ax)
    ax = axes[2]
    _base_axis(ax, "C. Actual-head signed rule / residual", "Held-out signed logits: median", extent, window)
    margin = False
    for key, role, label in [
        ("signed_test_gamma_q50", "fourier_addition", "Rule margin gamma: median"),
        ("signed_test_beta_q50", "difference_or_residual", "Signed residual beta: median"),
        ("signed_test_bound_q50", "total_or_neutral", "Pointwise gamma−beta: median"),
        ("actual_test_margin_q50", "actual_test", "Actual margin: median")]:
        margin |= _line(ax, obs, key, role, label)
    ax.axhline(0, color="#999999", linewidth=.8)
    if margin:
        _legend(ax, loc="upper left")
    else:
        _missing(ax)
    ax = axes[3]
    _base_axis(ax, "D. Refit margins and head discrepancy", "Held-out centered-logit units", extent, window)
    values = False
    for key, role, label in [
        ("refit_test_margin_q50", "refit_test", "Refit margin: median"),
        ("actual_test_margin_q50", "actual_test", "Actual margin: median"),
        ("tracking_test_discrepancy_q50", "difference_or_residual", "Actual−refit discrepancy: median")]:
        values |= _line(ax, obs, key, role, label)
    for stem, role in [("refit_test_margin", "refit_test"), ("actual_test_margin", "actual_test")]:
        x, low = _series(obs, stem + "_q05")
        _, high = _series(obs, stem + "_q95")
        if len(x) and len(low) == len(high):
            ax.fill_between(x, low, high, color=PALETTE[role][0], alpha=.12)
    ax.axhline(0, color="#999999", linewidth=.8)
    if values:
        _legend(ax, loc="upper left")
    else:
        _missing(ax)
    ax = axes[4]
    _base_axis(ax, "E. Parameter norms", "Frobenius matrix norm", extent, window)
    norms = _line(ax, trace if _has(trace, "W_norm") else obs, "W_norm", "fourier_addition", "Hidden matrix W")
    norms |= _line(ax, trace if _has(trace, "V_norm") else obs, "V_norm", "total_or_neutral", "Actual output head V")
    if norms:
        ax.set_yscale("log")
        _legend(ax, loc="upper right")
    else:
        _missing(ax, "Matrix norms were not recorded.")
    ax = axes[5]
    _base_axis(ax, "F. CE increments and first-order terms", "CE change per update", extent, window)
    increments = False
    for key, role, label in [
        ("ce_increment", "actual_train", "Actual CE(t)−CE(t−1)"),
        ("linear_total", "total_or_neutral", "Gradient inner-product sum"),
        ("linear_gradient_term", "fourier_addition", "Gradient-update contribution"),
        ("linear_decay_term", "difference_or_residual", "Decay contribution"),
        ("linear_remainder", "oracle_projection_error", "Observed change minus linear term")]:
        increments |= _line(ax, trace, key, role, label)
    ax.axhline(0, color="#999999", linewidth=.8)
    if increments:
        _legend(ax, loc="upper right")
    else:
        _missing(ax, "Every-update gradient/update inner products were not recorded. Sparse CE changes are not substituted.")
    return fig


def _companion_sheet(trace, obs, score, config, extent, window):
    fig, axes = _make_sheet(config, score, "Linear objective and controls in their own units")
    ax = axes[0]
    _base_axis(ax, "A. Full-trajectory CE: linear companion", "Original mean CE (linear scale)", extent, window)
    present = _line(ax, trace, "train_ce", "actual_train", "Actual train CE")
    present |= _line(ax, trace, "test_ce", "actual_test", "Actual held-out CE")
    if present:
        ax.set_ylim(bottom=0)
        ax.axhline(np.log(float(config['p'])), **_style("random_or_threshold", label="Uniform logits: log(p)"))
        _legend(ax, loc="upper right")
    else:
        _missing(ax)
    ax = axes[1]
    _base_axis(ax, "B. AGOP rank-aware companion", "Explicitly distinct overlap normalizations", extent, window, fraction=True)
    present = _line(ax, obs, "agop_q4_coverage", "agop_current", "Coverage = overlap / target rank")
    present |= _line(ax, obs, "agop_q4_requested_rank_score", "agop_fixed", "Requested score = overlap / q")
    if present:
        _legend(ax, loc="upper left")
    else:
        _missing(ax)
    ax = axes[2]
    _base_axis(ax, "C. Centered logit scale", "Class-centered actual-logit RMS", extent, window)
    present = _line(ax, trace, "test_logit_centered_rms", "actual_test", "Actual held-out centered-logit RMS")
    if not present:
        present = _line(ax, trace, "logit_rms", "total_or_neutral", "Actual logit RMS (recorded row set)")
    if not present:
        present = _line(ax, obs, "actual_test_logit_centered_rms", "actual_test", "Actual held-out centered-logit RMS")
    if present:
        ax.set_ylim(bottom=0)
        _legend(ax, loc="upper right")
    else:
        _missing(ax)
    ax = axes[3]
    gate_interval = config.get("observer_interval_updates", "declared interval")
    _base_axis(ax, "D. ReLU gate changes", f"Gate-change fraction ({gate_interval} updates)" if isinstance(gate_interval, (int, float)) else "Gate-change fraction (declared interval)", extent, window, fraction=True)
    present = _line(ax, obs, "gate_change_fraction", "difference_or_residual", "Hidden ReLU gate-change fraction")
    if not present:
        present = _line(ax, trace, "gate_change_fraction", "difference_or_residual", "Hidden ReLU gate-change fraction")
    if present:
        _legend(ax, loc="upper left")
    else:
        _missing(ax, "Gate changes were not recorded; fixed gates are not assumed.")
    ax = axes[4]
    _base_axis(ax, "E. Normalized feature displacement", "Declared normalized feature displacement", extent, window)
    present = _line(ax, obs, "normalized_representation_displacement_previous", "fourier_addition", "From previous observer checkpoint")
    present |= _line(ax, obs, "normalized_representation_displacement_initial", "total_or_neutral", "From initialization (normalized features)")
    if not present:
        present = _line(ax, obs, "representation_displacement", "fourier_addition", "Normalized hidden-representation motion")
    if not present:
        present = _line(ax, trace, "representation_displacement", "fourier_addition", "Normalized hidden-representation motion")
    if present:
        ax.set_ylim(bottom=0)
        _legend(ax, loc="upper left")
    else:
        _missing(ax, "Normalized representation displacement was not recorded; parameter norms do not substitute for it.")
    ax = axes[5]
    _base_axis(ax, "F. Pointwise prediction certificates", "Fraction of held-out examples certified", extent, window, fraction=True)
    present = _line(ax, obs, "signed_test_certified_fraction", "fourier_addition", "Signed rule−residual bound > 0")
    present |= _line(ax, obs, "tracking_test_certified_fraction", "refit_test", "Refit margin−2 discrepancy > 0")
    present |= _line(ax, trace, "test_acc", "actual_test", "Actual held-out accuracy (not a bound)")
    if present:
        _legend(ax, loc="lower right")
        if not _has(obs, "tracking_test_certified_fraction"):
            _note(ax, "Head-transfer certificate: not measured.", location="top")
    else:
        _missing(ax)
    return fig


def plot_run(trace, observations, score, config, outdir, stem="run"):
    trace, obs = _columns(trace), _columns(observations)
    for target, source in (("train_acc", "train_accuracy"), ("test_acc", "test_accuracy")):
        if target not in trace and source in trace:
            trace[target] = trace[source]
    if "step" not in trace or len(trace["step"]) == 0:
        raise ValueError("A nonempty global-step training trace is required")
    if float(config.get("p", 0)) <= 1:
        raise ValueError("A modulus p > 1 is required")
    if Path(stem).name != stem or stem in ("", ".", ".."):
        raise ValueError("stem must be a plain output basename")
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    for sheet in ("trajectories", "mechanism", "controls"):
        for suffix in ("png", "svg", "pdf"):
            if (outdir / f"{stem}_{sheet}.{suffix}").exists():
                raise FileExistsError(f"Refusing to overwrite {stem}_{sheet}.{suffix}")
    extent = (float(trace["step"][0]), max(float(trace["step"][-1]), float(trace["step"][0]) + 1))
    window = _window(score)
    rc = {"font.size": 9, "axes.labelsize": 9, "legend.fontsize": 9,
          "xtick.labelsize": 9, "ytick.labelsize": 9, "savefig.dpi": 200,
          "svg.fonttype": "none", "pdf.fonttype": 42, "figure.facecolor": "white"}
    output = {"protocol_version": "1.0-proposed", "unsmoothed": True,
              "window": list(window) if window else None,
              "style": "plot_style_v1.json semantic palette", "files": {}}
    with plt.rc_context(rc):
        for name, builder in [("trajectories", _trajectory_sheet), ("mechanism", _mechanism_sheet),
                              ("controls", _companion_sheet)]:
            fig = builder(trace, obs, score, config, extent, window)
            output["files"][name] = _save(fig, outdir, f"{stem}_{name}", _footer(config, score, name, obs))
    return output
