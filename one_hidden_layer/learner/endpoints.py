from __future__ import annotations

import math
from typing import Any, Mapping

import numpy as np


PROTOCOL = "1.0-proposed/CE-endpoints-v1"
FLAGS = (
    "P10_raw", "P03_raw", "P01_raw", "fit_strict", "F02", "F05", "F10", "F15",
    "AGOP02", "useful_refit", "finite_recovery90", "finite_recovery95",
    "finite_recovery99", "finite_recovery100", "bad_test_delay",
    "delayed_acquisition", "PA10_F02", "joint_PAG10_F02",
    "AGOP_supported_joint", "representation_first", "full_transition_covered",
    "late_overlap", "recovery_after_collapse", "persistence90_three",
)


def _and(*values):
    if any(v is False for v in values):
        return False
    return None if any(v is None for v in values) else True


def _series(data, key, steps_key):
    if key not in data or data[key] is None:
        return None
    if steps_key not in data or data[steps_key] is None:
        raise ValueError(f"{key} requires {steps_key}")
    s = np.asarray(data[steps_key])
    v = np.asarray(data[key])
    if s.ndim != 1 or v.ndim != 1 or s.shape != v.shape:
        raise ValueError(f"{key} must be 1D and match {steps_key}")
    if not np.all(np.isfinite(s)) or np.any(s < 0) or np.any(s != np.floor(s)):
        raise ValueError(f"{steps_key} must contain nonnegative integer times")
    if np.any(np.diff(s) <= 0):
        raise ValueError(f"{steps_key} must be strictly increasing")
    if key.endswith("_resolved") and not np.all(np.isin(v, [True, False])):
        raise ValueError(f"{key} must contain only explicit boolean values")
    return s.astype(np.int64), v


def _interval(series, t0, t1, dense=False):
    if series is None:
        return None
    s, v = series
    take = (s >= t0) & (s <= t1)
    a, b = s[take], v[take]
    if len(a) == 0 or a[0] != t0 or a[-1] != t1:
        return None
    if dense and not np.array_equal(a, np.arange(t0, t1 + 1)):
        return None
    return a, b


def _at(series, t):
    if series is None:
        return None
    s, v = series
    i = int(np.searchsorted(s, t))
    if i == len(s) or s[i] != t:
        return None
    return v[i].item()


def _valid_accuracy(series):
    return series is not None and np.all(np.isfinite(series[1])) and bool(
        np.all((series[1] >= 0) & (series[1] <= 1)))


def _crossing(series, threshold, after=None, inclusive=True):
    if not _valid_accuracy(series):
        return None
    s, v = series
    mask = v >= threshold
    if after is not None:
        mask &= s >= after if inclusive else s > after
    ix = np.flatnonzero(mask)
    if not len(ix):
        return None
    i = int(ix[0])
    return {"time": int(s[i]), "bracket": [int(s[max(0, i - 1)]), int(s[i])],
            "accuracy": float(v[i]), "index": i}


def _fit_state(data, interval, margin_tolerance):
    if interval is None:
        return None
    s, v = interval
    resolved = _series(data, "train_margin_resolved", "train_step")
    if not np.all(np.isfinite(v)):
        return None
    if np.any(v < -margin_tolerance):
        return False
    if resolved is not None:
        r = _interval(resolved, int(s[0]), int(s[-1]), dense=True)
        if r is None or not np.all(r[1].astype(bool)):
            return None
    if np.any(v <= margin_tolerance):
        return None
    return True


def _ce_stats(data, interval):
    if interval is None:
        return None
    s, v = interval
    resolved = _series(data, "train_ce_resolved", "train_step")
    if resolved is not None:
        r = _interval(resolved, int(s[0]), int(s[-1]), dense=True)
        if r is None or not np.all(r[1].astype(bool)):
            return None
    dtype = v.dtype if np.issubdtype(v.dtype, np.floating) else np.dtype("float64")
    if not np.all(np.isfinite(v)) or np.any(v < np.finfo(dtype).tiny):
        return None
    lo, avg, hi = float(np.min(v)), float(np.mean(v)), float(np.max(v))
    centered_steps = s - np.mean(s)
    slope = float(np.dot(centered_steps, v - avg) / np.dot(centered_steps, centered_steps))
    return {"CE_min": lo, "CE_mean": avg, "CE_max": hi,
            "v_min": (hi - lo) / lo, "v_mean": (hi - lo) / avg,
            "signed_endpoint_drift": float(v[-1] - v[0]),
            "signed_endpoint_drift_relative": float((v[-1] - v[0]) / v[0]),
            "maximum_drawdown_absolute": float(np.max(np.maximum.accumulate(v) - v)),
            "fitted_slope_per_update": slope}


def _window(data, cfg, tfit):
    b, grid = int(cfg["budget"]), int(cfg.get("observer_grid", 500))
    min_length = max(1000, math.ceil(.05 * b))
    rule = cfg.get("window_rule", "fixed")
    if grid <= 0 or b <= 0:
        raise ValueError("budget and observer_grid must be positive")
    if rule == "fixed":
        t0, t1 = int(cfg["t0"]), int(cfg["t1"])
        if t0 < 0 or t1 > b or t1 - t0 < min_length or t0 % grid or t1 % grid:
            raise ValueError("fixed window violates horizon, minimum length, or observer grid")
        return (t0, t1), "fixed", None
    if rule != "training_only":
        raise ValueError("window_rule must be fixed or training_only")
    w = int(cfg["window_length"])
    if w < min_length or w % grid or w > b:
        raise ValueError("training_only length violates minimum, grid, or horizon")
    if tfit is None:
        margins = _interval(_series(data, "train_min_margin", "train_step"), 0, b, True)
        if margins is not None and np.all(np.isfinite(margins[1])) and np.all(
                margins[1] < -cfg.get("margin_tolerance", 1e-10)):
            return None, "no_qualifying_window", "training never fitted during complete declared horizon"
        return None, "unresolved", "no resolved first training fit"
    ce = _series(data, "train_ce", "train_step")
    margins = _series(data, "train_min_margin", "train_step")
    for t0 in range(math.ceil(tfit / grid) * grid, b - w + 1, grid):
        fit = _fit_state(data, _interval(margins, t0, t0 + w, dense=True),
                         cfg.get("margin_tolerance", 1e-10))
        stats = _ce_stats(data, _interval(ce, t0, t0 + w, dense=True))
        if fit is False or (stats is not None and stats["v_min"] > .10):
            continue
        if fit is None or stats is None:
            return None, "unresolved", "earlier candidate window is censored or numerically unresolved"
        return (t0, t0 + w), "earliest_training_only", None
    return None, "no_qualifying_window", "no training-only qualifying window in declared horizon"


def _json_safe(value):
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _finalize(out):
    if not out["numerical_health_ok"]:
        out["reasons"]["numerical_health"] = "declared learner failure or nonfinite original learner metrics"
        for key in ("PA10_F02", "joint_PAG10_F02", "AGOP_supported_joint", "representation_first",
                    "full_transition_covered"):
            if out[key] is True:
                out[key] = None
    for field in FLAGS:
        if out[field] is None and field not in out["reasons"]:
            out["reasons"][field] = "required event or observer unavailable / unresolved"
    return _json_safe(out)


def score_run(data: Mapping[str, Any], config: Mapping[str, Any]) -> dict:
    cfg = dict(config)
    b, p = int(cfg["budget"]), int(cfg["p"])
    if b <= 0 or p < 7:
        raise ValueError("this CE protocol requires budget > 0 and p >= 7")
    for name in ("train_step", "eval_step", "obs_step", "agop_step", "refit_step"):
        if name in data and data[name] is not None and np.any(np.asarray(data[name]) > b):
            raise ValueError(f"{name} includes states beyond declared budget; do not extend endpoint horizon silently")
    out = {k: None for k in FLAGS}
    out.update(protocol_version=PROTOCOL, window_rule=cfg.get("window_rule", "fixed"),
               t0=None, t1=None, T_fit=None, T_grok=None, T_grok_bracket=None,
               T_refit=None, T_refit_bracket=None, T20=None, T20_bracket=None,
               terminal_test_acc=None, P10_cycle=None,
               period_two_status="not_applicable_to_primary_ReLU_CE",
               reasons={}, missing_observers=[])
    reasons = out["reasons"]
    margin_tol = float(cfg.get("margin_tolerance", 1e-10))
    train = _series(data, "train_ce", "train_step")
    margins = _series(data, "train_min_margin", "train_step")
    test = _series(data, "test_acc", "eval_step")
    out["numerical_health_ok"] = bool(not data.get("numerical_failure", False) and all(
        series is None or np.all(np.isfinite(series[1])) for series in (train, margins, test)))
    if margins is not None and len(margins[0]) and margins[0][0] == 0:
        s, m = margins
        valid = np.isfinite(m) & (m > margin_tol)
        resolved = _series(data, "train_margin_resolved", "train_step")
        if resolved is not None:
            valid &= resolved[1].astype(bool)
        hits = np.flatnonzero(valid)
        if len(hits):
            out["T_fit"] = int(s[hits[0]])
    tfit = out["T_fit"]
    if tfit is None:
        reasons["T_fit"] = "first resolved training fit unavailable (initial state required)"
        whole_margin = _interval(margins, 0, b, True)
        if whole_margin is not None and np.all(np.isfinite(whole_margin[1])) and np.all(
                whole_margin[1] < -margin_tol):
            out["bad_test_delay"] = False
            out["delayed_acquisition"] = False
    if _valid_accuracy(test) and len(test[0]):
        out["terminal_test_acc"] = float(test[1][-1])
        for threshold, suffix in ((.90, "90"), (.95, "95"), (.99, "99"), (1., "100")):
            event = _crossing(test, threshold)
            out["finite_recovery" + suffix] = event is not None if test[0][-1] >= b else (True if event else None)
            out["T" + suffix] = None if event is None else event["time"]
        first90 = _crossing(test, .90)
        high = test[1] >= .90
        streaks = np.flatnonzero(high[:-2] & high[1:-1] & high[2:])
        if len(streaks):
            out["persistence90_three"] = True
            out["T90_persistent3_start"] = int(test[0][streaks[0]])
            out["T90_persistent3_confirmed"] = int(test[0][streaks[0] + 2])
        else:
            out["persistence90_three"] = False if test[0][-1] >= b else None
        if first90:
            laterlow = np.flatnonzero((test[0] > first90["time"]) & (test[1] < .90))
            out["recovery_after_collapse"] = bool(len(laterlow) and np.any(
                (test[0] > test[0][laterlow[0]]) & (test[1] >= .90)))
        else:
            out["recovery_after_collapse"] = False if test[0][-1] >= b else None
    else:
        first90 = None
        reasons["finite_recovery90"] = "actual held-out accuracy missing or invalid"
        out["missing_observers"].append("actual_test_accuracy")

    delay = max(1000, math.ceil(.02 * b))
    out["required_bad_test_delay"] = delay
    if tfit is not None and _valid_accuracy(test) and len(test[0]):
        s, a = test
        prefix = (s >= tfit) & (s <= tfit + delay)
        if np.any(a[prefix] > .20):
            out["bad_test_delay"] = False
        elif not np.any(prefix) or s[0] != 0 or s[-1] < tfit + delay:
            reasons["bad_test_delay"] = "bad-test prefix lacks temporal coverage"
        else:
            delay_fit = _fit_state(data, _interval(margins, tfit, tfit + delay, True), margin_tol)
            out["bad_test_delay"] = delay_fit
        earlier90 = bool(np.any((s <= tfit + delay) & (a >= .90)))
        if earlier90:
            out["delayed_acquisition"] = False
        elif first90:
            out["T_grok"], out["T_grok_bracket"] = first90["time"], first90["bracket"]
            persistence_fit = _fit_state(data, _interval(margins, tfit, first90["time"], True), margin_tol)
            out["delayed_acquisition"] = _and(out["bad_test_delay"], persistence_fit)
        else:
            out["delayed_acquisition"] = False if s[-1] >= b else None
        postdelay20 = np.flatnonzero((s > tfit + delay) & (a > .20))
        if len(postdelay20):
            i = int(postdelay20[0])
            out["T20"], out["T20_bracket"] = int(s[i]), [int(s[max(0, i - 1)]), int(s[i])]
        if out["T_grok"] is not None:
            out["fit_to_grok_delay_updates"] = out["T_grok"] - tfit
            if out["T20"] is not None:
                out["transition_duration_updates"] = out["T_grok"] - out["T20"]
                out["transition_to_required_bad_delay_ratio"] = out["transition_duration_updates"] / delay

    window, selection, issue = _window(data, cfg, tfit)
    out["window_selection_status"] = selection
    if issue:
        reasons["window"] = issue
    if window is None:
        if selection == "no_qualifying_window":
            for field in ("PA10_F02", "joint_PAG10_F02", "AGOP_supported_joint", "representation_first",
                          "full_transition_covered"):
                out[field] = False
        return _finalize(out)
    t0, t1 = window
    out["t0"], out["t1"] = t0, t1
    stats = _ce_stats(data, _interval(train, t0, t1, True))
    if stats is None:
        reasons["plateau"] = "requires every-update, finite, resolved, strictly positive original CE"
    else:
        out.update(stats)
        out["CE_over_logp"] = stats["CE_mean"] / math.log(p)
        for field, threshold in (("P10_raw", .10), ("P03_raw", .03), ("P01_raw", .01)):
            out[field] = stats["v_min"] <= threshold
    mi = _interval(margins, t0, t1, True)
    out["fit_strict"] = _fit_state(data, mi, margin_tol)
    out["min_train_margin"] = float(np.min(mi[1])) if mi is not None and np.all(np.isfinite(mi[1])) else None
    train_acc = _series(data, "train_acc", "train_step")
    ta = _interval(train_acc, t0, t1, True)
    out["min_train_acc"] = float(np.min(ta[1])) if ta is not None and _valid_accuracy(ta) else None
    ta0, ta1 = _at(train_acc, t0), _at(train_acc, t1)
    out["train_acc_start_end"] = [ta0, ta1]
    out["fit_accuracy99_endpoints"] = (bool(min(ta0, ta1) >= .99)
                                        if ta0 is not None and ta1 is not None
                                        and np.isfinite(ta0) and np.isfinite(ta1) else None)
    if out["fit_strict"] is None:
        reasons["fit_strict"] = "requires resolved strictly positive margins at every window update"

    ah = _series(data, "AH", "obs_step")
    ea = _series(data, "Eadd", "obs_step")
    et = _series(data, "Etotal", "obs_step")
    ar = _series(data, "AH_resolved", "obs_step")
    energy_tolerance = float(cfg.get("energy_tolerance", 1e-20))
    vals = [_at(series, t) for series in (ah, ea, et) for t in window]
    energy_ok = all(v is not None and np.isfinite(v) for v in vals)
    if energy_ok:
        a0, a1, e0, e1, d0, d1 = vals
        energy_ok = (0 <= a0 <= 1 and 0 <= a1 <= 1 and 0 <= e0 <= d0 and 0 <= e1 <= d1
                     and d0 > energy_tolerance and d1 > energy_tolerance)
        if energy_ok:
            energy_ok = bool(np.isclose(a0, e0 / d0, rtol=1e-8, atol=1e-10)
                             and np.isclose(a1, e1 / d1, rtol=1e-8, atol=1e-10))
    if ar is not None:
        energy_ok = energy_ok and all(_at(ar, t) is True for t in window)
    if energy_ok:
        out.update(AH_start=a0, AH_end=a1, delta_AH=a1 - a0,
                   Eadd_start=e0, Eadd_end=e1, Etotal_start=d0, Etotal_end=d1)
        for field, threshold in (("F02", .02), ("F05", .05), ("F10", .10), ("F15", .15)):
            out[field] = (a1 - a0) >= threshold
    else:
        reasons["Fourier"] = "requires exact endpoint AH and resolved nonzero denominator energies"
        out["missing_observers"].append("resolved_hidden_Fourier_endpoint_energies")

    q = int(cfg.get("q_requested", 4))
    if q <= 0:
        raise ValueError("q_requested must be positive")
    out["q_requested"] = q
    ak = ("agop_purity", "agop_selected_rank", "agop_mapped_rank", "agop_target_rank",
          "agop_gap_ratio", "agop_lambda_max")
    av = {k: [_at(_series(data, k, "agop_step"), t) for t in window] for k in ak}
    agop_ok = all(v is not None and np.isfinite(v) for vs in av.values() for v in vs)
    if agop_ok:
        agop_ok = (all(0 <= v <= 1 for v in av["agop_purity"])
                   and av["agop_selected_rank"] == [q, q]
                   and av["agop_mapped_rank"] == [q, q]
                   and av["agop_target_rank"][0] == av["agop_target_rank"][1]
                   and av["agop_target_rank"][0] >= q
                   and min(av["agop_gap_ratio"]) > 1e-8
                   and min(av["agop_lambda_max"]) > 0)
    out["ranks_start_end"] = {k: av[k] for k in ak if "rank" in k}
    if agop_ok:
        out["delta_AGOP"] = av["agop_purity"][1] - av["agop_purity"][0]
        out["AGOP02"] = out["delta_AGOP"] >= .02
    else:
        reasons["AGOP02"] = "missing, changing, deficient, or boundary-gap-unresolved fixed-q AGOP endpoints"
        out["missing_observers"].append("resolved_canonical_AGOP_gain")
    for key in ("agop_requested_rank_score", "agop_coverage", "agop_null_median", "agop_fixed_purity"):
        vals = [_at(_series(data, key, "agop_step"), t) for t in window]
        out[key + "_start_end"] = vals
        if all(v is not None and np.isfinite(v) for v in vals):
            out["delta_" + key] = vals[1] - vals[0]
    if "delta_agop_null_median" in out:
        out["delta_null"] = out["delta_agop_null_median"]
        if agop_ok:
            out["delta_AGOP_minus_delta_null"] = out["delta_AGOP"] - out["delta_null"]
            out["AGOP_end_minus_null"] = av["agop_purity"][1] - out["agop_null_median_start_end"][1]

    refit = _series(data, "refit_acc", "refit_step")
    r0, r1 = _at(refit, t0), _at(refit, t1)
    out["refit_acc_start"], out["refit_acc_end"] = r0, r1
    for name, t in (("start", t0), ("end", t1)):
        out["refit_MSE_" + name] = _at(_series(data, "refit_mse", "refit_step"), t)
    refit_event = None
    if _valid_accuracy(refit) and r0 is not None and r1 is not None:
        refit_event = _crossing(refit, .90, t0, inclusive=False)
        if refit_event is not None and refit_event["time"] > t1:
            refit_event = None
        out["useful_refit"] = bool(r0 <= .70 and refit_event is not None
                                   and refit_event["accuracy"] - r0 >= .20)
        if refit_event:
            out["T_refit"], out["T_refit_bracket"] = refit_event["time"], refit_event["bracket"]
    else:
        reasons["useful_refit"] = "specified refit or exact common-window endpoints missing / invalid"
        out["missing_observers"].append("train_only_refit")

    a0, a1 = _at(test, t0), _at(test, t1)
    out["test_acc_start_end"] = [a0, a1]
    out["PA10_F02"] = _and(out["P10_raw"], out["F02"])
    ingrok = (t0 <= out["T_grok"] <= t1) if out["T_grok"] is not None else (
        False if out["delayed_acquisition"] is False else None)
    startbad = a0 <= .20 if a0 is not None and np.isfinite(a0) else None
    persistent_fit = (_fit_state(data, _interval(margins, tfit, t1, True), margin_tol)
                      if tfit is not None and tfit <= t0 else False if tfit is not None else None)
    out["fit_persists_from_first_fit"] = persistent_fit
    out["joint_PAG10_F02"] = _and(out["PA10_F02"], out["fit_strict"], persistent_fit,
                                  out["delayed_acquisition"], ingrok, startbad)
    out["AGOP_supported_joint"] = _and(out["joint_PAG10_F02"], out["AGOP02"])
    in20 = t0 <= out["T20"] <= t1 if out["T20"] is not None else None
    out["full_transition_covered"] = _and(out["joint_PAG10_F02"], in20)
    if out["T20"] is not None and out["T_grok"] is not None:
        overlaps = max(t0, out["T20"]) <= min(t1, out["T_grok"])
        out["late_overlap"] = bool(overlaps and not (in20 and ingrok))
    chronology = None
    if refit_event is not None and out["T_grok"] is not None:
        atrefit = _at(test, refit_event["time"])
        if atrefit is not None and np.isfinite(atrefit):
            if atrefit > .20 or not (t0 < refit_event["time"] < out["T_grok"] <= t1):
                chronology = False
            elif refit_event["bracket"][1] < out["T_grok_bracket"][0]:
                chronology = True
            else:
                reasons["representation_first"] = "refit / actual crossing brackets cannot be strictly ordered"
        else:
            reasons["representation_first"] = "actual accuracy not logged at refit crossing"
    out["representation_first"] = _and(out["joint_PAG10_F02"], out["useful_refit"], chronology)
    return _finalize(out)
