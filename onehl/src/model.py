from __future__ import annotations

import hashlib
import json
import math

import numpy as np

import legacy_core

VERSION = "1hl-fixed-polar-ngd-1.0-20260927"


def config_hash(config):
    return hashlib.sha256(json.dumps(config, sort_keys=True, separators=(",", ":"),
                                      allow_nan=False).encode()).hexdigest()


def validated_config(config):
    result = dict(config)
    defaults = dict(width=512, c=24.0, train_fraction=.8, eta_W=.03, eta_V=.03,
                    decay_W=.005, decay_V=.2, margin_tolerance=1e-10)
    for key, value in defaults.items():
        result.setdefault(key, value)
    for key in ("p", "width"):
        value = result[key]
        if isinstance(value, bool) or int(value) != value or value < (3 if key == "p" else 1):
            raise ValueError(f"invalid {key}")
        result[key] = int(value)
    for key in ("c", "train_fraction", "eta_W", "eta_V", "decay_W", "decay_V", "margin_tolerance"):
        value = float(result[key])
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"invalid {key}")
        result[key] = value
    if result["c"] <= 0 or not 0 < result["train_fraction"] < 1:
        raise ValueError("positive c and a proper split are required")
    n = int(result["train_fraction"] * result["p"] ** 2)
    if n < 1 or n >= result["p"] ** 2:
        raise ValueError("empty training or held-out split")
    cutoff = dict(result.get("polar_cutoff", {"mode": "relative", "value": 1e-10}))
    if cutoff.get("mode") not in ("relative", "legacy"):
        raise ValueError("polar_cutoff.mode must be relative or legacy")
    if cutoff["mode"] == "relative":
        if not 0 < float(cutoff.get("value", 0)) < 1:
            raise ValueError("relative cutoff must be in (0,1)")
        cutoff["value"] = float(cutoff["value"])
    result["polar_cutoff"] = cutoff
    return result


def initialize(config, init_seed, split_seed):
    config = validated_config(config)
    p, width, c = config["p"], config["width"], config["c"]
    rng = np.random.default_rng(init_seed)
    w = c * rng.normal(size=(width, 2 * p)) / math.sqrt(p)
    v = rng.normal(size=(p, width)) / (c * math.sqrt(width))
    permutation = np.random.default_rng(split_seed).permutation(p * p)
    n = int(config["train_fraction"] * p * p)
    tr, te = np.sort(permutation[:n]), np.sort(permutation[n:])
    first = np.repeat(np.arange(p), p)
    second = np.tile(np.arange(p), p)
    x = np.concatenate([np.eye(p)[first], np.eye(p)[second]], 1)
    y = (first + second) % p
    return x, y, tr, te, w, v


def polar_direction(force, cutoff):
    if not np.all(np.isfinite(force)):
        raise ArithmeticError("nonfinite feature force")
    scale = float(np.abs(force).max())
    if scale == 0:
        return np.zeros_like(force), dict(polar_rank=0, polar_threshold=0.0,
                                         polar_sigma_max=0.0, polar_relative_threshold=0.0)
    u, s, vh = np.linalg.svd(force / scale, full_matrices=False)
    rtol = (np.finfo(np.float64).eps * max(force.shape)
            if cutoff["mode"] == "legacy" else cutoff["value"])
    threshold = rtol * s[0]
    keep = s > threshold
    rank = int(keep.sum())
    direction = ((u[:, keep] @ vh[keep]) / math.sqrt(rank)
                 if rank else np.zeros_like(force))
    return direction, dict(polar_rank=rank, polar_threshold=float(threshold * scale),
                          polar_sigma_max=float(s[0] * scale),
                          polar_relative_threshold=float(rtol))


def update(w, v, force_w, force_v, config):
    q, stats = polar_direction(force_w, config["polar_cutoff"])
    z = legacy_core.head_direction(force_v)
    if not np.all(np.isfinite(force_v)):
        raise ArithmeticError("nonfinite head force")
    wg = config["eta_W"] * q
    wd = -config["eta_W"] * config["decay_W"] * w
    vg = config["eta_V"] * z
    vd = -config["eta_V"] * config["decay_V"] * v
    nw = w + config["eta_W"] * (q - config["decay_W"] * w)
    nv = v + config["eta_V"] * (z - config["decay_V"] * v)
    dw, dv = nw - w, nv - v
    linear_gradient = -float(np.sum(force_w * wg) + np.sum(force_v * vg))
    linear_decay = -float(np.sum(force_w * wd) + np.sum(force_v * vd))
    linear_actual = -float(np.sum(force_w * dw) + np.sum(force_v * dv))
    stats.update(grad_W_norm=float(np.linalg.norm(force_w)), grad_V_norm=float(np.linalg.norm(force_v)),
                 update_W_norm=float(np.linalg.norm(dw)), update_V_norm=float(np.linalg.norm(dv)),
                 gradient_update_W_norm=float(np.linalg.norm(wg)), gradient_update_V_norm=float(np.linalg.norm(vg)),
                 decay_update_W_norm=float(np.linalg.norm(wd)), decay_update_V_norm=float(np.linalg.norm(vd)),
                 ce_linear_gradient=linear_gradient, ce_linear_decay=linear_decay,
                 ce_linear_total=linear_actual,
                 ce_linear_rounding=linear_actual - linear_gradient - linear_decay)
    if not np.all(np.isfinite(nw)) or not np.all(np.isfinite(nv)):
        raise ArithmeticError("nonfinite new parameters")
    return nw, nv, stats


def forward(x, y, tr, te, w, v, margin_tolerance=1e-10):
    values, pre, h, f, residual = legacy_core.forward(x, y, tr, te, w, v)
    if not all(np.all(np.isfinite(a)) for a in (pre, h, f, residual)):
        raise ArithmeticError("nonfinite network state")
    competitors = f.copy()
    competitors[np.arange(len(y)), y] = -np.inf
    margins = f[np.arange(len(y)), y] - competitors.max(axis=1)
    centered = f - f.mean(axis=1, keepdims=True)
    for name, rows in (("train", tr), ("test", te)):
        vals = margins[rows]
        resolution = np.maximum(margin_tolerance, 64 * np.finfo(np.float64).eps *
                                np.maximum(1., np.max(np.abs(centered[rows]), axis=1)))
        values[name + "_accuracy"] = float(np.mean(vals > 0))
        values[name + "_argmax_accuracy"] = float(np.mean(np.argmax(f[rows], axis=1) == y[rows]))
        values[name + "_resolved_accuracy"] = float(np.mean(vals > resolution))
        values[name + "_margin_resolved"] = bool(np.all(vals > resolution))
        values[name + "_near_tie_fraction"] = float(np.mean(np.abs(vals) <= resolution))
        values[name + "_margin_resolution_max"] = float(np.max(resolution))
        for quantile, value in zip(("q05", "q50", "q95"), np.quantile(vals, [.05, .5, .95])):
            values[name + "_margin_" + quantile] = float(value)
        values[name + "_logit_centered_rms"] = float(np.sqrt(np.mean(centered[rows] ** 2)))
        values[name + "_ce_positive_resolved"] = bool(values[name + "_ce"] > 0)
    values["train_ce_over_logp"] = values["train_ce"] / math.log(v.shape[0])
    values["train_ce_below_logp"] = math.log(v.shape[0]) - values["train_ce"]
    values["W_norm"] = float(np.linalg.norm(w))
    values["V_norm"] = float(np.linalg.norm(v))
    return values, pre, h, f, residual


def forces(x, pre, h, residual, v):
    return legacy_core.forces(x, pre, h, residual, v)
