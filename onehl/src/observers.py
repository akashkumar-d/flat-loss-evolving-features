from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy import linalg, special

VERSION = "1hl-observers-1.0-20260927"
EPS = np.finfo(np.float64).eps
DEFAULT_RANK_RTOL = 1e-10
DEFAULT_GAP_RTOL = 1e-8
MARGIN_ATOL = 1e-10


def _finite_matrix(value, name):
    value = np.asarray(value, dtype=np.float64)
    if value.ndim != 2 or not np.all(np.isfinite(value)):
        raise ValueError(f"{name} must be a finite two-dimensional array")
    return value


def _indices(value, total, name, allow_empty=False):
    raw = np.asarray(value)
    if raw.ndim != 1 or raw.dtype.kind not in "iu":
        raise ValueError(f"{name} must be a one-dimensional integer array")
    value = raw.astype(np.int64, copy=False)
    if (not allow_empty and not len(value)) or len(np.unique(value)) != len(value):
        raise ValueError(f"{name} must be nonempty and contain distinct rows")
    if len(value) and (np.min(value) < 0 or np.max(value) >= total):
        raise ValueError(f"{name} contains out-of-range rows")
    return value


def _labels(value, n, p):
    value = np.asarray(value)
    if value.shape != (n,) or value.dtype.kind not in "iu":
        raise ValueError("labels must be a length-n integer array")
    if np.any((value < 0) | (value >= p)):
        raise ValueError("label out of range")
    return value.astype(np.int64, copy=False)


def grid_labels(p):
    if isinstance(p, bool) or int(p) != p or p < 3:
        raise ValueError("p must be an integer at least 3")
    a, b = np.indices((int(p), int(p)))
    return ((a + b) % p).ravel(), ((a - b) % p).ravel()


def addition_basis(p):
    y, _ = grid_labels(p)
    columns = []
    for k in range(1, (p + 1) // 2):
        angle = (2 * np.pi * k / p) * y
        columns.extend((np.sqrt(2) / p * np.cos(angle),
                        np.sqrt(2) / p * np.sin(angle)))
    if p % 2 == 0:
        columns.append(np.cos(np.pi * y) / p)
    return np.column_stack(columns)


def _center(value):
    return value - value.mean(axis=0, keepdims=True)


def _unit_interval(value, name):
    if not np.isfinite(value) or value < -1e-9 or value > 1 + 1e-9:
        raise ArithmeticError(f"{name} outside [0,1]: {value}")
    return float(np.clip(value, 0, 1))


def _nonnegative(value, scale, name):
    tolerance = 256 * EPS * max(float(scale), np.finfo(float).tiny)
    if not np.isfinite(value) or value < -tolerance:
        raise ArithmeticError(f"invalid negative/nonfinite {name}: {value}")
    return float(max(0, value))


def _class_means(value, classes, p):
    order = np.argsort(classes, kind="stable")
    return value[order].reshape(p, p, value.shape[1]).mean(axis=1)


def hidden_fourier(H, p):
    H = _finite_matrix(H, "H")
    if H.shape[0] != p * p:
        raise ValueError("H must contain all p^2 grid rows")
    addition, difference = grid_labels(p)
    mean = H.mean(axis=0)
    Hc = H - mean
    total = float(np.sum(Hc * Hc) / len(H))
    raw = float(np.sum(H * H) / len(H))
    dc = float(mean @ mean)
    add = float(np.sum(_class_means(Hc, addition, p) ** 2) / p)
    diff = float(np.sum(_class_means(Hc, difference, p) ** 2) / p)
    if not all(np.isfinite(v) for v in (total, raw, dc, add, diff)):
        raise ArithmeticError("feature energy overflow")
    residual = _nonnegative(total - add, total, "non-rule energy")
    _nonnegative(total - diff, total, "non-difference energy")
    floor = (64 * EPS) ** 2 * raw
    valid = bool(total > max(floor, np.finfo(float).tiny))
    return {"hidden_AH_add": _unit_interval(add / total, "AH") if valid else None,
            "hidden_AH_diff": _unit_interval(diff / total, "difference fraction") if valid else None,
            "hidden_E_add": add, "hidden_E_diff": diff,
            "hidden_E_total": total, "hidden_E_residual": residual,
            "hidden_E_DC": dc, "hidden_energy_valid": valid,
            "hidden_energy_resolution_floor": floor}


def fourier_spectrum(H, p):
    H = _finite_matrix(H, "H")
    if len(H) != p * p:
        raise ValueError("H must contain all p^2 grid rows")
    coefficients = np.fft.fft2(H.reshape(p, p, -1), axes=(0, 1), norm="ortho")
    energy = np.sum(np.abs(coefficients) ** 2, axis=2) / (p * p)
    k = np.arange(1, p)
    return {"addition_frequency_energy": energy[k, k],
            "difference_frequency_energy": energy[k, (-k) % p],
            "all_frequency_energy": energy}


def _orth(value, rank_rtol=DEFAULT_RANK_RTOL, reference_norm=None):
    value = _finite_matrix(value, "basis candidate")
    if min(value.shape) == 0:
        return value[:, :0], np.empty(0), 0.0
    U, s, _ = linalg.svd(value, full_matrices=False, check_finite=False,
                         lapack_driver="gesdd")
    threshold = rank_rtol * s[0]
    if reference_norm is not None:
        threshold = max(threshold, 64 * EPS * float(reference_norm))
    rank = int(np.count_nonzero(s > threshold))
    return U[:, :rank], s, float(threshold)


def mapped_overlap(H_rows, axes, Qadd_rows, rank_rtol=DEFAULT_RANK_RTOL):
    H_rows = _finite_matrix(H_rows, "H_rows")
    axes = _finite_matrix(axes, "axes")
    Qadd_rows = _finite_matrix(Qadd_rows, "restricted Fourier basis")
    if H_rows.shape[1] != axes.shape[0] or len(H_rows) != len(Qadd_rows):
        raise ValueError("incompatible mapping dimensions")
    raw = H_rows @ axes
    QZ, sz, tz = _orth(_center(raw), rank_rtol, np.linalg.norm(raw))
    QF, sf, tf = _orth(_center(Qadd_rows), rank_rtol, np.linalg.norm(Qadd_rows))
    rz, rf, q = QZ.shape[1], QF.shape[1], axes.shape[1]
    overlap = float(np.sum((QZ.T @ QF) ** 2))
    return {"overlap": overlap, "purity": _unit_interval(overlap / rz, "purity") if rz else None,
            "requested_rank_score": _unit_interval(overlap / q, "requested score") if q else None,
            "coverage": _unit_interval(overlap / rf, "coverage") if rf else None,
            "mean_squared_CCA": _unit_interval(overlap / min(rz, rf), "CCA") if min(rz, rf) else None,
            "mapped_rank": rz, "target_rank": rf, "selected_rank": q,
            "mapping_singular_values": sz, "target_singular_values": sf,
            "mapping_rank_threshold": tz, "target_rank_threshold": tf}


def _mapped_to_target(H_rows, axes, QF, rank_rtol):
    raw = H_rows @ axes
    QZ, sz, threshold = _orth(_center(raw), rank_rtol, np.linalg.norm(raw))
    rz, rf = QZ.shape[1], QF.shape[1]
    overlap = float(np.sum((QZ.T @ QF) ** 2))
    return {"overlap": overlap, "purity": _unit_interval(overlap / rz, "purity") if rz else None,
            "coverage": _unit_interval(overlap / rf, "coverage") if rf else None,
            "mean_squared_CCA": _unit_interval(overlap / min(rz, rf), "CCA") if min(rz, rf) else None,
            "mapped_rank": rz, "target_rank": rf,
            "mapping_rank_threshold": threshold, "mapping_singular_values": sz}


def ridge_refit(H, train_indices, train_labels, p, alpha=1e-3):
    H = _finite_matrix(H, "H")
    train = _indices(train_indices, len(H), "train_indices")
    y = _labels(train_labels, len(train), p)
    if not np.isfinite(alpha) or alpha <= 0:
        raise ValueError("alpha must be fixed, finite and strictly positive")
    X = H[train]
    Y = np.eye(p, dtype=np.float64)[y] - 1.0 / p
    penalty = len(train) * float(alpha)
    if len(train) <= H.shape[1]:
        matrix = X @ X.T
        matrix.flat[::len(matrix) + 1] += penalty
        factor = linalg.cho_factor(matrix, lower=True, check_finite=False)
        B = X.T @ linalg.cho_solve(factor, Y, check_finite=False)
        solver = "dual_cholesky"
    else:
        matrix = X.T @ X
        matrix.flat[::len(matrix) + 1] += penalty
        factor = linalg.cho_factor(matrix, lower=True, check_finite=False)
        B = linalg.cho_solve(factor, X.T @ Y, check_finite=False)
        solver = "primal_cholesky"
    if not np.all(np.isfinite(B)):
        raise ArithmeticError("nonfinite ridge coefficients")
    residual = X.T @ (X @ B - Y) / len(train) + alpha * B
    denominator = np.linalg.norm(X.T @ Y / len(train)) + alpha * np.linalg.norm(B)
    stationarity = float(np.linalg.norm(residual) / max(denominator, np.finfo(float).tiny))
    return B, {"alpha": float(alpha), "normal_equation_penalty": penalty,
               "solver": solver, "stationarity_relative_residual": stationarity,
               "rule_id": "raw_H_no_intercept_centered_onehot_mean_MSE_alpha_1e-3" if alpha == 1e-3 else
                          f"raw_H_no_intercept_centered_onehot_mean_MSE_alpha_{alpha:g}",
               "fit_rows_sha256": hashlib.sha256(train.tobytes()).hexdigest()}


def margins(logits, labels):
    logits = _finite_matrix(logits, "logits")
    labels = _labels(labels, len(logits), logits.shape[1])
    target = logits[np.arange(len(logits)), labels]
    competitors = logits.copy()
    competitors[np.arange(len(logits)), labels] = -np.inf
    return target - np.max(competitors, axis=1)


def _quantiles(values, prefix):
    q = np.quantile(values, [0, .05, .5, .95])
    return {prefix + suffix: float(v) for suffix, v in zip(("min", "q05", "q50", "q95"), q)}


def prediction_metrics(logits, labels, margin_atol=MARGIN_ATOL):
    logits = _finite_matrix(logits, "logits")
    labels = _labels(labels, len(logits), logits.shape[1])
    centered = logits - logits.mean(axis=1, keepdims=True)
    margin = margins(logits, labels)
    if not np.isfinite(margin_atol) or margin_atol < 0:
        raise ValueError("margin_atol must be finite and nonnegative")
    resolution = np.maximum(margin_atol, 64 * EPS * np.maximum(1.0, np.max(np.abs(centered), axis=1)))
    relative = logits - logits[np.arange(len(logits)), labels, None]
    relative[np.arange(len(logits)), labels] = -np.inf
    per_ce = np.logaddexp(0.0, special.logsumexp(relative, axis=1))
    target_raw = np.eye(logits.shape[1])[labels]
    target = target_raw - 1.0 / logits.shape[1]
    mse = float(np.mean(np.sum((logits - target_raw) ** 2, axis=1)))
    stats = {"accuracy": float(np.mean(logits.argmax(axis=1) == labels)),
             "strict_accuracy": float(np.mean(margin > 0)),
             "resolved_accuracy": float(np.mean(margin > resolution)),
             "near_tie_fraction": float(np.mean(np.abs(margin) <= resolution)),
             "ce": float(np.mean(per_ce)), "ce_positive_resolved": bool(np.all(per_ce > 0)),
             "ce_zero_rows": int(np.count_nonzero(per_ce == 0)),
             "ce_over_logp": float(np.mean(per_ce) / np.log(logits.shape[1])),
             "logp_minus_ce": float(np.log(logits.shape[1]) - np.mean(per_ce)),
             "mse": mse,
             "centered_target_mse": float(np.mean(np.sum((centered - target) ** 2, axis=1))),
             "logit_raw_rms": float(np.sqrt(np.mean(logits ** 2))),
             "logit_centered_rms": float(np.sqrt(np.mean(centered ** 2)))}
    stats.update(_quantiles(margin, "margin_"))
    return stats, {"margin": margin, "margin_resolution": resolution, "ce": per_ce}


def signed_rule(logits, p, labels=None):
    logits = _finite_matrix(logits, "logits")
    if logits.shape != (p * p, p):
        raise ValueError("logits must have shape (p^2,p)")
    add, _ = grid_labels(p)
    labels = add if labels is None else _labels(labels, len(logits), p)
    Fc = logits - logits.mean(axis=1, keepdims=True)
    means = _class_means(Fc, add, p)
    rule = (means - Fc.mean(axis=0, keepdims=True))[add]
    residual = Fc - rule
    gamma = margins(rule, labels)
    beta = -margins(residual, labels)
    bound = gamma - beta
    actual = margins(Fc, labels)
    slack = actual - bound
    if np.any(slack < -1024 * EPS * np.maximum(1, np.max(np.abs(Fc), axis=1))):
        raise ArithmeticError("signed-rule pointwise inequality violated")
    return {"rule_logits": rule, "residual_logits": residual, "gamma": gamma,
            "beta": beta, "bound": bound, "actual_margin": actual, "slack": slack}


@dataclass
class ObserverContext:
    p: int
    width: int
    train_indices: np.ndarray
    test_indices: np.ndarray
    random_seed: int = 20260927
    rank_rtol: float = DEFAULT_RANK_RTOL
    gap_rtol: float = DEFAULT_GAP_RTOL
    output_scale: float = 1.0

    def __post_init__(self):
        self.labels, _ = grid_labels(self.p)
        if self.width < 1 or int(self.width) != self.width:
            raise ValueError("width must be a positive integer")
        if not 0 < self.rank_rtol < 1 or not 0 < self.gap_rtol < 1:
            raise ValueError("rank/gap tolerances must lie in (0,1)")
        if not np.isfinite(self.output_scale) or self.output_scale <= 0:
            raise ValueError("output_scale must be finite and positive")
        self.train_indices = _indices(self.train_indices, self.p ** 2, "train_indices").copy()
        self.test_indices = _indices(self.test_indices, self.p ** 2, "test_indices").copy()
        if len(np.intersect1d(self.train_indices, self.test_indices)) or len(self.train_indices) + len(self.test_indices) != self.p ** 2:
            raise ValueError("train and test must be a disjoint partition of the grid")
        self.Qadd = addition_basis(self.p)
        restricted = self.Qadd[self.test_indices]
        self.Qtarget, self.target_singular_values, self.target_threshold = _orth(
            _center(restricted), self.rank_rtol, np.linalg.norm(restricted))
        self.qs = {"q4": 4, "qcomprehensive": min(self.p - 1, self.width)}
        self.random_frames = {}
        for name, q in self.qs.items():
            frames = []
            for i in range(8):
                if q > self.width:
                    frames.append(np.empty((self.width, 0)))
                    continue
                rng = np.random.default_rng(np.random.SeedSequence([self.random_seed, self.p, self.width, q, i]))
                Q, R = linalg.qr(rng.normal(size=(self.width, q)), mode="economic")
                Q *= np.where(np.diag(R) >= 0, 1.0, -1.0)
                frames.append(Q)
            self.random_frames[name] = frames
        self.fixed_axes = {}
        self.fixed_axes_validity = {}

    def set_fixed_axes(self, axes, source_validity=None):
        copied = {}
        for name, value in axes.items():
            if name not in self.qs:
                raise ValueError(f"unknown rank label {name}")
            value = _finite_matrix(value, "fixed axes")
            if value.shape[0] != self.width or value.shape[1] > self.qs[name]:
                raise ValueError("fixed axes dimensions do not match requested rank")
            if not np.allclose(value.T @ value, np.eye(value.shape[1]), atol=1e-9, rtol=1e-9):
                raise ValueError("fixed axes must be orthonormal")
            copied[name] = value.copy()
        self.fixed_axes = copied
        self.fixed_axes_validity = {} if source_validity is None else dict(source_validity)

    def evaluate(self, H, V, fixed_axes=None, actual_logits=None, fixed_axes_validity=None):
        H = _finite_matrix(H, "H")
        V = _finite_matrix(V, "V")
        if H.shape != (self.p ** 2, self.width) or V.shape != (self.p, self.width):
            raise ValueError("H or V dimensions do not match context")
        expected = self.output_scale * (H @ V.T)
        logits = expected if actual_logits is None else _finite_matrix(actual_logits, "actual_logits")
        if logits.shape != expected.shape:
            raise ValueError("actual_logits dimensions do not match")
        if actual_logits is not None and not np.allclose(logits, expected, rtol=1e-10, atol=1e-12):
            raise ValueError("actual_logits must equal the declared scaled linear head")
        metrics = hidden_fourier(H, self.p)
        arrays = fourier_spectrum(H, self.p)
        metadata = {"version": VERSION, "row_order": "a-major,b-minor",
                    "output_convention": "class-centered logits", "output_scale": self.output_scale,
                    "gram_rows": "training (Jacobian constant for 1HL linear head)",
                    "mapping_rows": "all held-out", "rank_rtol": self.rank_rtol,
                    "agop_positive_eigenvalue_rtol": self.rank_rtol,
                    "boundary_gap_rtol": self.gap_rtol,
                    "tie_rule": "argmax chooses first class; resolved margins separately",
                    "margin_resolution": "max(1e-10,64*float64_eps*max(1,max_abs_class_centered_logit)) per example",
                    "train_rows_sha256": hashlib.sha256(self.train_indices.tobytes()).hexdigest(),
                    "mapping_rows_sha256": hashlib.sha256(self.test_indices.tobytes()).hexdigest(),
                    "random_frame_seed": self.random_seed, "random_frames_per_q": 8,
                    "availability_row_set": "full grid (oracle)",
                    "energy_metric_id": "AH_add_centered_fullgrid",
                    "agop_metric_id": "CE_class_centered_hidden_AGOP_trainGram_heldoutMap"}

        Hc = _center(H)
        UH, sh, th = _orth(Hc, self.rank_rtol, np.linalg.norm(H))
        avail = float(np.sum((UH.T @ self.Qadd) ** 2) / (self.p - 1))
        metrics.update({"availability_A_plus": _unit_interval(avail, "availability"),
                        "availability_oracle_error": _unit_interval(1 - avail, "oracle error"),
                        "availability_rank": UH.shape[1], "availability_rank_threshold": th,
                        "availability_condition_retained": float(sh[0] / sh[UH.shape[1] - 1]) if UH.shape[1] else None})
        arrays["hidden_singular_values"] = sh
        Vc = self.output_scale * (V - V.mean(axis=0, keepdims=True))
        _, s, right = linalg.svd(Vc, full_matrices=False, check_finite=False)
        eig = s * s
        lead = float(eig[0]) if len(eig) else 0.0
        eig_threshold = self.rank_rtol * lead
        noise_floor = (64 * EPS * self.output_scale * np.linalg.norm(V)) ** 2
        eig_threshold = max(eig_threshold, noise_floor)
        positive = int(np.count_nonzero(eig > eig_threshold))
        arrays["agop_eigenvalues"] = eig
        metadata["agop_positive_eigenvalue_threshold"] = eig_threshold
        axes = {}
        axes_validity = {}
        Htest = H[self.test_indices]
        fixed = self.fixed_axes if fixed_axes is None else fixed_axes
        fixed_validity = self.fixed_axes_validity if fixed_axes is None else (fixed_axes_validity or {})
        for name, q in self.qs.items():
            prefix = "agop_" + name + "_"
            selected = min(q, positive)
            axes[name] = right[:selected].T.copy()
            mapped = _mapped_to_target(Htest, axes[name], self.Qtarget, self.rank_rtol)
            complete = positive >= q and q <= self.width
            gap = float((eig[q - 1] - (eig[q] if q < len(eig) else 0)) / lead) if complete and lead > 0 else None
            uncertainty = 64 * EPS * max(V.shape)
            gap_resolved = gap is not None and gap > max(self.gap_rtol, uncertainty)
            primary_valid = bool(complete and mapped["mapped_rank"] == q and mapped["target_rank"] >= q and gap_resolved)
            axes_validity[name] = primary_valid
            for key in ("overlap", "purity", "coverage", "mean_squared_CCA", "mapped_rank", "target_rank", "mapping_rank_threshold"):
                metrics[prefix + key] = mapped[key]
            metrics.update({prefix + "requested_rank_score": _unit_interval(mapped["overlap"] / q, "requested AGOP score"),
                            prefix + "valid": primary_valid, prefix + "q_requested": q,
                            prefix + "positive_rank": positive, prefix + "selected_rank": selected,
                            prefix + "relative_eigengap": gap, prefix + "gap_resolved": bool(gap_resolved),
                            prefix + "lambda_max": lead,
                            prefix + "relative_eigenvalue_uncertainty": uncertainty})
            arrays[prefix + "mapping_singular_values"] = mapped["mapping_singular_values"]
            null_scores, null_ranks = [], []
            for frame in self.random_frames[name]:
                result = _mapped_to_target(Htest, frame, self.Qtarget, self.rank_rtol)
                null_ranks.append(result["mapped_rank"])
                null_scores.append(result["purity"] if result["mapped_rank"] == q and result["target_rank"] >= q else None)
            valid_null = [v for v in null_scores if v is not None]
            for key, func in (("median", np.median), ("min", np.min), ("max", np.max)):
                metrics[prefix + "null_" + key] = float(func(valid_null)) if len(valid_null) == 8 else None
            metrics[prefix + "null_valid_count"] = len(valid_null)
            null_med = metrics[prefix + "null_median"]
            metrics[prefix + "minus_null"] = mapped["purity"] - null_med if primary_valid and null_med is not None else None
            metadata[prefix + "null_scores"] = null_scores
            metadata[prefix + "null_mapped_ranks"] = null_ranks
            metrics[prefix + "fixed_purity"] = None
            metrics[prefix + "fixed_valid"] = False
            metrics[prefix + "fixed_source_valid"] = fixed_validity.get(name)
            metrics[prefix + "fixed_mapping_valid"] = False
            if name in fixed:
                fixed_value = _finite_matrix(fixed[name], "fixed axes")
                if fixed_value.shape[0] != self.width or fixed_value.shape[1] > q:
                    raise ValueError("fixed axes shape mismatch")
                if not np.allclose(fixed_value.T @ fixed_value, np.eye(fixed_value.shape[1]), atol=1e-9, rtol=1e-9):
                    raise ValueError("fixed axes must be orthonormal")
                remapped = _mapped_to_target(Htest, fixed_value, self.Qtarget, self.rank_rtol)
                for key in ("purity", "coverage", "mapped_rank", "target_rank"):
                    metrics[prefix + "fixed_" + key] = remapped[key]
                metrics[prefix + "fixed_requested_rank_score"] = remapped["overlap"] / q
                fixed_mapping_valid = bool(fixed_value.shape[1] == q and remapped["mapped_rank"] == q and remapped["target_rank"] >= q)
                metrics[prefix + "fixed_mapping_valid"] = fixed_mapping_valid
                metrics[prefix + "fixed_valid"] = bool(fixed_mapping_valid and fixed_validity.get(name) is True)

        B, ridge_info = ridge_refit(H, self.train_indices, self.labels[self.train_indices], self.p)
        refit = H @ B
        metadata["refit"] = ridge_info
        arrays["refit_head"] = B
        for split_name, rows in (("train", self.train_indices), ("test", self.test_indices)):
            for kind, prediction in (("actual", logits), ("refit", refit)):
                stat, detailed = prediction_metrics(prediction[rows], self.labels[rows])
                for key, value in stat.items():
                    metrics[kind + "_" + split_name + "_" + key] = value
                for key, value in detailed.items():
                    arrays[kind + "_" + split_name + "_" + key] = value
        target = np.eye(self.p)[self.labels] - 1.0 / self.p
        for name, rows in (("train", self.train_indices), ("test", self.test_indices)):
            metrics["refit_" + name + "_mse"] = float(np.mean(np.sum((refit[rows] - target[rows]) ** 2, axis=1)))
        diff = logits - refit
        diff -= diff.mean(axis=1, keepdims=True)
        discrepancy = np.max(np.abs(diff), axis=1)
        refit_margin = margins(refit, self.labels)
        tracking_bound = refit_margin - 2 * discrepancy
        actual_margin = margins(logits, self.labels)
        if np.any(actual_margin < tracking_bound - 1024 * EPS * np.maximum(1, np.max(np.abs(logits), axis=1))):
            raise ArithmeticError("actual-head pointwise tracking inequality violated")
        arrays["tracking_test_discrepancy"] = discrepancy[self.test_indices]
        arrays["tracking_test_bound"] = tracking_bound[self.test_indices]
        metrics.update(_quantiles(discrepancy[self.test_indices], "tracking_test_discrepancy_"))
        metrics.update(_quantiles(tracking_bound[self.test_indices], "tracking_test_bound_"))
        metrics["tracking_test_certified_fraction"] = float(np.mean(tracking_bound[self.test_indices] > 0))
        centered_logits = logits - logits.mean(axis=1, keepdims=True)
        refit_centered = refit - refit.mean(axis=1, keepdims=True)
        tracking_resolution = np.maximum(MARGIN_ATOL, 1024 * EPS * np.maximum(1.0, np.maximum(
            np.max(np.abs(centered_logits), axis=1), np.max(np.abs(refit_centered), axis=1))))
        arrays["tracking_test_resolution"] = tracking_resolution[self.test_indices]
        metrics["tracking_test_resolved_certified_fraction"] = float(np.mean(
            tracking_bound[self.test_indices] > tracking_resolution[self.test_indices]))
        signed = signed_rule(logits, self.p)
        for key in ("gamma", "beta", "bound", "actual_margin", "slack"):
            values = signed[key][self.test_indices]
            arrays["signed_test_" + key] = values
            metrics.update(_quantiles(values, "signed_test_" + key + "_"))
        metrics["signed_test_certified_fraction"] = float(np.mean(signed["bound"][self.test_indices] > 0))
        signed_resolution = np.maximum(MARGIN_ATOL, 1024 * EPS * np.maximum(1.0, np.max(np.abs(centered_logits), axis=1)))
        arrays["signed_test_resolution"] = signed_resolution[self.test_indices]
        metrics["signed_test_resolved_certified_fraction"] = float(np.mean(
            signed["bound"][self.test_indices] > signed_resolution[self.test_indices]))
        metadata["certificate_resolution_note"] = "raw bound>0 and float64-resolution-aware fractions are separate; resolution is a numerical screen, not interval arithmetic"
        return {"metrics": metrics, "metadata": metadata, "arrays": arrays, "axes": axes,
                "axes_validity": axes_validity}
