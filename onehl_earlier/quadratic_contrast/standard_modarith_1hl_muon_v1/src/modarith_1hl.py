from __future__ import annotations

import csv
import hashlib
import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


OPTIMIZERS = (
    "polar_muon",
    "fro_ngd",
    "polar_feature_gd_head",
    "fro_feature_gd_head",
    "head_only",
    "gd",
)
DIRECTION_KINDS = ("polar_muon", "fro_ngd", "gd")
ACTIVATIONS = ("square", "relu", "softplus")
FEATURE_GEOMETRIES = ("euclidean", "row_sphere")
EPS = 1.0e-30


@dataclass(frozen=True)
class RunConfig:
    seed: int = 0
    split_seed: int = 1
    p: int = 31
    train_fraction: float = 0.3
    width: int = 128
    steps: int = 2_000
    optimizer: str = "polar_muon"
    lr_feature: float = 0.01
    lr_head: float = 0.01
    momentum: float = 0.0
    weight_decay_feature: float = 0.0
    weight_decay_head: float = 0.0
    feature_init_scale: float = 1.0 / math.sqrt(2.0)
    head_init_scale: float = 0.05
    log_every: int = 10
    device: str = "cpu"
    dtype: str = "float64"
    strict_polar_rank: bool = False
    train_accuracy_threshold: float = 0.99
    test_accuracy_threshold: float = 0.90
    alignment_threshold: float = 0.80
    minimum_delay_steps: int = 100
    agop_ranks: tuple[int, ...] = ()
    agop_ridge_alphas: tuple[float, ...] = (
        1.0e-5,
        1.0e-4,
        1.0e-3,
        1.0e-2,
        1.0e-1,
        1.0,
    )
    agop_probe_folds: int = 3
    agop_probe_seed: int = 314159
    agop_random_controls: int = 4
    activation: str = "square"
    feature_geometry: str = "euclidean"
    theory_gradient_observer: bool = False
    rule_margin_observer: bool = False
    adaptive_balance_start_step: int = -1
    adaptive_balance_denominator_floor: float = 1.0e-12
    adaptive_balance_decay_min: float | None = None
    adaptive_balance_decay_max: float | None = None

    def validate(self) -> None:
        if self.p < 3 or self.p % 2 == 0 or not _is_prime(self.p):
            raise ValueError("p must be an odd prime")
        if not 0.0 < self.train_fraction < 1.0:
            raise ValueError("train_fraction must lie in (0, 1)")
        if self.width < 1 or self.steps < 0 or self.log_every < 1:
            raise ValueError("width and log_every must be positive; steps must be nonnegative")
        if self.optimizer not in OPTIMIZERS:
            raise ValueError(f"optimizer must be one of {OPTIMIZERS}")
        if self.activation not in ACTIVATIONS:
            raise ValueError(f"activation must be one of {ACTIVATIONS}")
        if self.feature_geometry not in FEATURE_GEOMETRIES:
            raise ValueError(
                f"feature_geometry must be one of {FEATURE_GEOMETRIES}"
            )
        if not isinstance(self.theory_gradient_observer, bool):
            raise ValueError("theory_gradient_observer must be boolean")
        if not isinstance(self.rule_margin_observer, bool):
            raise ValueError("rule_margin_observer must be boolean")
        if (
            not isinstance(self.adaptive_balance_start_step, int)
            or isinstance(self.adaptive_balance_start_step, bool)
            or self.adaptive_balance_start_step < -1
        ):
            raise ValueError(
                "adaptive_balance_start_step must be -1 (disabled) or a "
                "nonnegative integer"
            )
        if (
            not math.isfinite(self.adaptive_balance_denominator_floor)
            or self.adaptive_balance_denominator_floor < 0.0
        ):
            raise ValueError(
                "adaptive_balance_denominator_floor must be finite and "
                "nonnegative"
            )
        for name, value in (
            ("adaptive_balance_decay_min", self.adaptive_balance_decay_min),
            ("adaptive_balance_decay_max", self.adaptive_balance_decay_max),
        ):
            if value is not None and (not math.isfinite(value) or value < 0.0):
                raise ValueError(f"{name} must be None or finite and nonnegative")
        if (
            self.adaptive_balance_decay_min is not None
            and self.adaptive_balance_decay_max is not None
            and self.adaptive_balance_decay_min
            > self.adaptive_balance_decay_max
        ):
            raise ValueError(
                "adaptive_balance_decay_min cannot exceed "
                "adaptive_balance_decay_max"
            )
        if self.adaptive_balance_start_step >= 0:
            if self.optimizer != "polar_feature_gd_head":
                raise ValueError(
                    "adaptive head-decay balancing requires "
                    "optimizer='polar_feature_gd_head'"
                )
            if self.feature_geometry != "row_sphere":
                raise ValueError(
                    "adaptive head-decay balancing requires "
                    "feature_geometry='row_sphere'"
                )
            if self.momentum != 0.0:
                raise ValueError(
                    "adaptive head-decay balancing requires zero momentum"
                )
        if self.lr_feature <= 0.0 or self.lr_head <= 0.0:
            raise ValueError("learning rates must be positive")
        if not 0.0 <= self.momentum < 1.0:
            raise ValueError("momentum must lie in [0, 1)")
        if self.optimizer != "polar_muon" and self.momentum != 0.0:
            raise ValueError("momentum is currently defined only for polar_muon")
        if min(self.weight_decay_feature, self.weight_decay_head) < 0.0:
            raise ValueError("weight decays must be nonnegative")
        if min(self.feature_init_scale, self.head_init_scale) < 0.0:
            raise ValueError("initialization scales must be nonnegative")
        if self.optimizer == "head_only" and self.weight_decay_feature != 0.0:
            raise ValueError(
                "head_only freezes feature weights, so weight_decay_feature must be zero"
            )
        if self.feature_geometry == "row_sphere":
            if self.weight_decay_feature != 0.0:
                raise ValueError(
                    "row_sphere fixes identifiable feature-row norms, so "
                    "weight_decay_feature must be zero"
                )
            if self.feature_init_scale <= 0.0:
                raise ValueError(
                    "row_sphere requires a positive feature_init_scale"
                )
        if self.dtype not in {"float32", "float64"}:
            raise ValueError("dtype must be float32 or float64")
        for name, value in (
            ("train_accuracy_threshold", self.train_accuracy_threshold),
            ("test_accuracy_threshold", self.test_accuracy_threshold),
            ("alignment_threshold", self.alignment_threshold),
        ):
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must lie in [0, 1]")
        if self.minimum_delay_steps < 0:
            raise ValueError("minimum_delay_steps must be nonnegative")
        if any(rank < 1 for rank in self.agop_ranks):
            raise ValueError("all requested AGOP ranks must be positive")
        maximum_agop_rank = min(self.width, self.p - 1)
        if any(rank > maximum_agop_rank for rank in self.agop_ranks):
            raise ValueError(
                "AGOP ranks cannot exceed min(width, p - 1)="
                f"{maximum_agop_rank}"
            )
        if not self.agop_ridge_alphas or any(
            not math.isfinite(alpha) or alpha <= 0.0
            for alpha in self.agop_ridge_alphas
        ):
            raise ValueError("AGOP ridge alphas must be finite and positive")
        if self.agop_probe_folds < 2:
            raise ValueError("agop_probe_folds must be at least two")
        if self.agop_random_controls < 1:
            raise ValueError("agop_random_controls must be positive")


@dataclass(frozen=True)
class ModularData:
    p: int
    pairs: torch.Tensor
    labels: torch.Tensor
    train_idx: torch.Tensor
    test_idx: torch.Tensor
    train_idx_numpy: np.ndarray
    test_idx_numpy: np.ndarray
    split_fingerprint: str


def _is_prime(value: int) -> bool:
    return value >= 2 and all(value % d for d in range(2, int(math.sqrt(value)) + 1))


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    if device.type == "mps":
        raise ValueError("MPS is not used: exact SVD and float64 support are insufficient")
    return device


def resolve_dtype(name: str) -> torch.dtype:
    return {"float32": torch.float32, "float64": torch.float64}[name]


def configure_determinism(seed: int, device: torch.device) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False


def make_data(
    p: int,
    train_fraction: float,
    split_seed: int,
    device: torch.device,
) -> ModularData:
    if p < 3 or not 0.0 < train_fraction < 1.0:
        raise ValueError("invalid modulus or training fraction")
    a = np.repeat(np.arange(p, dtype=np.int64), p)
    b = np.tile(np.arange(p, dtype=np.int64), p)
    pairs_np = np.stack((a, b), axis=1)
    labels_np = (a + b) % p
    permutation = np.random.default_rng(split_seed).permutation(p * p)
    train_count = int(train_fraction * p * p)
    if not 1 <= train_count < p * p:
        raise ValueError("the realized split must contain both train and test examples")
    train_idx = np.asarray(permutation[:train_count], dtype=np.int64)
    test_idx = np.asarray(permutation[train_count:], dtype=np.int64)
    specification = {
        "operation": "a+b mod p",
        "p": int(p),
        "requested_train_fraction": float(train_fraction),
        "realized_train_fraction": float(train_count / (p * p)),
        "split_seed": int(split_seed),
        "algorithm": "numpy.default_rng(seed).permutation; floor(fraction*p^2)",
    }
    digest = hashlib.sha256(json.dumps(specification, sort_keys=True).encode())
    digest.update(train_idx.tobytes())
    digest.update(test_idx.tobytes())
    return ModularData(
        p=int(p),
        pairs=torch.as_tensor(pairs_np, dtype=torch.long, device=device),
        labels=torch.as_tensor(labels_np, dtype=torch.long, device=device),
        train_idx=torch.as_tensor(train_idx, dtype=torch.long, device=device),
        test_idx=torch.as_tensor(test_idx, dtype=torch.long, device=device),
        train_idx_numpy=train_idx,
        test_idx_numpy=test_idx,
        split_fingerprint=digest.hexdigest(),
    )


def orthogonal_complement_basis(vector: torch.Tensor) -> torch.Tensor:
    if vector.ndim != 1 or not torch.isfinite(vector).all():
        raise ValueError("vector must be finite and one-dimensional")
    normalized = vector / torch.linalg.vector_norm(vector)
    last = torch.zeros_like(normalized)
    last[-1] = 1.0
    difference = last - normalized
    if float(torch.linalg.vector_norm(difference)) < 1.0e-14:
        return torch.eye(len(vector), dtype=vector.dtype, device=vector.device)[:, :-1]
    householder = torch.eye(len(vector), dtype=vector.dtype, device=vector.device)
    householder -= 2.0 * torch.outer(difference, difference) / torch.dot(difference, difference)
    basis = householder[:, :-1]
    return basis


def input_gauge_basis(p: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    null = torch.cat(
        (torch.ones(p, device=device, dtype=dtype), -torch.ones(p, device=device, dtype=dtype))
    )
    return orthogonal_complement_basis(null)


def output_gauge_basis(p: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    return orthogonal_complement_basis(torch.ones(p, device=device, dtype=dtype))


class QuadraticOneHiddenLayer(nn.Module):
    def __init__(
        self,
        p: int,
        width: int,
        feature_init_scale: float,
        head_init_scale: float,
        *,
        device: torch.device,
        dtype: torch.dtype,
        activation: str = "square",
    ) -> None:
        super().__init__()
        if activation not in ACTIVATIONS:
            raise ValueError(f"activation must be one of {ACTIVATIONS}")
        self.p = int(p)
        self.width = int(width)
        self.activation = activation
        self.feature_weight = nn.Parameter(
            feature_init_scale * torch.randn(width, 2 * p, device=device, dtype=dtype)
        )
        self.head_weight = nn.Parameter(
            head_init_scale * torch.randn(p, width, device=device, dtype=dtype)
        )
        self.register_buffer("input_basis", input_gauge_basis(p, device, dtype))
        self.register_buffer("output_basis", output_gauge_basis(p, device, dtype))
        self.project_gauges_()

    @torch.no_grad()
    def project_gauges_(self) -> None:
        null = torch.cat(
            (
                torch.ones(self.p, device=self.feature_weight.device, dtype=self.feature_weight.dtype),
                -torch.ones(self.p, device=self.feature_weight.device, dtype=self.feature_weight.dtype),
            )
        )
        null /= torch.linalg.vector_norm(null)
        self.feature_weight.sub_(torch.outer(self.feature_weight @ null, null))
        self.head_weight.sub_(self.head_weight.mean(dim=0, keepdim=True))

    def hidden(self, pairs: torch.Tensor) -> torch.Tensor:
        first = self.feature_weight[:, pairs[:, 0]].T
        second = self.feature_weight[:, self.p + pairs[:, 1]].T
        preactivation = first + second
        if self.activation == "square":
            return preactivation.square()
        if self.activation == "relu":
            return F.relu(preactivation)
        return F.softplus(preactivation)

    def forward(self, pairs: torch.Tensor) -> torch.Tensor:
        return self.hidden(pairs) @ self.head_weight.T / math.sqrt(self.width)


def support_polar_factor(
    matrix: torch.Tensor,
    *,
    strict_rank: bool = False,
) -> tuple[torch.Tensor, dict[str, float | int]]:
    if matrix.ndim != 2 or not torch.isfinite(matrix).all():
        raise TypeError("support_polar_factor requires a finite matrix")
    u, singular, vh = torch.linalg.svd(matrix, full_matrices=False)
    largest = float(singular[0].item()) if singular.numel() else 0.0
    tiny = torch.finfo(matrix.dtype).tiny
    cutoff = torch.finfo(matrix.dtype).eps * max(matrix.shape) * max(largest, tiny)
    mask = singular > cutoff
    rank = int(torch.count_nonzero(mask).item())
    required = min(matrix.shape)
    if strict_rank and rank < required:
        raise RuntimeError(
            f"polar gradient rank {rank} is below {required}; "
            f"s_max={largest:.3e}, cutoff={cutoff:.3e}"
        )
    if rank == 0:
        direction = torch.zeros_like(matrix)
    else:
        direction = u[:, mask] @ vh[mask, :]
    nonzero = singular[mask]
    discarded = singular[~mask]
    smallest_nonzero = float(nonzero[-1].item()) if rank else 0.0
    return direction, {
        "rank": rank,
        "required_rank": required,
        "nuclear_norm": float(singular.sum().item()),
        "retained_nuclear_norm": float(nonzero.sum().item()),
        "discarded_nuclear_norm": float(discarded.sum().item()),
        "s_max": largest,
        "s_min_nonzero": smallest_nonzero,
        "cutoff": float(cutoff),
        "condition_nonzero": float(largest / max(smallest_nonzero, EPS)) if rank else math.inf,
    }


def transformed_direction(
    gradient: torch.Tensor,
    optimizer: str,
    *,
    strict_rank: bool,
) -> tuple[torch.Tensor, dict[str, float | int]]:
    if optimizer not in DIRECTION_KINDS:
        raise ValueError(f"unknown optimizer {optimizer!r}")
    polar, diagnostics = support_polar_factor(gradient, strict_rank=strict_rank)
    raw_norm = float(torch.linalg.matrix_norm(gradient, ord="fro").item())
    if optimizer == "polar_muon":
        direction = polar
    elif optimizer == "fro_ngd":
        target_norm = math.sqrt(int(diagnostics["rank"]))
        direction = gradient * (target_norm / raw_norm) if raw_norm > 0.0 else torch.zeros_like(gradient)
    else:
        direction = gradient
    return direction, {
        **diagnostics,
        "raw_gradient_fro": raw_norm,
        "direction_fro": float(torch.linalg.matrix_norm(direction, ord="fro").item()),
    }


def rowwise_tangent_projection(
    gradient: torch.Tensor,
    point: torch.Tensor,
) -> torch.Tensor:
    if gradient.ndim != 2 or point.ndim != 2 or gradient.shape != point.shape:
        raise ValueError("gradient and point must be same-shaped matrices")
    if not bool(torch.isfinite(gradient).all()) or not bool(torch.isfinite(point).all()):
        raise ValueError("gradient and point must be finite")
    squared_norms = point.square().sum(dim=1, keepdim=True)
    if bool(torch.any(squared_norms <= 0.0)):
        raise ValueError("sphere points must have nonzero rows")
    radial_coefficients = (gradient * point).sum(dim=1, keepdim=True) / squared_norms
    return gradient - radial_coefficients * point


def retract_rows_to_fixed_norms(
    point: torch.Tensor,
    target_norms: torch.Tensor,
) -> torch.Tensor:
    if point.ndim != 2 or target_norms.ndim != 1:
        raise ValueError("point must be a matrix and target_norms a vector")
    if point.shape[0] != target_norms.shape[0]:
        raise ValueError("one target norm is required for every point row")
    if not bool(torch.isfinite(point).all()) or not bool(torch.isfinite(target_norms).all()):
        raise ValueError("point and target_norms must be finite")
    if bool(torch.any(target_norms <= 0.0)):
        raise ValueError("target row norms must be strictly positive")
    current_norms = torch.linalg.vector_norm(point, dim=1)
    if bool(torch.any(current_norms <= 0.0)):
        raise FloatingPointError("cannot retract a zero row onto a positive sphere")
    return point * (target_norms / current_norms)[:, None]


class EndToEndMatrixOptimizer:
    def __init__(self, model: QuadraticOneHiddenLayer, config: RunConfig) -> None:
        if (
            config.feature_geometry == "row_sphere"
            and config.weight_decay_feature != 0.0
        ):
            raise ValueError(
                "row_sphere fixes identifiable feature-row norms, so "
                "weight_decay_feature must be zero"
            )
        if config.optimizer == "head_only" and config.weight_decay_feature != 0.0:
            raise ValueError(
                "head_only freezes feature weights, so weight_decay_feature must be zero"
            )
        self.model = model
        self.config = config
        self.feature_momentum = torch.zeros_like(model.feature_weight)
        self.head_momentum = torch.zeros_like(model.head_weight)
        self.step_index = 0
        with torch.no_grad():
            identifiable = model.feature_weight @ model.input_basis
            self.feature_row_norm_targets = torch.linalg.vector_norm(
                identifiable, dim=1
            ).clone()
        if config.feature_geometry == "row_sphere" and bool(
            torch.any(self.feature_row_norm_targets <= 0.0)
        ):
            raise ValueError("row_sphere initialization produced a zero feature row")

    def zero_grad(self) -> None:
        self.model.zero_grad(set_to_none=True)

    def _effective_gradients(self) -> tuple[torch.Tensor, torch.Tensor]:
        feature_grad = self.model.feature_weight.grad
        head_grad = self.model.head_weight.grad
        if feature_grad is None or head_grad is None:
            raise RuntimeError("both matrix gradients are required")
        feature_effective = feature_grad @ self.model.input_basis
        head_effective = self.model.output_basis.T @ head_grad
        return feature_effective, head_effective

    @torch.no_grad()
    def feature_row_norm_diagnostics(self) -> dict[str, float]:
        identifiable = self.model.feature_weight @ self.model.input_basis
        norms = torch.linalg.vector_norm(identifiable, dim=1)
        absolute_error = torch.abs(norms - self.feature_row_norm_targets)
        positive_targets = self.feature_row_norm_targets.clamp_min(
            torch.finfo(norms.dtype).tiny
        )
        relative_error = absolute_error / positive_targets
        return {
            "feature_identifiable_row_norm_min": float(norms.min().item()),
            "feature_identifiable_row_norm_mean": float(norms.mean().item()),
            "feature_identifiable_row_norm_max": float(norms.max().item()),
            "feature_identifiable_target_row_norm_min": float(
                self.feature_row_norm_targets.min().item()
            ),
            "feature_identifiable_target_row_norm_mean": float(
                self.feature_row_norm_targets.mean().item()
            ),
            "feature_identifiable_target_row_norm_max": float(
                self.feature_row_norm_targets.max().item()
            ),
            "feature_row_norm_max_abs_error": float(absolute_error.max().item()),
            "feature_row_norm_max_relative_error": float(relative_error.max().item()),
        }

    @torch.no_grad()
    def step(self) -> dict[str, float | int]:
        optimizer_source_state_step = self.step_index
        optimizer_update_number = self.step_index + 1
        feature_effective, head_effective = self._effective_gradients()
        feature_gradient_full = self.model.feature_weight.grad
        head_gradient_full = self.model.head_weight.grad
        if feature_gradient_full is None or head_gradient_full is None:
            raise RuntimeError("both matrix gradients are required")
        feature_point_before_full = self.model.feature_weight.clone()
        feature_data_effective = feature_effective
        head_data_effective = head_effective
        if self.config.optimizer == "polar_muon" and self.config.momentum > 0.0:
            beta = self.config.momentum
            self.feature_momentum.mul_(beta).add_(
                feature_effective @ self.model.input_basis.T, alpha=1.0 - beta
            )
            self.head_momentum.mul_(beta).add_(
                self.model.output_basis @ head_effective, alpha=1.0 - beta
            )
            feature_effective = self.feature_momentum @ self.model.input_basis
            head_effective = self.model.output_basis.T @ self.head_momentum

        feature_pre_tangent = feature_effective
        feature_data_tangent = feature_data_effective
        feature_point_effective: torch.Tensor | None = None
        tangent_max_abs_cosine = math.nan
        if self.config.feature_geometry == "row_sphere":
            feature_point_effective = (
                self.model.feature_weight @ self.model.input_basis
            )
            feature_effective = rowwise_tangent_projection(
                feature_pre_tangent, feature_point_effective
            )
            feature_data_tangent = rowwise_tangent_projection(
                feature_data_effective, feature_point_effective
            )
            tangent_inner = torch.abs(
                (feature_effective * feature_point_effective).sum(dim=1)
            )
            tangent_denominator = (
                torch.linalg.vector_norm(feature_effective, dim=1)
                * torch.linalg.vector_norm(feature_point_effective, dim=1)
            )
            tangent_cosines = torch.where(
                tangent_denominator > 0.0,
                tangent_inner / tangent_denominator,
                torch.zeros_like(tangent_denominator),
            )
            tangent_max_abs_cosine = float(tangent_cosines.max().item())
        radial_gradient = feature_pre_tangent - feature_effective

        feature_kind, head_kind = {
            "polar_muon": ("polar_muon", "polar_muon"),
            "fro_ngd": ("fro_ngd", "fro_ngd"),
            "polar_feature_gd_head": ("polar_muon", "gd"),
            "fro_feature_gd_head": ("fro_ngd", "gd"),
            "head_only": ("gd", "gd"),
            "gd": ("gd", "gd"),
        }[self.config.optimizer]
        feature_direction_eff, feature_diag = transformed_direction(
            feature_effective,
            feature_kind,
            strict_rank=self.config.strict_polar_rank,
        )
        head_direction_eff, head_diag = transformed_direction(
            head_effective,
            head_kind,
            strict_rank=self.config.strict_polar_rank,
        )
        if self.config.optimizer == "head_only":
            feature_direction_eff = torch.zeros_like(feature_direction_eff)
            feature_diag = {**feature_diag, "direction_fro": 0.0}
        if self.config.momentum == 0.0:
            feature_tangent_gradient_nuclear_norm = float(
                feature_diag["nuclear_norm"]
            )
            feature_tangent_gradient_retained_nuclear_norm = float(
                feature_diag["retained_nuclear_norm"]
            )
        else:
            feature_tangent_gradient_nuclear_norm = float(
                torch.linalg.matrix_norm(feature_data_tangent, ord="nuc").item()
            )
            feature_tangent_gradient_retained_nuclear_norm = float(
                feature_diag["retained_nuclear_norm"]
            )
        feature_data_direction_pairing = float(
            torch.sum(feature_data_tangent * feature_direction_eff).item()
        )
        feature_pre_retraction_step_eff = (
            -self.config.lr_feature * feature_direction_eff
        )
        feature_pre_retraction_signed_first_order_ce_change = float(
            torch.sum(
                feature_data_tangent * feature_pre_retraction_step_eff
            ).item()
        )
        head_direction = self.model.output_basis @ head_direction_eff
        head_point_before = self.model.head_weight.clone()
        head_point_effective = self.model.output_basis.T @ head_point_before
        adaptive_head_decay_active = bool(
            self.config.adaptive_balance_start_step >= 0
            and optimizer_update_number
            > self.config.adaptive_balance_start_step
        )
        adaptive_head_decay_denominator = float(
            -torch.sum(head_data_effective * head_point_effective).item()
        )
        adaptive_head_decay_formula_used = bool(
            adaptive_head_decay_active
            and adaptive_head_decay_denominator
            > self.config.adaptive_balance_denominator_floor
        )
        if adaptive_head_decay_formula_used:
            adaptive_head_decay_unclipped_lambda = float(
                (
                    torch.sum(head_data_effective.square()).item()
                    + (self.config.lr_feature / self.config.lr_head)
                    * feature_data_direction_pairing
                )
                / adaptive_head_decay_denominator
            )
        else:
            adaptive_head_decay_unclipped_lambda = float(
                self.config.weight_decay_head
            )
        chosen_head_decay = adaptive_head_decay_unclipped_lambda
        if adaptive_head_decay_active:
            if self.config.adaptive_balance_decay_min is not None:
                chosen_head_decay = max(
                    chosen_head_decay,
                    self.config.adaptive_balance_decay_min,
                )
            if self.config.adaptive_balance_decay_max is not None:
                chosen_head_decay = min(
                    chosen_head_decay,
                    self.config.adaptive_balance_decay_max,
                )
        adaptive_head_decay_was_clipped = bool(
            adaptive_head_decay_active
            and chosen_head_decay != adaptive_head_decay_unclipped_lambda
        )
        head_decay_direction = chosen_head_decay * head_point_before
        head_combined_direction = head_direction + head_decay_direction
        head_data_gradient = self.model.output_basis @ head_data_effective
        head_data_gradient_dot_combined_direction = float(
            torch.sum(
                head_data_gradient * head_combined_direction
            ).item()
        )
        head_signed_first_order_ce_change = (
            -self.config.lr_head
            * head_data_gradient_dot_combined_direction
        )
        total_signed_first_order_ce_change = (
            feature_pre_retraction_signed_first_order_ce_change
            + head_signed_first_order_ce_change
        )
        adaptive_head_decay_balance_residual = float(
            torch.sum(head_data_effective.square()).item()
            + (self.config.lr_feature / self.config.lr_head)
            * feature_data_direction_pairing
            + chosen_head_decay
            * torch.sum(head_data_effective * head_point_effective).item()
        )
        head_direction_norm = float(
            torch.linalg.matrix_norm(head_direction, ord="fro").item()
        )
        head_decay_direction_norm = float(
            torch.linalg.matrix_norm(head_decay_direction, ord="fro").item()
        )
        head_combined_direction_norm = float(
            torch.linalg.matrix_norm(head_combined_direction, ord="fro").item()
        )
        if head_direction_norm > 0.0 and head_decay_direction_norm > 0.0:
            head_direction_decay_cosine = float(
                torch.sum(head_direction * head_decay_direction).item()
                / (head_direction_norm * head_decay_direction_norm)
            )
            head_decay_balance_ratio = (
                head_direction_norm / head_decay_direction_norm
            )
        else:
            head_direction_decay_cosine = math.nan
            head_decay_balance_ratio = math.nan

        transformed_radial_fro = math.nan
        pre_retraction_max_abs_error = math.nan
        post_retraction_max_abs_error = math.nan
        retraction_correction_fro = 0.0
        feature_retraction_gradient_inner_product = math.nan
        actual_feature_step_fro = math.nan
        if self.config.optimizer == "head_only":
            actual_feature_step_fro = 0.0
            self.model.head_weight.mul_(
                1.0 - self.config.lr_head * chosen_head_decay
            )
            self.model.head_weight.add_(head_direction, alpha=-self.config.lr_head)
            self.model.head_weight.sub_(
                self.model.head_weight.mean(dim=0, keepdim=True)
            )
        elif self.config.feature_geometry == "euclidean":
            feature_direction = feature_direction_eff @ self.model.input_basis.T
            self.model.feature_weight.mul_(
                1.0 - self.config.lr_feature * self.config.weight_decay_feature
            )
            self.model.head_weight.mul_(
                1.0 - self.config.lr_head * chosen_head_decay
            )
            self.model.feature_weight.add_(
                feature_direction, alpha=-self.config.lr_feature
            )
            self.model.head_weight.add_(head_direction, alpha=-self.config.lr_head)
            self.model.project_gauges_()
        else:
            if feature_point_effective is None:
                raise AssertionError("row_sphere point was not initialized")
            squared_point_norms = feature_point_effective.square().sum(
                dim=1, keepdim=True
            )
            transformed_radial_coefficients = (
                (feature_direction_eff * feature_point_effective).sum(
                    dim=1, keepdim=True
                )
                / squared_point_norms
            )
            transformed_radial = (
                transformed_radial_coefficients * feature_point_effective
            )
            feature_posttransform_tangent_direction_eff = (
                feature_direction_eff - transformed_radial
            )
            transformed_radial_fro = float(
                torch.linalg.matrix_norm(transformed_radial, ord="fro").item()
            )
            provisional = (
                feature_point_effective
                - self.config.lr_feature * feature_direction_eff
            )
            provisional_norms = torch.linalg.vector_norm(provisional, dim=1)
            pre_retraction_max_abs_error = float(
                torch.abs(
                    provisional_norms - self.feature_row_norm_targets
                ).max().item()
            )
            retracted = retract_rows_to_fixed_norms(
                provisional, self.feature_row_norm_targets
            )
            retraction_correction_fro = float(
                torch.linalg.matrix_norm(retracted - provisional, ord="fro").item()
            )
            actual_feature_step_fro = float(
                torch.linalg.matrix_norm(
                    retracted - feature_point_effective, ord="fro"
                ).item()
            )
            post_retraction_max_abs_error = float(
                torch.abs(
                    torch.linalg.vector_norm(retracted, dim=1)
                    - self.feature_row_norm_targets
                ).max().item()
            )
            self.model.feature_weight.copy_(
                retracted @ self.model.input_basis.T
            )
            self.model.head_weight.mul_(
                1.0 - self.config.lr_head * chosen_head_decay
            )
            self.model.head_weight.add_(head_direction, alpha=-self.config.lr_head)
            self.model.head_weight.sub_(
                self.model.head_weight.mean(dim=0, keepdim=True)
            )
            feature_retraction_error_full = (
                self.model.feature_weight - feature_point_before_full
                + self.config.lr_feature
                * (
                    feature_posttransform_tangent_direction_eff
                    @ self.model.input_basis.T
                )
            )
            feature_retraction_gradient_inner_product = float(
                torch.sum(
                    feature_gradient_full * feature_retraction_error_full
                ).item()
            )
        feature_exact_gradient_increment_inner_product = float(
            torch.sum(
                feature_gradient_full
                * (self.model.feature_weight - feature_point_before_full)
            ).item()
        )
        head_exact_gradient_increment_inner_product = float(
            torch.sum(
                head_gradient_full
                * (self.model.head_weight - head_point_before)
            ).item()
        )
        full_exact_gradient_increment_inner_product = (
            feature_exact_gradient_increment_inner_product
            + head_exact_gradient_increment_inner_product
        )
        actual_head_step_fro = float(
            torch.linalg.matrix_norm(
                self.model.head_weight - head_point_before, ord="fro"
            ).item()
        )
        self.step_index += 1
        return {
            "optimizer_step": self.step_index,
            "feature_gradient_fro": float(feature_diag["raw_gradient_fro"]),
            "feature_pre_tangent_gradient_fro": float(
                torch.linalg.matrix_norm(feature_pre_tangent, ord="fro").item()
            ),
            "feature_tangent_gradient_fro": float(
                feature_diag["raw_gradient_fro"]
            ),
            "feature_tangent_gradient_nuclear_norm": (
                feature_tangent_gradient_nuclear_norm
            ),
            "feature_tangent_gradient_retained_nuclear_norm": (
                feature_tangent_gradient_retained_nuclear_norm
            ),
            "feature_data_direction_pairing": feature_data_direction_pairing,
            "feature_pre_retraction_signed_first_order_ce_change": (
                feature_pre_retraction_signed_first_order_ce_change
            ),
            "feature_radial_gradient_fro": float(
                torch.linalg.matrix_norm(radial_gradient, ord="fro").item()
            ),
            "feature_tangent_max_abs_cosine": tangent_max_abs_cosine,
            "head_gradient_fro": float(head_diag["raw_gradient_fro"]),
            "feature_direction_fro": float(feature_diag["direction_fro"]),
            "feature_transformed_direction_radial_fro": transformed_radial_fro,
            "head_direction_fro": float(head_diag["direction_fro"]),
            "feature_update_fro": self.config.lr_feature * float(feature_diag["direction_fro"]),
            "feature_actual_step_fro": actual_feature_step_fro,
            "feature_pre_retraction_row_norm_max_abs_error": (
                pre_retraction_max_abs_error
            ),
            "feature_post_retraction_row_norm_max_abs_error": (
                post_retraction_max_abs_error
            ),
            "feature_retraction_correction_fro": retraction_correction_fro,
            "feature_retraction_gradient_inner_product": (
                feature_retraction_gradient_inner_product
            ),
            "feature_exact_gradient_increment_inner_product": (
                feature_exact_gradient_increment_inner_product
            ),
            "head_update_fro": self.config.lr_head * float(head_diag["direction_fro"]),
            "head_decay_update_fro": (
                self.config.lr_head * head_decay_direction_norm
            ),
            "head_combined_direction_fro": head_combined_direction_norm,
            "head_data_gradient_dot_combined_direction": (
                head_data_gradient_dot_combined_direction
            ),
            "head_signed_first_order_ce_change": (
                head_signed_first_order_ce_change
            ),
            "total_signed_first_order_ce_change": (
                total_signed_first_order_ce_change
            ),
            "head_exact_gradient_increment_inner_product": (
                head_exact_gradient_increment_inner_product
            ),
            "full_exact_gradient_increment_inner_product": (
                full_exact_gradient_increment_inner_product
            ),
            "head_actual_step_fro": actual_head_step_fro,
            "head_direction_decay_cosine": head_direction_decay_cosine,
            "head_decay_balance_ratio": head_decay_balance_ratio,
            "adaptive_head_decay_active": int(adaptive_head_decay_active),
            "adaptive_head_decay_formula_used": int(
                adaptive_head_decay_formula_used
            ),
            "adaptive_head_decay_was_clipped": int(
                adaptive_head_decay_was_clipped
            ),
            "adaptive_head_decay_lambda": chosen_head_decay,
            "adaptive_head_decay_denominator": (
                adaptive_head_decay_denominator
            ),
            "adaptive_head_decay_unclipped_lambda": (
                adaptive_head_decay_unclipped_lambda
            ),
            "adaptive_head_decay_balance_residual": (
                adaptive_head_decay_balance_residual
            ),
            "adaptive_head_decay_source_state_step": optimizer_source_state_step,
            "adaptive_head_decay_update_number": optimizer_update_number,
            "feature_gradient_rank": int(feature_diag["rank"]),
            "head_gradient_rank": int(head_diag["rank"]),
            "feature_required_rank": int(feature_diag["required_rank"]),
            "head_required_rank": int(head_diag["required_rank"]),
            "feature_gradient_condition": float(feature_diag["condition_nonzero"]),
            "head_gradient_condition": float(head_diag["condition_nonzero"]),
        }


def classification_margins(
    logits: torch.Tensor, labels: torch.Tensor
) -> torch.Tensor:
    if logits.ndim != 2 or labels.ndim != 1 or logits.shape[0] != len(labels):
        raise ValueError("logits and labels have incompatible shapes")
    if logits.shape[1] < 2:
        raise ValueError("classification margins require at least two classes")
    if labels.dtype != torch.long or labels.device != logits.device:
        raise ValueError("labels must be torch.long on the logits device")
    return (
        logits.gather(1, labels[:, None]).squeeze(1)
        - logits.masked_fill(
            F.one_hot(labels, logits.shape[1]).bool(), -torch.inf
        ).max(dim=1).values
    )


@torch.no_grad()
def partition_metrics(
    model: QuadraticOneHiddenLayer,
    pairs: torch.Tensor,
    labels: torch.Tensor,
) -> dict[str, float]:
    logits = model(pairs)
    margins = classification_margins(logits, labels)
    return {
        "ce": float(F.cross_entropy(logits, labels).item()),
        "accuracy": float((logits.argmax(dim=1) == labels).double().mean().item()),
        "margin_mean": float(margins.mean().item()),
        "margin_min": float(margins.min().item()),
    }


@torch.no_grad()
def empirical_gram_head_diagnostics(
    train_hidden: torch.Tensor,
    *,
    width: int,
    head_learning_rate: float,
    head_weight_decay: float,
) -> dict[str, float]:
    if train_hidden.ndim != 2 or train_hidden.shape[0] < 1:
        raise ValueError("train_hidden must be a nonempty matrix")
    if train_hidden.shape[1] != width or width < 1:
        raise ValueError("width must equal the hidden feature dimension")
    if not bool(torch.isfinite(train_hidden).all()):
        raise ValueError("train_hidden must be finite")
    if not math.isfinite(head_learning_rate) or head_learning_rate <= 0.0:
        raise ValueError("head_learning_rate must be finite and positive")
    if not math.isfinite(head_weight_decay) or head_weight_decay < 0.0:
        raise ValueError("head_weight_decay must be finite and nonnegative")

    sample_count = train_hidden.shape[0]
    gram = train_hidden.T @ train_hidden / sample_count
    gram_largest = torch.linalg.eigvalsh(gram)[-1].clamp_min(0.0)
    regularization = float(head_weight_decay)
    smoothness = regularization + float(gram_largest.item()) / (2.0 * width)
    trace_envelope = regularization + float(
        train_hidden.square().sum().item()
    ) / (2.0 * sample_count * width)
    stepsize_lhs = head_learning_rate * (smoothness + regularization)
    contraction_factor = max(
        abs(1.0 - head_learning_rate * regularization),
        abs(1.0 - head_learning_rate * smoothness),
    )
    return {
        "head_empirical_gram_smoothness": smoothness,
        "head_empirical_trace_smoothness_envelope": trace_envelope,
        "head_contraction_stepsize_lhs": stepsize_lhs,
        "head_contraction_factor": contraction_factor,
    }


class AllStepTheoryBookkeeper:
    def __init__(
        self,
        data: ModularData,
        *,
        width: int,
        head_learning_rate: float,
        head_weight_decay: float,
    ) -> None:
        if width < 1:
            raise ValueError("width must be positive")
        self.data = data
        self.width = int(width)
        self.head_learning_rate = float(head_learning_rate)
        self.head_weight_decay = float(head_weight_decay)
        self.last_step: int | None = None
        self.head_trace_smoothness_max = -math.inf
        self.head_trace_contraction_lhs_max = -math.inf
        self.fixed_decay_contraction_applicable = True
        self.first_fit_step: int | None = None
        self.postfit_train_margin_min_running = math.inf
        self.witness_flat_index: int | None = None
        self.witness_operand_a: int | None = None
        self.witness_operand_b: int | None = None
        self.witness_target_class: int | None = None
        self.witness_competitor_class: int | None = None
        self.witness_margin = math.nan
        self.witness_initial_margin = math.nan
        self.witness_cumulative_u_abs = 0.0
        self.witness_cumulative_u_hybrid = 0.0
        self.previous_witness_hidden: torch.Tensor | None = None
        self.previous_witness_head_difference: torch.Tensor | None = None

    @torch.no_grad()
    def update(
        self,
        model: QuadraticOneHiddenLayer,
        step: int,
        *,
        fixed_decay_contraction_applicable: bool = True,
    ) -> None:
        if self.last_step is not None and step <= self.last_step:
            raise ValueError("all-step diagnostics require increasing step indices")
        if model.width != self.width or model.p != self.data.p:
            raise ValueError("model shape does not match the diagnostic protocol")

        train_pairs = self.data.pairs[self.data.train_idx]
        train_labels = self.data.labels[self.data.train_idx]
        train_hidden = model.hidden(train_pairs)
        train_logits = train_hidden @ model.head_weight.T / math.sqrt(self.width)
        train_margins = classification_margins(train_logits, train_labels)

        if not isinstance(fixed_decay_contraction_applicable, bool):
            raise ValueError("fixed_decay_contraction_applicable must be boolean")
        if not fixed_decay_contraction_applicable:
            self.fixed_decay_contraction_applicable = False
        if self.fixed_decay_contraction_applicable:
            sample_count = train_hidden.shape[0]
            trace_smoothness = self.head_weight_decay + float(
                train_hidden.square().sum().item()
            ) / (2.0 * sample_count * self.width)
            trace_lhs = self.head_learning_rate * (
                trace_smoothness + self.head_weight_decay
            )
            self.head_trace_smoothness_max = max(
                self.head_trace_smoothness_max, trace_smoothness
            )
            self.head_trace_contraction_lhs_max = max(
                self.head_trace_contraction_lhs_max, trace_lhs
            )

        exactly_fitted = bool(torch.all(train_logits.argmax(dim=1) == train_labels))
        just_froze_witness = False
        if exactly_fitted and self.first_fit_step is None:
            self.first_fit_step = int(step)
            self.postfit_train_margin_min_running = float(train_margins.min().item())

            test_pairs = self.data.pairs[self.data.test_idx]
            test_labels = self.data.labels[self.data.test_idx]
            test_hidden = model.hidden(test_pairs)
            test_logits = test_hidden @ model.head_weight.T / math.sqrt(self.width)
            target_logits = test_logits.gather(1, test_labels[:, None])
            witness_margins = target_logits - test_logits
            witness_margins = witness_margins.masked_fill(
                F.one_hot(test_labels, self.data.p).bool(), torch.inf
            )
            flat_location = int(torch.argmin(witness_margins).item())
            test_position, competitor = divmod(flat_location, self.data.p)
            flat_index = int(self.data.test_idx[test_position].item())
            target = int(test_labels[test_position].item())
            pair = self.data.pairs[flat_index]
            head_difference = (
                model.head_weight[target] - model.head_weight[competitor]
            ).detach().clone()
            witness_hidden = test_hidden[test_position].detach().clone()
            margin = float(
                torch.dot(head_difference, witness_hidden).item()
                / math.sqrt(self.width)
            )
            self.witness_flat_index = flat_index
            self.witness_operand_a = int(pair[0].item())
            self.witness_operand_b = int(pair[1].item())
            self.witness_target_class = target
            self.witness_competitor_class = int(competitor)
            self.witness_margin = margin
            self.witness_initial_margin = margin
            self.previous_witness_hidden = witness_hidden
            self.previous_witness_head_difference = head_difference
            just_froze_witness = True
        elif self.first_fit_step is not None:
            self.postfit_train_margin_min_running = min(
                self.postfit_train_margin_min_running,
                float(train_margins.min().item()),
            )

        if self.witness_flat_index is not None and not just_froze_witness:
            assert self.witness_target_class is not None
            assert self.witness_competitor_class is not None
            assert self.previous_witness_hidden is not None
            assert self.previous_witness_head_difference is not None
            witness_pair = self.data.pairs[self.witness_flat_index][None, :]
            witness_hidden = model.hidden(witness_pair)[0].detach().clone()
            head_difference = (
                model.head_weight[self.witness_target_class]
                - model.head_weight[self.witness_competitor_class]
            ).detach().clone()
            delta_head = head_difference - self.previous_witness_head_difference
            delta_hidden = witness_hidden - self.previous_witness_hidden
            root_width = math.sqrt(self.width)
            u_abs = float(
                (
                    torch.linalg.vector_norm(delta_head)
                    * torch.linalg.vector_norm(self.previous_witness_hidden)
                    + torch.linalg.vector_norm(head_difference)
                    * torch.linalg.vector_norm(delta_hidden)
                ).item()
                / root_width
            )
            u_hybrid = float(
                (
                    torch.dot(delta_head, self.previous_witness_hidden)
                    + torch.linalg.vector_norm(head_difference)
                    * torch.linalg.vector_norm(delta_hidden)
                ).item()
                / root_width
            )
            self.witness_cumulative_u_abs += u_abs
            self.witness_cumulative_u_hybrid += u_hybrid
            self.witness_margin = float(
                torch.dot(head_difference, witness_hidden).item() / root_width
            )
            self.previous_witness_hidden = witness_hidden
            self.previous_witness_head_difference = head_difference

        self.last_step = int(step)

    def snapshot(self) -> dict[str, float | int]:
        result: dict[str, float | int] = {
            "theory_all_step_head_trace_smoothness_max": (
                self.head_trace_smoothness_max
                if self.fixed_decay_contraction_applicable
                else math.nan
            ),
            "theory_all_step_head_trace_contraction_stepsize_lhs_max": (
                self.head_trace_contraction_lhs_max
                if self.fixed_decay_contraction_applicable
                else math.nan
            ),
            "theory_all_step_head_fixed_decay_contraction_applicable": int(
                self.fixed_decay_contraction_applicable
            ),
            "theory_all_step_first_exact_train_fit_step": (
                self.first_fit_step if self.first_fit_step is not None else math.nan
            ),
            "theory_all_step_postfit_train_margin_min_running": (
                self.postfit_train_margin_min_running
                if self.first_fit_step is not None
                else math.nan
            ),
        }
        witness_identity = {
            "theory_all_step_witness_flat_index": self.witness_flat_index,
            "theory_all_step_witness_operand_a": self.witness_operand_a,
            "theory_all_step_witness_operand_b": self.witness_operand_b,
            "theory_all_step_witness_target_class": self.witness_target_class,
            "theory_all_step_witness_competitor_class": (
                self.witness_competitor_class
            ),
        }
        for name, value in witness_identity.items():
            result[name] = value if value is not None else math.nan
        if self.witness_flat_index is None:
            result.update(
                {
                    "theory_all_step_witness_margin": math.nan,
                    "theory_all_step_witness_cumulative_u_abs": math.nan,
                    "theory_all_step_witness_cumulative_u_hybrid": math.nan,
                    "theory_all_step_witness_abs_margin_upper_envelope": math.nan,
                    "theory_all_step_witness_hybrid_margin_upper_envelope": math.nan,
                }
            )
        else:
            result.update(
                {
                    "theory_all_step_witness_margin": self.witness_margin,
                    "theory_all_step_witness_cumulative_u_abs": (
                        self.witness_cumulative_u_abs
                    ),
                    "theory_all_step_witness_cumulative_u_hybrid": (
                        self.witness_cumulative_u_hybrid
                    ),
                    "theory_all_step_witness_abs_margin_upper_envelope": (
                        self.witness_initial_margin
                        + self.witness_cumulative_u_abs
                    ),
                    "theory_all_step_witness_hybrid_margin_upper_envelope": (
                        self.witness_initial_margin
                        + self.witness_cumulative_u_hybrid
                    ),
                }
            )
        return result


@torch.no_grad()
def rule_margin_diagnostics(
    logits: torch.Tensor,
    pairs: torch.Tensor,
    labels: torch.Tensor,
    test_idx: torch.Tensor,
    p: int,
) -> dict[str, float]:
    if p < 3 or p % 2 == 0:
        raise ValueError("p must be odd and at least three")
    if logits.ndim != 2 or logits.shape != (p * p, p):
        raise ValueError("logits must have shape (p^2, p)")
    if pairs.ndim != 2 or pairs.shape != (p * p, 2):
        raise ValueError("pairs must contain the complete ordered p x p table")
    if labels.ndim != 1 or labels.shape[0] != p * p:
        raise ValueError("labels must contain one class per pair")
    if test_idx.ndim != 1 or test_idx.numel() < 1:
        raise ValueError("test_idx must be a nonempty vector")
    if not bool(torch.isfinite(logits).all()):
        raise ValueError("logits must be finite")

    centered = logits - logits.mean(dim=1, keepdim=True)
    baseline = torch.full_like(centered, -1.0 / p)
    sum_teacher = baseline.scatter(1, labels[:, None], 1.0 - 1.0 / p)
    difference_labels = torch.remainder(pairs[:, 0] - pairs[:, 1], p)
    difference_teacher = baseline.scatter(
        1, difference_labels[:, None], 1.0 - 1.0 / p
    )
    sum_norm_sq = sum_teacher.square().sum()
    difference_norm_sq = difference_teacher.square().sum()
    addition_coefficient = (centered * sum_teacher).sum() / sum_norm_sq
    difference_coefficient = (
        (centered * difference_teacher).sum() / difference_norm_sq
    )
    ideal = (
        addition_coefficient * sum_teacher
        + difference_coefficient * difference_teacher
    )
    residual = centered - ideal

    heldout_logits = centered[test_idx]
    heldout_ideal = ideal[test_idx]
    heldout_labels = labels[test_idx]
    false_mask = ~F.one_hot(heldout_labels, p).bool()
    actual_target = heldout_logits.gather(1, heldout_labels[:, None])
    ideal_target = heldout_ideal.gather(1, heldout_labels[:, None])
    actual_margins = actual_target - heldout_logits
    ideal_margins = ideal_target - heldout_ideal
    discrepancy = torch.abs(actual_margins - ideal_margins)
    margin_discrepancy_max = discrepancy[false_mask].max()
    actual_margin_min = actual_margins.masked_fill(~false_mask, torch.inf).min()
    ideal_margin_min = ideal_margins.masked_fill(~false_mask, torch.inf).min()

    addition = float(addition_coefficient.item())
    difference = float(difference_coefficient.item())
    kappa = addition + difference
    delta = float(margin_discrepancy_max.item())
    mixture_valid = addition > 0.0 and difference > 0.0
    interface_valid = mixture_valid and 0.0 <= delta < kappa
    tau = addition / kappa if mixture_valid else math.nan
    xi = math.log(addition / difference) if mixture_valid else math.nan
    if interface_valid:
        threshold = 2.0 * math.atanh(delta / kappa)
        bad_threshold = -threshold
        good_threshold = threshold
    else:
        bad_threshold = math.nan
        good_threshold = math.nan
    centered_norm = float(torch.linalg.matrix_norm(centered, ord="fro").item())
    residual_norm = float(torch.linalg.matrix_norm(residual, ord="fro").item())
    relative_residual = residual_norm / centered_norm if centered_norm > 0.0 else 0.0

    return {
        "rule_addition_coefficient": addition,
        "rule_difference_coefficient": difference,
        "rule_kappa": kappa,
        "rule_tau": tau,
        "rule_xi": xi,
        "rule_ideal_sum_vs_difference_margin": addition - difference,
        "rule_full_grid_relative_residual_fro": relative_residual,
        "rule_heldout_logit_residual_max": float(
            torch.abs(residual[test_idx]).max().item()
        ),
        "rule_heldout_margin_discrepancy_max": delta,
        "rule_heldout_actual_margin_min": float(actual_margin_min.item()),
        "rule_heldout_ideal_margin_min": float(ideal_margin_min.item()),
        "rule_margin_interface_valid": float(interface_valid),
        "rule_xi_bad_test_threshold": bad_threshold,
        "rule_xi_good_test_threshold": good_threshold,
    }


def _safe_tensor_cosine(first: torch.Tensor, second: torch.Tensor) -> float:
    first_flat = first.reshape(-1)
    second_flat = second.reshape(-1)
    denominator = (
        torch.linalg.vector_norm(first_flat)
        * torch.linalg.vector_norm(second_flat)
    )
    if float(denominator.item()) == 0.0:
        return 0.0
    cosine = torch.dot(first_flat, second_flat) / denominator
    return float(torch.clamp(cosine, min=-1.0, max=1.0).item())


@torch.no_grad()
def feature_induced_logit_velocity(
    model: QuadraticOneHiddenLayer,
    pairs: torch.Tensor,
    feature_direction: torch.Tensor,
) -> torch.Tensor:
    if feature_direction.shape != model.feature_weight.shape:
        raise ValueError("feature_direction must match feature_weight")
    if not bool(torch.isfinite(feature_direction).all()):
        raise ValueError("feature_direction must be finite")
    first = model.feature_weight[:, pairs[:, 0]].T
    second = model.feature_weight[:, model.p + pairs[:, 1]].T
    preactivation = first + second
    direction_first = feature_direction[:, pairs[:, 0]].T
    direction_second = feature_direction[:, model.p + pairs[:, 1]].T
    preactivation_velocity = direction_first + direction_second
    if model.activation == "square":
        activation_derivative = 2.0 * preactivation
    elif model.activation == "relu":
        activation_derivative = (preactivation > 0.0).to(preactivation.dtype)
    else:
        activation_derivative = torch.sigmoid(preactivation)
    hidden_velocity = activation_derivative * preactivation_velocity
    logit_velocity = (
        hidden_velocity @ model.head_weight.T / math.sqrt(model.width)
    )
    return logit_velocity - logit_velocity.mean(dim=1, keepdim=True)


@torch.no_grad()
def fourier_logit_velocity_projection(
    logit_velocity: torch.Tensor,
    p: int,
) -> dict[str, float]:
    if logit_velocity.ndim != 2 or logit_velocity.shape[0] != p * p:
        raise ValueError("logit_velocity must have one row per ordered input pair")
    if not bool(torch.isfinite(logit_velocity).all()):
        raise ValueError("logit_velocity must be finite")
    centered = logit_velocity - logit_velocity.mean(dim=0, keepdim=True)
    centered = centered - centered.mean(dim=1, keepdim=True)
    sum_basis, difference_basis, _ = real_fourier_bases(
        p, logit_velocity.device, logit_velocity.dtype
    )
    total_energy = centered.square().sum()
    sum_energy = (sum_basis.T @ centered).square().sum()
    difference_energy = (difference_basis.T @ centered).square().sum()
    if float(total_energy.item()) > 0.0:
        sum_fraction = sum_energy / total_energy
        difference_fraction = difference_energy / total_energy
    else:
        sum_fraction = torch.zeros_like(total_energy)
        difference_fraction = torch.zeros_like(total_energy)
    return {
        "fro": float(torch.sqrt(total_energy).item()),
        "fourier_sum_energy": float(sum_energy.item()),
        "fourier_difference_energy": float(difference_energy.item()),
        "fourier_sum_fraction": float(sum_fraction.item()),
        "fourier_difference_fraction": float(difference_fraction.item()),
        "fourier_sum_minus_difference_fraction": float(
            (sum_fraction - difference_fraction).item()
        ),
    }


def theory_gradient_diagnostics(
    model: QuadraticOneHiddenLayer,
    pairs: torch.Tensor,
    labels: torch.Tensor,
    train_idx: torch.Tensor,
    *,
    config: RunConfig | None = None,
    optimizer_state: EndToEndMatrixOptimizer | None = None,
) -> dict[str, float]:
    if pairs.ndim != 2 or pairs.shape != (model.p * model.p, 2):
        raise ValueError("pairs must contain the complete ordered p x p table")
    if labels.ndim != 1 or labels.shape[0] != pairs.shape[0]:
        raise ValueError("labels must contain one class per pair")
    if train_idx.ndim != 1 or train_idx.numel() < 1:
        raise ValueError("train_idx must be a nonempty vector")

    def raw_gradients(indices: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        with torch.enable_grad():
            loss = F.cross_entropy(model(pairs[indices]), labels[indices])
            feature_gradient, head_gradient = torch.autograd.grad(
                loss,
                (model.feature_weight, model.head_weight),
                create_graph=False,
                retain_graph=False,
            )
        return feature_gradient.detach(), head_gradient.detach()

    train_feature, train_head = raw_gradients(train_idx)
    full_idx = torch.arange(pairs.shape[0], device=pairs.device)
    full_feature, full_head = raw_gradients(full_idx)
    train_feature_effective = train_feature @ model.input_basis
    full_feature_effective = full_feature @ model.input_basis
    train_head_effective = model.output_basis.T @ train_head
    full_head_effective = model.output_basis.T @ full_head

    rule_target = F.one_hot(labels, model.p).to(model.feature_weight.dtype)
    rule_target = rule_target - 1.0 / model.p
    with torch.enable_grad():
        population_addition_correlation = (
            model(pairs) * rule_target
        ).sum() / (model.p * model.p)
        population_correlation_gradient = torch.autograd.grad(
            population_addition_correlation,
            model.feature_weight,
            create_graph=False,
            retain_graph=False,
        )[0].detach()
    feature_point_effective = model.feature_weight.detach() @ model.input_basis
    population_correlation_effective = (
        population_correlation_gradient @ model.input_basis
    )
    entry_j = rowwise_tangent_projection(
        population_correlation_effective, feature_point_effective
    )
    entry_k = -rowwise_tangent_projection(
        train_feature_effective, feature_point_effective
    )
    entry_polar_k, entry_polar_diagnostics = support_polar_factor(
        entry_k, strict_rank=False
    )
    entry_posttangent_polar_k = rowwise_tangent_projection(
        entry_polar_k, feature_point_effective
    )
    entry_rank = int(entry_polar_diagnostics["rank"])
    entry_k_nuclear_norm = float(torch.sum(entry_k * entry_polar_k).item())
    entry_j_minus_k_fro = float(
        torch.linalg.matrix_norm(entry_j - entry_k, ord="fro").item()
    )
    entry_margin_lower_bound = (
        entry_k_nuclear_norm
        - math.sqrt(entry_rank) * entry_j_minus_k_fro
    )
    entry_direct_cone = float(
        torch.sum(entry_j * entry_posttangent_polar_k).item()
    )

    train_feature_direction = train_feature_effective @ model.input_basis.T
    full_feature_direction = full_feature_effective @ model.input_basis.T
    train_velocity = feature_induced_logit_velocity(
        model, pairs, train_feature_direction
    )
    full_velocity = feature_induced_logit_velocity(
        model, pairs, full_feature_direction
    )
    train_velocity_projection = fourier_logit_velocity_projection(
        train_velocity, model.p
    )
    full_velocity_projection = fourier_logit_velocity_projection(
        full_velocity, model.p
    )
    centered_train_velocity = train_velocity - train_velocity.mean(
        dim=0, keepdim=True
    )
    centered_full_velocity = full_velocity - full_velocity.mean(
        dim=0, keepdim=True
    )
    result = {
        "theory_train_vs_full_feature_raw_gradient_cosine": _safe_tensor_cosine(
            train_feature_effective, full_feature_effective
        ),
        "theory_train_vs_full_head_raw_gradient_cosine": _safe_tensor_cosine(
            train_head_effective, full_head_effective
        ),
        "theory_train_feature_raw_gradient_fro": float(
            torch.linalg.matrix_norm(train_feature_effective, ord="fro").item()
        ),
        "theory_full_feature_raw_gradient_fro": float(
            torch.linalg.matrix_norm(full_feature_effective, ord="fro").item()
        ),
        "theory_train_head_raw_gradient_fro": float(
            torch.linalg.matrix_norm(train_head_effective, ord="fro").item()
        ),
        "theory_full_head_raw_gradient_fro": float(
            torch.linalg.matrix_norm(full_head_effective, ord="fro").item()
        ),
        "theory_train_vs_full_feature_logit_velocity_cosine": (
            _safe_tensor_cosine(
                centered_train_velocity, centered_full_velocity
            )
        ),
        "theory_train_raw_feature_update_rule_cosine": _safe_tensor_cosine(
            -centered_train_velocity, rule_target
        ),
        "theory_full_raw_feature_update_rule_cosine": _safe_tensor_cosine(
            -centered_full_velocity, rule_target
        ),
        "theory_finite_head_entry_k_nuclear_norm": entry_k_nuclear_norm,
        "theory_finite_head_entry_k_rank": entry_rank,
        "theory_finite_head_entry_j_minus_k_fro": entry_j_minus_k_fro,
        "theory_finite_head_entry_margin_lower_bound": entry_margin_lower_bound,
        "theory_finite_head_entry_direct_cone": entry_direct_cone,
        "theory_finite_head_entry_normalized_direct_cone": _safe_tensor_cosine(
            entry_j, entry_posttangent_polar_k
        ),
    }
    for source, projection in (
        ("train", train_velocity_projection),
        ("full", full_velocity_projection),
    ):
        for name, value in projection.items():
            result[f"theory_{source}_feature_logit_velocity_{name}"] = value

    if config is not None:
        feature_kind = {
            "polar_muon": "polar_muon",
            "fro_ngd": "fro_ngd",
            "polar_feature_gd_head": "polar_muon",
            "fro_feature_gd_head": "fro_ngd",
            "head_only": "gd",
            "gd": "gd",
        }[config.optimizer]
        def optimizer_feature_direction(
            raw_effective: torch.Tensor,
        ) -> torch.Tensor:
            pretransform = raw_effective
            if config.optimizer == "polar_muon" and config.momentum > 0.0:
                if optimizer_state is None:
                    raise ValueError(
                        "optimizer_state is required for momentum-aware theory diagnostics"
                    )
                current_momentum = (
                    optimizer_state.feature_momentum @ model.input_basis
                )
                pretransform = (
                    config.momentum * current_momentum
                    + (1.0 - config.momentum) * raw_effective
                )
            if config.feature_geometry == "row_sphere":
                pretransform = rowwise_tangent_projection(
                    pretransform, feature_point_effective
                )
            direction, _ = transformed_direction(
                pretransform,
                feature_kind,
                strict_rank=False,
            )
            if config.optimizer == "head_only":
                direction = torch.zeros_like(direction)
            return direction

        train_optimizer_direction = optimizer_feature_direction(
            train_feature_effective
        )
        full_optimizer_direction = optimizer_feature_direction(
            full_feature_effective
        )
        if config.feature_geometry == "row_sphere":
            train_optimizer_posttangent_direction = rowwise_tangent_projection(
                train_optimizer_direction, feature_point_effective
            )
            full_optimizer_posttangent_direction = rowwise_tangent_projection(
                full_optimizer_direction, feature_point_effective
            )
        else:
            train_optimizer_posttangent_direction = train_optimizer_direction
            full_optimizer_posttangent_direction = full_optimizer_direction
        train_optimizer_velocity = feature_induced_logit_velocity(
            model,
            pairs,
            train_optimizer_direction @ model.input_basis.T,
        )
        full_optimizer_velocity = feature_induced_logit_velocity(
            model,
            pairs,
            full_optimizer_direction @ model.input_basis.T,
        )
        train_optimizer_posttangent_velocity = feature_induced_logit_velocity(
            model,
            pairs,
            train_optimizer_posttangent_direction @ model.input_basis.T,
        )
        full_optimizer_posttangent_velocity = feature_induced_logit_velocity(
            model,
            pairs,
            full_optimizer_posttangent_direction @ model.input_basis.T,
        )
        train_optimizer_centered = (
            train_optimizer_velocity
            - train_optimizer_velocity.mean(dim=0, keepdim=True)
        )
        full_optimizer_centered = (
            full_optimizer_velocity
            - full_optimizer_velocity.mean(dim=0, keepdim=True)
        )
        train_optimizer_posttangent_centered = (
            train_optimizer_posttangent_velocity
            - train_optimizer_posttangent_velocity.mean(dim=0, keepdim=True)
        )
        full_optimizer_posttangent_centered = (
            full_optimizer_posttangent_velocity
            - full_optimizer_posttangent_velocity.mean(dim=0, keepdim=True)
        )
        result.update(
            {
                "theory_train_vs_full_optimizer_feature_direction_cosine": (
                    _safe_tensor_cosine(
                        train_optimizer_direction, full_optimizer_direction
                    )
                ),
                "theory_train_vs_full_optimizer_feature_logit_velocity_cosine": (
                    _safe_tensor_cosine(
                        train_optimizer_centered, full_optimizer_centered
                    )
                ),
                "theory_train_optimizer_feature_update_rule_cosine": (
                    _safe_tensor_cosine(-train_optimizer_centered, rule_target)
                ),
                "theory_full_optimizer_feature_update_rule_cosine": (
                    _safe_tensor_cosine(-full_optimizer_centered, rule_target)
                ),
                "theory_train_optimizer_feature_direction_fro": float(
                    torch.linalg.matrix_norm(
                        train_optimizer_direction, ord="fro"
                    ).item()
                ),
                "theory_full_optimizer_feature_direction_fro": float(
                    torch.linalg.matrix_norm(
                        full_optimizer_direction, ord="fro"
                    ).item()
                ),
                "theory_train_vs_full_optimizer_posttangent_feature_direction_cosine": (
                    _safe_tensor_cosine(
                        train_optimizer_posttangent_direction,
                        full_optimizer_posttangent_direction,
                    )
                ),
                "theory_train_vs_full_optimizer_posttangent_feature_logit_velocity_cosine": (
                    _safe_tensor_cosine(
                        train_optimizer_posttangent_centered,
                        full_optimizer_posttangent_centered,
                    )
                ),
                "theory_train_optimizer_posttangent_feature_update_rule_cosine": (
                    _safe_tensor_cosine(
                        -train_optimizer_posttangent_centered, rule_target
                    )
                ),
                "theory_full_optimizer_posttangent_feature_update_rule_cosine": (
                    _safe_tensor_cosine(
                        -full_optimizer_posttangent_centered, rule_target
                    )
                ),
                "theory_train_optimizer_posttangent_feature_direction_fro": float(
                    torch.linalg.matrix_norm(
                        train_optimizer_posttangent_direction, ord="fro"
                    ).item()
                ),
                "theory_full_optimizer_posttangent_feature_direction_fro": float(
                    torch.linalg.matrix_norm(
                        full_optimizer_posttangent_direction, ord="fro"
                    ).item()
                ),
            }
        )
        for source, projection in (
            (
                "train",
                fourier_logit_velocity_projection(
                    train_optimizer_velocity, model.p
                ),
            ),
            (
                "full",
                fourier_logit_velocity_projection(
                    full_optimizer_velocity, model.p
                ),
            ),
        ):
            for name, value in projection.items():
                result[
                    f"theory_{source}_optimizer_feature_logit_velocity_{name}"
                ] = value
        for source, projection in (
            (
                "train",
                fourier_logit_velocity_projection(
                    train_optimizer_posttangent_velocity, model.p
                ),
            ),
            (
                "full",
                fourier_logit_velocity_projection(
                    full_optimizer_posttangent_velocity, model.p
                ),
            ),
        ):
            for name, value in projection.items():
                result[
                    f"theory_{source}_optimizer_posttangent_feature_logit_velocity_{name}"
                ] = value
    return result


def real_fourier_bases(
    p: int,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if p < 3 or p % 2 == 0:
        raise ValueError("the real Fourier code requires an odd modulus")
    labels = torch.arange(p, device=device, dtype=dtype)
    code_columns = []
    for frequency in range(1, (p + 1) // 2):
        angle = (2.0 * math.pi * frequency / p) * labels
        code_columns.extend((torch.cos(angle), torch.sin(angle)))
    codebook = torch.stack(code_columns, dim=1)
    first = labels.repeat_interleave(p)
    second = labels.repeat(p)
    sum_labels = torch.remainder(first + second, p).long()
    difference_labels = torch.remainder(first - second, p).long()
    normalization = math.sqrt(2.0 / (p * p))
    return (
        normalization * codebook[sum_labels],
        normalization * codebook[difference_labels],
        codebook,
    )


def numerical_column_basis(
    matrix: torch.Tensor,
    *,
    relative_tolerance: float | None = None,
) -> torch.Tensor:
    if matrix.ndim != 2 or not matrix.is_floating_point():
        raise TypeError("matrix must be a floating-point matrix")
    u, singular, _ = torch.linalg.svd(matrix, full_matrices=False)
    if not singular.numel():
        return u[:, :0]
    if relative_tolerance is None:
        cutoff = (
            torch.finfo(matrix.dtype).eps
            * max(matrix.shape)
            * singular[0].clamp_min(torch.finfo(matrix.dtype).tiny)
        )
    else:
        if not math.isfinite(relative_tolerance) or relative_tolerance < 0.0:
            raise ValueError("relative_tolerance must be finite and nonnegative")
        cutoff = relative_tolerance * singular[0].clamp_min(1.0e-300)
    return u[:, singular > cutoff]


@torch.no_grad()
def hidden_state_agop(head_weight: torch.Tensor) -> torch.Tensor:
    if (
        head_weight.ndim != 2
        or not head_weight.is_floating_point()
        or not bool(torch.isfinite(head_weight).all())
    ):
        raise TypeError("head_weight must be a finite floating-point matrix")
    width = head_weight.shape[1]
    if width < 1:
        raise ValueError("head_weight must have at least one hidden coordinate")
    centered_head = head_weight - head_weight.mean(dim=0, keepdim=True)
    agop = centered_head.T @ centered_head / width
    return 0.5 * (agop + agop.T)


def resolved_agop_ranks(
    p: int,
    width: int,
    ranks: Sequence[int] = (),
) -> tuple[int, ...]:
    maximum = min(p - 1, width)
    if maximum < 1:
        raise ValueError("p and width leave no identifiable AGOP direction")
    if ranks:
        if any(int(rank) != rank or not 1 <= int(rank) <= maximum for rank in ranks):
            raise ValueError(f"AGOP ranks must be integers in [1, {maximum}]")
        candidates = [int(rank) for rank in ranks]
    else:
        candidates = [min(rank, maximum) for rank in (4, 8, 16)]
    candidates.append(maximum)
    return tuple(sorted(set(candidates)))


def _ridge_predict(
    x_fit: torch.Tensor,
    y_fit: torch.Tensor,
    x_evaluate: torch.Tensor,
    alpha: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    mean = x_fit.mean(dim=0)
    scale = x_fit.std(dim=0, unbiased=False).clamp_min(1.0e-10)
    fit = (x_fit - mean) / scale
    evaluate = (x_evaluate - mean) / scale
    target_mean = y_fit.mean(dim=0, keepdim=True)
    covariance = fit.T @ fit / len(fit)
    rhs = fit.T @ (y_fit - target_mean) / len(fit)
    identity = torch.eye(
        covariance.shape[0], device=x_fit.device, dtype=x_fit.dtype
    )
    weights = torch.linalg.solve(covariance + float(alpha) * identity, rhs)
    return evaluate @ weights + target_mean, target_mean


def _coefficient_of_determination(
    prediction: torch.Tensor,
    target: torch.Tensor,
    baseline: torch.Tensor,
) -> torch.Tensor:
    denominator = (target - baseline).square().sum().clamp_min(1.0e-20)
    return 1.0 - (prediction - target).square().sum() / denominator


def _train_only_fourier_probe(
    features: torch.Tensor,
    labels: torch.Tensor,
    train_idx: torch.Tensor,
    test_idx: torch.Tensor,
    codebook: torch.Tensor,
    ridge_alphas: Sequence[float],
    folds: int,
    seed: int,
) -> dict[str, float]:
    x_train, x_test = features[train_idx], features[test_idx]
    targets = codebook[labels]
    y_train, y_test = targets[train_idx], targets[test_idx]
    if len(x_train) < 1 or len(x_test) < 1:
        raise ValueError("the Fourier probe requires nonempty train and test sets")

    alphas = tuple(float(alpha) for alpha in ridge_alphas)
    if not alphas or any(not math.isfinite(alpha) or alpha <= 0.0 for alpha in alphas):
        raise ValueError("ridge_alphas must be finite and positive")
    if folds < 2:
        raise ValueError("folds must be at least two")

    if len(x_train) == 1:
        selected_alpha = max(alphas)
        selected_score = math.nan
    else:
        effective_folds = min(folds, len(x_train))
        permutation = np.random.default_rng(seed).permutation(len(x_train))
        fold_ids_numpy = np.empty(len(x_train), dtype=np.int64)
        fold_ids_numpy[permutation] = np.arange(len(x_train)) % effective_folds
        fold_ids = torch.as_tensor(
            fold_ids_numpy, device=features.device, dtype=torch.long
        )
        scores = []
        for alpha in alphas:
            fold_scores = []
            for fold in range(effective_folds):
                fit = fold_ids != fold
                validation = fold_ids == fold
                prediction, baseline = _ridge_predict(
                    x_train[fit], y_train[fit], x_train[validation], alpha
                )
                fold_scores.append(
                    float(
                        _coefficient_of_determination(
                            prediction, y_train[validation], baseline
                        ).item()
                    )
                )
            scores.append(float(np.mean(fold_scores)))
        selected = max(
            range(len(alphas)), key=lambda index: (scores[index], alphas[index])
        )
        selected_alpha = alphas[selected]
        selected_score = scores[selected]

    prediction, baseline = _ridge_predict(
        x_train, y_train, x_test, selected_alpha
    )
    predicted_labels = torch.argmax(prediction @ codebook.T, dim=1)
    return {
        "heldout_fourier_r2": float(
            _coefficient_of_determination(prediction, y_test, baseline).item()
        ),
        "heldout_modular_accuracy": float(
            (predicted_labels == labels[test_idx]).to(features.dtype).mean().item()
        ),
        "ridge_alpha": float(selected_alpha),
        "inner_cv_r2": float(selected_score),
    }


def _subspace_purity_coverage(
    features: torch.Tensor,
    targets: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, int, int]:
    feature_basis = numerical_column_basis(
        features - features.mean(dim=0, keepdim=True),
        relative_tolerance=1.0e-10,
    )
    target_basis = numerical_column_basis(
        targets - targets.mean(dim=0, keepdim=True),
        relative_tolerance=1.0e-10,
    )
    feature_rank = feature_basis.shape[1]
    target_rank = target_basis.shape[1]
    zero = torch.zeros((), device=features.device, dtype=features.dtype)
    if feature_rank == 0 or target_rank == 0:
        return zero, zero, feature_rank, target_rank
    overlap = (feature_basis.T @ target_basis).square().sum()
    purity = torch.clamp(overlap / min(feature_rank, target_rank), 0.0, 1.0)
    coverage = torch.clamp(overlap / target_rank, 0.0, 1.0)
    return purity, coverage, feature_rank, target_rank


def _stable_diagnostic_seed(*values: int) -> int:
    payload = json.dumps(tuple(int(value) for value in values)).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little") % (2**63 - 1)


def _random_thin_basis(
    width: int,
    rank: int,
    *,
    seed: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    matrix = torch.randn(
        (width, rank), generator=generator, device=device, dtype=dtype
    )
    return torch.linalg.qr(matrix, mode="reduced").Q


@torch.no_grad()
def agop_fourier_alignment(
    hidden: torch.Tensor,
    head_weight: torch.Tensor,
    p: int,
    labels: torch.Tensor,
    train_idx: torch.Tensor,
    test_idx: torch.Tensor,
    *,
    ranks: Sequence[int] = (),
    ridge_alphas: Sequence[float] = (1.0e-5, 1.0e-4, 1.0e-3, 1.0e-2, 1.0e-1, 1.0),
    probe_folds: int = 3,
    probe_seed: int = 314159,
    random_controls: int = 4,
) -> dict[str, float | int]:
    if hidden.ndim != 2 or hidden.shape[0] != p * p:
        raise ValueError("hidden must contain the complete ordered p x p domain")
    if head_weight.shape != (p, hidden.shape[1]):
        raise ValueError("head_weight must have shape (p, hidden_width)")
    if labels.shape != (p * p,) or labels.dtype != torch.long:
        raise ValueError("labels must be a length-p^2 torch.long tensor")
    if any(
        tensor.device != hidden.device
        for tensor in (head_weight, labels, train_idx, test_idx)
    ):
        raise ValueError("all AGOP observer tensors must be on the same device")
    if head_weight.dtype != hidden.dtype:
        raise ValueError("hidden and head_weight must use the same dtype")
    if random_controls < 1:
        raise ValueError("random_controls must be positive")

    centered_hidden = hidden - hidden.mean(dim=0, keepdim=True)
    sum_basis, difference_basis, codebook = real_fourier_bases(
        p, hidden.device, hidden.dtype
    )
    agop = hidden_state_agop(head_weight)
    eigenvalues, eigenvectors = torch.linalg.eigh(agop)
    eigenvalues = eigenvalues.clamp_min(0.0)
    descending = eigenvalues.flip(0)
    trace = descending.sum()
    spectral_norm = descending[0] if descending.numel() else trace
    if float(spectral_norm.item()) > 0.0:
        rank_cutoff = (
            torch.finfo(hidden.dtype).eps
            * max(agop.shape)
            * spectral_norm
        )
        numerical_rank = int(torch.count_nonzero(descending > rank_cutoff).item())
    else:
        numerical_rank = 0

    rule_rank = min(p - 1, hidden.shape[1])
    resolved_ranks = resolved_agop_ranks(p, hidden.shape[1], ranks)
    full_hidden_probe = _train_only_fourier_probe(
        centered_hidden,
        labels,
        train_idx,
        test_idx,
        codebook,
        ridge_alphas,
        probe_folds,
        probe_seed,
    )
    result: dict[str, float | int] = {
        "agop_trace": float(trace.item()),
        "agop_spectral_norm": float(spectral_norm.item()),
        "agop_numerical_rank": numerical_rank,
        "agop_rule_rank": rule_rank,
        "agop_random_control_count": int(random_controls),
        "full_hidden_train_only_refit_heldout_modular_accuracy": (
            full_hidden_probe["heldout_modular_accuracy"]
        ),
        "full_hidden_train_only_refit_heldout_fourier_r2": (
            full_hidden_probe["heldout_fourier_r2"]
        ),
        "full_hidden_train_only_refit_ridge_alpha": full_hidden_probe[
            "ridge_alpha"
        ],
        "full_hidden_train_only_refit_inner_cv_r2": full_hidden_probe[
            "inner_cv_r2"
        ],
    }

    for rank in resolved_ranks:
        prefix = f"agop_top{rank}"
        selected_vectors = eigenvectors[:, -rank:]
        selected_features = centered_hidden @ selected_vectors
        (
            sum_purity,
            sum_coverage,
            heldout_selected_rank,
            heldout_sum_rank,
        ) = _subspace_purity_coverage(
            selected_features[test_idx], sum_basis[test_idx]
        )
        (
            difference_purity,
            difference_coverage,
            heldout_difference_selected_rank,
            heldout_difference_rank,
        ) = _subspace_purity_coverage(
            selected_features[test_idx], difference_basis[test_idx]
        )
        if heldout_selected_rank != heldout_difference_selected_rank:
            raise AssertionError("held-out feature rank cannot depend on the target")
        (
            full_sum_purity,
            full_sum_coverage,
            full_selected_rank,
            full_sum_rank,
        ) = _subspace_purity_coverage(selected_features, sum_basis)
        (
            full_difference_purity,
            full_difference_coverage,
            full_difference_selected_rank,
            full_difference_rank,
        ) = _subspace_purity_coverage(selected_features, difference_basis)
        if full_selected_rank != full_difference_selected_rank:
            raise AssertionError("full-grid feature rank cannot depend on the target")
        top_energy = descending[:rank].sum()
        energy_fraction = (
            top_energy / trace
            if float(trace.item()) > 0.0
            else torch.zeros((), device=hidden.device, dtype=hidden.dtype)
        )
        next_eigenvalue = (
            descending[rank]
            if rank < descending.numel()
            else torch.zeros((), device=hidden.device, dtype=hidden.dtype)
        )
        eigengap = torch.clamp(descending[rank - 1] - next_eigenvalue, min=0.0)
        relative_eigengap = (
            eigengap / spectral_norm
            if float(spectral_norm.item()) > 0.0
            else torch.zeros((), device=hidden.device, dtype=hidden.dtype)
        )
        probe = _train_only_fourier_probe(
            selected_features,
            labels,
            train_idx,
            test_idx,
            codebook,
            ridge_alphas,
            probe_folds,
            probe_seed,
        )
        random_rows = []
        for replicate in range(random_controls):
            random_vectors = _random_thin_basis(
                hidden.shape[1],
                rank,
                seed=_stable_diagnostic_seed(probe_seed, rank, replicate),
                device=hidden.device,
                dtype=hidden.dtype,
            )
            random_features = centered_hidden @ random_vectors
            (
                random_sum_purity,
                random_sum_coverage,
                _,
                _,
            ) = _subspace_purity_coverage(
                random_features[test_idx], sum_basis[test_idx]
            )
            (
                random_difference_purity,
                random_difference_coverage,
                _,
                _,
            ) = _subspace_purity_coverage(
                random_features[test_idx], difference_basis[test_idx]
            )
            random_probe = _train_only_fourier_probe(
                random_features,
                labels,
                train_idx,
                test_idx,
                codebook,
                ridge_alphas,
                probe_folds,
                probe_seed,
            )
            random_rows.append(
                {
                    "fourier_sum_purity": float(random_sum_purity.item()),
                    "fourier_sum_coverage": float(random_sum_coverage.item()),
                    "fourier_difference_purity": float(
                        random_difference_purity.item()
                    ),
                    "fourier_difference_coverage": float(
                        random_difference_coverage.item()
                    ),
                    "heldout_fourier_r2": random_probe["heldout_fourier_r2"],
                    "heldout_modular_accuracy": random_probe[
                        "heldout_modular_accuracy"
                    ],
                }
            )

        def random_mean(name: str) -> float:
            return float(np.mean([row[name] for row in random_rows]))

        def random_std(name: str) -> float:
            return float(np.std([row[name] for row in random_rows], ddof=0))

        random_sum_purity_mean = random_mean("fourier_sum_purity")
        random_sum_coverage_mean = random_mean("fourier_sum_coverage")
        random_r2_mean = random_mean("heldout_fourier_r2")
        random_accuracy_mean = random_mean("heldout_modular_accuracy")
        metrics: dict[str, float | int] = {
            f"{prefix}_energy": float(top_energy.item()),
            f"{prefix}_energy_fraction": float(energy_fraction.item()),
            f"{prefix}_eigengap": float(eigengap.item()),
            f"{prefix}_relative_eigengap": float(relative_eigengap.item()),
            f"{prefix}_selected_feature_rank": int(heldout_selected_rank),
            f"{prefix}_heldout_selected_feature_rank": int(heldout_selected_rank),
            f"{prefix}_heldout_fourier_sum_target_rank": int(heldout_sum_rank),
            f"{prefix}_heldout_fourier_difference_target_rank": int(
                heldout_difference_rank
            ),
            f"{prefix}_fourier_sum_purity": float(sum_purity.item()),
            f"{prefix}_fourier_sum_coverage": float(sum_coverage.item()),
            f"{prefix}_fourier_difference_purity": float(difference_purity.item()),
            f"{prefix}_fourier_difference_coverage": float(
                difference_coverage.item()
            ),
            f"{prefix}_fourier_sum_minus_difference_purity": float(
                (sum_purity - difference_purity).item()
            ),
            f"{prefix}_heldout_fourier_sum_purity": float(sum_purity.item()),
            f"{prefix}_heldout_fourier_sum_coverage": float(sum_coverage.item()),
            f"{prefix}_heldout_fourier_difference_purity": float(
                difference_purity.item()
            ),
            f"{prefix}_heldout_fourier_difference_coverage": float(
                difference_coverage.item()
            ),
            f"{prefix}_full_grid_selected_feature_rank": int(full_selected_rank),
            f"{prefix}_full_grid_fourier_sum_target_rank": int(full_sum_rank),
            f"{prefix}_full_grid_fourier_difference_target_rank": int(
                full_difference_rank
            ),
            f"{prefix}_full_grid_fourier_sum_purity": float(
                full_sum_purity.item()
            ),
            f"{prefix}_full_grid_fourier_sum_coverage": float(
                full_sum_coverage.item()
            ),
            f"{prefix}_full_grid_fourier_difference_purity": float(
                full_difference_purity.item()
            ),
            f"{prefix}_full_grid_fourier_difference_coverage": float(
                full_difference_coverage.item()
            ),
            f"{prefix}_heldout_fourier_r2": probe["heldout_fourier_r2"],
            f"{prefix}_heldout_modular_accuracy": probe[
                "heldout_modular_accuracy"
            ],
            f"{prefix}_ridge_alpha": probe["ridge_alpha"],
            f"{prefix}_inner_cv_r2": probe["inner_cv_r2"],
            f"{prefix}_random_mean_fourier_sum_purity": random_sum_purity_mean,
            f"{prefix}_random_std_fourier_sum_purity": random_std(
                "fourier_sum_purity"
            ),
            f"{prefix}_random_mean_fourier_sum_coverage": random_sum_coverage_mean,
            f"{prefix}_random_std_fourier_sum_coverage": random_std(
                "fourier_sum_coverage"
            ),
            f"{prefix}_random_mean_fourier_difference_purity": random_mean(
                "fourier_difference_purity"
            ),
            f"{prefix}_random_std_fourier_difference_purity": random_std(
                "fourier_difference_purity"
            ),
            f"{prefix}_random_mean_fourier_difference_coverage": random_mean(
                "fourier_difference_coverage"
            ),
            f"{prefix}_random_std_fourier_difference_coverage": random_std(
                "fourier_difference_coverage"
            ),
            f"{prefix}_random_mean_heldout_fourier_r2": random_r2_mean,
            f"{prefix}_random_std_heldout_fourier_r2": random_std(
                "heldout_fourier_r2"
            ),
            f"{prefix}_random_mean_heldout_modular_accuracy": random_accuracy_mean,
            f"{prefix}_random_std_heldout_modular_accuracy": random_std(
                "heldout_modular_accuracy"
            ),
            f"{prefix}_top_minus_random_fourier_sum_purity": float(
                sum_purity.item()
            )
            - random_sum_purity_mean,
            f"{prefix}_top_minus_random_fourier_sum_coverage": float(
                sum_coverage.item()
            )
            - random_sum_coverage_mean,
            f"{prefix}_top_minus_random_heldout_fourier_r2": probe[
                "heldout_fourier_r2"
            ]
            - random_r2_mean,
            f"{prefix}_top_minus_random_heldout_modular_accuracy": probe[
                "heldout_modular_accuracy"
            ]
            - random_accuracy_mean,
        }
        result.update(metrics)
        if rank == rule_rank:
            for name, value in metrics.items():
                suffix = name[len(prefix) :]
                result[f"agop_top_rule{suffix}"] = value
    return result


@torch.no_grad()
def fourier_alignment(hidden: torch.Tensor, p: int) -> dict[str, float | int]:
    if hidden.ndim != 2 or hidden.shape[0] != p * p:
        raise ValueError("hidden must contain the complete ordered p x p domain")
    if p < 3 or p % 2 == 0:
        raise ValueError("the real Fourier code requires an odd modulus")
    centered = hidden - hidden.mean(dim=0, keepdim=True)
    hidden_basis = numerical_column_basis(centered)
    sum_basis, difference_basis, _ = real_fourier_bases(
        p, hidden.device, hidden.dtype
    )

    def subspace_fraction(target_basis: torch.Tensor) -> torch.Tensor:
        if hidden_basis.shape[1] == 0:
            return torch.zeros((), device=hidden.device, dtype=hidden.dtype)
        cross = hidden_basis.T @ target_basis
        return torch.clamp(cross.square().sum() / (p - 1), 0.0, 1.0)

    sum_subspace = subspace_fraction(sum_basis)
    difference_subspace = subspace_fraction(difference_basis)

    field = centered.reshape(p, p, hidden.shape[1])
    coefficients = torch.fft.fft2(field, dim=(0, 1), norm="ortho")
    power = coefficients.abs().square()
    aggregate = power.sum(dim=2)
    aggregate[0, 0] = 0.0
    total = aggregate.sum()
    frequencies = torch.arange(1, p, device=hidden.device)
    sum_power_by_frequency = aggregate[frequencies, frequencies]
    difference_power_by_frequency = aggregate[frequencies, (-frequencies) % p]
    sum_power = sum_power_by_frequency.sum()
    difference_power = difference_power_by_frequency.sum()

    per_neuron_total = power.sum(dim=(0, 1)) - power[0, 0]
    per_neuron_sum = power[frequencies, frequencies].sum(dim=0)
    valid = per_neuron_total > 0.0
    per_neuron_fraction = torch.where(
        valid,
        per_neuron_sum / per_neuron_total.clamp_min(torch.finfo(hidden.dtype).tiny),
        torch.zeros_like(per_neuron_total),
    )
    if float(sum_power.item()) > 0.0:
        distribution = sum_power_by_frequency / sum_power
        positive = distribution > 0.0
        entropy = -(distribution[positive] * torch.log(distribution[positive])).sum()
        normalized_effective_frequency_count = torch.clamp(
            torch.exp(entropy) / (p - 1), 0.0, 1.0
        )
    else:
        normalized_effective_frequency_count = torch.zeros((), device=hidden.device)
    denominator = total.clamp_min(torch.finfo(hidden.dtype).tiny)
    addition_fraction = torch.clamp(sum_power / denominator, 0.0, 1.0)
    difference_fraction = torch.clamp(difference_power / denominator, 0.0, 1.0)
    covered_fraction = torch.clamp(
        addition_fraction * normalized_effective_frequency_count, 0.0, 1.0
    )
    return {
        "fourier_sum_full_hidden_span_coverage": float(sum_subspace.item()),
        "fourier_difference_full_hidden_span_coverage": float(
            difference_subspace.item()
        ),
        "fourier_sum_oracle_headfit_residual": float((1.0 - sum_subspace).item()),
        "fourier_sum_subspace_alignment": float(sum_subspace.item()),
        "fourier_difference_subspace_alignment": float(difference_subspace.item()),
        "fourier_sum_headfit_risk": float((1.0 - sum_subspace).item()),
        "fourier_sum_minus_difference_alignment": float(
            (sum_subspace - difference_subspace).item()
        ),
        "hidden_numerical_rank": int(hidden_basis.shape[1]),
        "fourier_addition_fraction": float(addition_fraction.item()),
        "fourier_difference_fraction": float(difference_fraction.item()),
        "fourier_addition_per_neuron_mean": float(per_neuron_fraction.mean().item()),
        "fourier_addition_per_neuron_median": float(per_neuron_fraction.median().item()),
        "fourier_addition_frequency_coverage": float(normalized_effective_frequency_count.item()),
        "fourier_addition_covered_fraction": float(covered_fraction.item()),
        "hidden_centered_energy": float(total.item()),
    }


def first_crossing(rows: Iterable[dict[str, Any]], field: str, threshold: float) -> int | None:
    for row in rows:
        if float(row[field]) >= threshold:
            return int(row["step"])
    return None


def summarize(rows: list[dict[str, Any]], config: RunConfig) -> dict[str, Any]:
    train_fit = first_crossing(rows, "train_accuracy", config.train_accuracy_threshold)
    test_generalization = first_crossing(rows, "test_accuracy", config.test_accuracy_threshold)
    alignment = first_crossing(
        rows, "fourier_sum_subspace_alignment", config.alignment_threshold
    )
    agop_alignment = first_crossing(
        rows, "agop_top_rule_fourier_sum_purity", config.alignment_threshold
    )
    delay = (
        int(test_generalization - train_fit)
        if train_fit is not None and test_generalization is not None
        else None
    )
    train_plateau_ok = False
    if train_fit is not None and test_generalization is not None and test_generalization >= train_fit:
        interval = [
            row
            for row in rows
            if train_fit <= int(row["step"]) <= test_generalization
        ]
        train_plateau_ok = bool(interval) and all(
            float(row["train_accuracy"]) >= config.train_accuracy_threshold for row in interval
        )
    grokking_observed = bool(
        delay is not None
        and delay >= config.minimum_delay_steps
        and train_plateau_ok
    )
    return {
        "train_fit_step": train_fit,
        "test_generalization_step": test_generalization,
        "fourier_alignment_step": alignment,
        "agop_fourier_alignment_step": agop_alignment,
        "fit_to_generalization_delay": delay,
        "train_plateau_preserved": train_plateau_ok,
        "grokking_observed_under_declared_thresholds": grokking_observed,
        "final_train_ce": float(rows[-1]["train_ce"]),
        "final_test_ce": float(rows[-1]["test_ce"]),
        "final_train_accuracy": float(rows[-1]["train_accuracy"]),
        "final_test_accuracy": float(rows[-1]["test_accuracy"]),
        "final_fourier_sum_subspace_alignment": float(
            rows[-1]["fourier_sum_subspace_alignment"]
        ),
        "final_fourier_sum_headfit_risk": float(rows[-1]["fourier_sum_headfit_risk"]),
        "maximum_fourier_sum_subspace_alignment": float(
            max(float(row["fourier_sum_subspace_alignment"]) for row in rows)
        ),
        "agop_rule_rank": int(rows[-1]["agop_rule_rank"]),
        "final_agop_top_rule_fourier_sum_purity": float(
            rows[-1]["agop_top_rule_fourier_sum_purity"]
        ),
        "final_agop_top_rule_fourier_sum_coverage": float(
            rows[-1]["agop_top_rule_fourier_sum_coverage"]
        ),
        "maximum_agop_top_rule_fourier_sum_purity": float(
            max(float(row["agop_top_rule_fourier_sum_purity"]) for row in rows)
        ),
        "final_agop_top_rule_heldout_fourier_r2": float(
            rows[-1]["agop_top_rule_heldout_fourier_r2"]
        ),
        "final_agop_top_rule_heldout_modular_accuracy": float(
            rows[-1]["agop_top_rule_heldout_modular_accuracy"]
        ),
        "final_fourier_addition_fraction": float(rows[-1]["fourier_addition_fraction"]),
        "maximum_fourier_addition_fraction": float(
            max(float(row["fourier_addition_fraction"]) for row in rows)
        ),
    }


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _write_outputs(
    output_dir: Path,
    config: RunConfig,
    data: ModularData,
    rows: list[dict[str, Any]],
    summary: dict[str, Any],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=False)
    payload = {
        "schema_version": 1,
        "status": "complete",
        "config": asdict(config),
        "task": {
            "operation": "a+b mod p",
            "prime_modulus": _is_prime(config.p),
            "examples": config.p * config.p,
            "train_examples": len(data.train_idx_numpy),
            "test_examples": len(data.test_idx_numpy),
            "realized_train_fraction": len(data.train_idx_numpy) / (config.p * config.p),
            "split_fingerprint": data.split_fingerprint,
        },
        "optimizer_semantics": {
            "joint_preupdate_gradient": True,
            "feature_and_head_both_trained": config.optimizer != "head_only",
            "feature_tensor_frozen": config.optimizer == "head_only",
            "activation": config.activation,
            "feature_geometry": config.feature_geometry,
            "polar": "compact support-polar U_r V_r^T in gauge-free coordinates",
            "fro_ngd": "layerwise raw gradient rescaled to the support-polar Frobenius norm",
            "mixed_modes": "feature matrix normalized; classifier head uses raw gradient descent",
            "head_only": (
                "feature tensor bitwise frozen; classifier head uses projected raw "
                "gradient descent and decoupled head decay"
            ),
            "weight_decay": "decoupled",
            "adaptive_head_decay": (
                "disabled when adaptive_balance_start_step=-1; otherwise "
                "the declared start is the completed warmup state T_P; "
                "ordinary numbered updates 1 through T_P produce theta_T_P, "
                "and adaptive balancing starts on numbered update T_P+1, "
                "which maps theta_T_P to theta_{T_P+1}; earlier updates use "
                "weight_decay_head, and "
                "subsequent row-sphere Polar-feature/GD-head updates choose "
                "lambda=(||G_V||_F^2+(lr_feature/lr_head)<H,Q_pol>)/"
                "(-<G_V,V>) when the denominator exceeds the declared floor; "
                "<H,Q_pol> is the retained finite-precision support-polar "
                "nuclear sum; "
                "the deterministic fallback is weight_decay_head and optional "
                "min/max clipping is applied only in the active phase; only "
                "the current training-loss gradient enters this choice"
            ),
            "head_balance_diagnostics": (
                "head_update_fro is the loss-direction component; "
                "head_decay_update_fro is the decay component; "
                "head_actual_step_fro and head_combined_direction_fro include "
                "their vector cancellation, whose angle is recorded by "
                "head_direction_decay_cosine"
            ),
            "adaptive_balance_residual": (
                "||G_V||_F^2+(lr_feature/lr_head)<H,Q_pol>+"
                "lambda<G_V,V>; its negative lr_head multiple equals the "
                "reported total pre-retraction signed first-order CE change "
                "for the supported adaptive optimizer"
            ),
            "signed_first_order_ce_diagnostics": (
                "the feature term pairs the current row-tangent CE gradient "
                "with the pre-retraction transformed feature step; the head "
                "term pairs the current CE data gradient with the combined "
                "data-plus-decay direction and includes lr_head; their sum is "
                "total_signed_first_order_ce_change. The separately reported "
                "feature_retraction_gradient_inner_product adds the exact "
                "row-sphere retraction contribution; "
                "full_exact_gradient_increment_inner_product pairs the full "
                "pre-update gradient with the exact two-block parameter "
                "increment. For adaptive-enabled runs, the runner also reports "
                "the realized one-step training-CE change, its Taylor remainder "
                "after subtracting that exact first-order inner product, and "
                "adaptive-phase cumulative absolute sums of all three terms"
            ),
            "adaptive_all_step_mechanism_counters": (
                "for adaptive-enabled runs, every optimizer update contributes "
                "to state/update-index and phase-boundary mismatch counts; every "
                "active update contributes to active, formula, fallback, and "
                "clipping counts plus running minimum denominator, maximum "
                "absolute balance residual, and a nonfinite-diagnostic count. "
                "These cumulative fields certify intervening updates that are "
                "not individually retained by checkpoint logging"
            ),
            "row_sphere": (
                "rowwise raw-gradient tangent projection and transformed-direction "
                "retraction at fixed initialized norms in identifiable coordinates"
            ),
        },
        "observer_semantics": {
            "hidden_state_agop": "V^T C V / width for centered logits; no labels or loss gradients",
            "primary_agop_fourier_geometry": (
                "held-out-row centered principal angles with relative SVD cutoff 1e-10"
            ),
            "agop_purity": "mean squared principal cosine",
            "agop_coverage": "principal-angle overlap divided by Fourier target rank",
            "population_geometry": "separately prefixed full_grid fields",
            "fourier_probe": (
                "ridge strength selected by inner training folds; score only on held-out rows"
            ),
            "full_hidden_train_only_refit": (
                "fresh affine ridge map fit on all centered hidden coordinates; "
                "ridge selected by inner training-only folds and held-out labels "
                "used only for final scoring; unlike agop_top*_ probes, no "
                "trained-head AGOP subspace selects the features"
            ),
            "head_empirical_gram_contraction": (
                "checkpoint fixed-feature head-Hessian bounds: Lambda is head "
                "decay plus lambda_max(H_train^T H_train/n)/(2*width); the "
                "trace field is its computable envelope; stepsize_lhs is "
                "lr_head*(Lambda+head_decay) and contraction_factor is the "
                "sharp spectral q; these fixed-objective fields are marked "
                "inapplicable (and emitted as null) after adaptive decay starts"
            ),
            "random_controls": (
                f"{config.agop_random_controls} fixed seeded Haar hidden-coordinate subspaces per rank"
            ),
            "legacy_full_span": (
                "preserved separately; measures availability in the entire hidden span"
            ),
        },
        "summary": summary,
        "trajectory": rows,
    }
    if config.theory_gradient_observer:
        payload["observer_semantics"]["theory_gradient_observer"] = (
            "checkpoint-only gauge-corrected train/full raw-gradient cosines; "
            "full-table labels are observer-only; feature-gradient logit JVPs "
            "are compared with the modular rule and sum/difference Fourier spaces; "
            "optimizer-direction JVPs include current momentum, row-sphere tangent "
            "projection, and Polar/Fro/GD transformation; optimizer_posttangent "
            "fields additionally apply the row-sphere tangent projection after "
            "that transform, giving the first-order retraction velocity; all fields "
            "exclude decay and finite retraction; the full-gradient direction shares "
            "the empirical optimizer's current momentum history; finite_head_entry "
            "fields evaluate J=Pi grad C and K=-Pi grad L_S in identifiable "
            "row-tangent coordinates; all_step fields are updated at every discrete "
            "iterate and include a fixed worst-at-interpolation held-out witness"
        )
    if config.rule_margin_observer:
        payload["observer_semantics"]["rule_margin_observer"] = (
            "checkpoint-only full-grid least-squares coefficients on the centered "
            "addition and difference teacher tensors; rule_xi is their log ratio; "
            "the held-out margin discrepancy is the worst target-versus-false-class "
            "difference between the network and its fitted two-rule component"
        )
    with (output_dir / "run.json").open("x", encoding="utf-8") as handle:
        json.dump(_jsonable(payload), handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    with (output_dir / "trajectory.csv").open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    np.savez_compressed(
        output_dir / "trajectory.npz",
        **{
            key: np.asarray([float(row[key]) for row in rows], dtype=np.float64)
            for key in rows[0]
        },
    )


def run_experiment(
    config: RunConfig,
    output_dir: Path | None = None,
) -> dict[str, Any]:
    config.validate()
    device = resolve_device(config.device)
    dtype = resolve_dtype(config.dtype)
    configure_determinism(config.seed, device)
    data = make_data(config.p, config.train_fraction, config.split_seed, device)
    model = QuadraticOneHiddenLayer(
        config.p,
        config.width,
        config.feature_init_scale,
        config.head_init_scale,
        device=device,
        dtype=dtype,
        activation=config.activation,
    )
    optimizer = EndToEndMatrixOptimizer(model, config)
    all_step_theory = (
        AllStepTheoryBookkeeper(
            data,
            width=config.width,
            head_learning_rate=config.lr_head,
            head_weight_decay=config.weight_decay_head,
        )
        if config.theory_gradient_observer
        else None
    )
    train_pairs, train_labels = data.pairs[data.train_idx], data.labels[data.train_idx]
    test_pairs, test_labels = data.pairs[data.test_idx], data.labels[data.test_idx]
    rows: list[dict[str, Any]] = []
    last_update: dict[str, float | int] = {
        "optimizer_step": 0,
        "feature_gradient_fro": math.nan,
        "feature_pre_tangent_gradient_fro": math.nan,
        "feature_tangent_gradient_fro": math.nan,
        "feature_tangent_gradient_nuclear_norm": math.nan,
        "feature_tangent_gradient_retained_nuclear_norm": math.nan,
        "feature_data_direction_pairing": math.nan,
        "feature_pre_retraction_signed_first_order_ce_change": math.nan,
        "feature_radial_gradient_fro": math.nan,
        "feature_tangent_max_abs_cosine": math.nan,
        "head_gradient_fro": math.nan,
        "feature_direction_fro": math.nan,
        "feature_transformed_direction_radial_fro": math.nan,
        "head_direction_fro": math.nan,
        "feature_update_fro": math.nan,
        "feature_actual_step_fro": math.nan,
        "feature_pre_retraction_row_norm_max_abs_error": math.nan,
        "feature_post_retraction_row_norm_max_abs_error": math.nan,
        "feature_retraction_correction_fro": math.nan,
        "feature_retraction_gradient_inner_product": math.nan,
        "feature_exact_gradient_increment_inner_product": math.nan,
        "head_update_fro": math.nan,
        "head_decay_update_fro": math.nan,
        "head_combined_direction_fro": math.nan,
        "head_data_gradient_dot_combined_direction": math.nan,
        "head_signed_first_order_ce_change": math.nan,
        "total_signed_first_order_ce_change": math.nan,
        "head_exact_gradient_increment_inner_product": math.nan,
        "full_exact_gradient_increment_inner_product": math.nan,
        "training_ce_realized_one_step_change": math.nan,
        "training_ce_first_order_taylor_remainder": math.nan,
        "adaptive_phase_cumulative_abs_training_ce_realized_change": (
            0.0 if config.adaptive_balance_start_step >= 0 else math.nan
        ),
        "adaptive_phase_cumulative_abs_full_exact_gradient_increment_inner_product": (
            0.0 if config.adaptive_balance_start_step >= 0 else math.nan
        ),
        "adaptive_phase_cumulative_abs_training_ce_taylor_remainder": (
            0.0 if config.adaptive_balance_start_step >= 0 else math.nan
        ),
        "adaptive_phase_active_update_count": (
            0 if config.adaptive_balance_start_step >= 0 else math.nan
        ),
        "adaptive_phase_expected_update_count": (
            0 if config.adaptive_balance_start_step >= 0 else math.nan
        ),
        "adaptive_phase_formula_used_update_count": (
            0 if config.adaptive_balance_start_step >= 0 else math.nan
        ),
        "adaptive_phase_fallback_update_count": (
            0 if config.adaptive_balance_start_step >= 0 else math.nan
        ),
        "adaptive_phase_clipped_update_count": (
            0 if config.adaptive_balance_start_step >= 0 else math.nan
        ),
        "adaptive_phase_minimum_denominator": math.nan,
        "adaptive_phase_maximum_absolute_balance_residual": math.nan,
        "adaptive_phase_nonfinite_mechanism_diagnostic_count": (
            0 if config.adaptive_balance_start_step >= 0 else math.nan
        ),
        "adaptive_balance_all_step_index_mismatch_count": (
            0 if config.adaptive_balance_start_step >= 0 else math.nan
        ),
        "adaptive_balance_all_step_phase_mismatch_count": (
            0 if config.adaptive_balance_start_step >= 0 else math.nan
        ),
        "head_actual_step_fro": math.nan,
        "head_direction_decay_cosine": math.nan,
        "head_decay_balance_ratio": math.nan,
        "adaptive_head_decay_active": 0,
        "adaptive_head_decay_formula_used": 0,
        "adaptive_head_decay_was_clipped": 0,
        "adaptive_head_decay_lambda": config.weight_decay_head,
        "adaptive_head_decay_denominator": math.nan,
        "adaptive_head_decay_unclipped_lambda": config.weight_decay_head,
        "adaptive_head_decay_balance_residual": math.nan,
        "adaptive_head_decay_source_state_step": -1,
        "adaptive_head_decay_update_number": 0,
        "feature_gradient_rank": 0,
        "head_gradient_rank": 0,
        "feature_required_rank": min(config.width, 2 * config.p - 1),
        "head_required_rank": min(config.p - 1, config.width),
        "feature_gradient_condition": math.nan,
        "head_gradient_condition": math.nan,
    }

    adaptive_cumulative_abs_realized_change = 0.0
    adaptive_cumulative_abs_exact_first_order = 0.0
    adaptive_cumulative_abs_taylor_remainder = 0.0
    adaptive_active_update_count = 0
    adaptive_formula_used_update_count = 0
    adaptive_fallback_update_count = 0
    adaptive_clipped_update_count = 0
    adaptive_minimum_denominator = math.inf
    adaptive_maximum_absolute_balance_residual = 0.0
    adaptive_nonfinite_mechanism_diagnostic_count = 0
    adaptive_all_step_index_mismatch_count = 0
    adaptive_all_step_phase_mismatch_count = 0

    def observe(step: int) -> None:
        theory_diagnostics: dict[str, float] = {}
        if config.theory_gradient_observer:
            theory_diagnostics = theory_gradient_diagnostics(
                model,
                data.pairs,
                data.labels,
                data.train_idx,
                config=config,
                optimizer_state=optimizer,
            )
        all_step_diagnostics = (
            all_step_theory.snapshot() if all_step_theory is not None else {}
        )
        rule_diagnostics: dict[str, float] = {}
        if config.rule_margin_observer:
            rule_diagnostics = rule_margin_diagnostics(
                model(data.pairs),
                data.pairs,
                data.labels,
                data.test_idx,
                config.p,
            )
        train = partition_metrics(model, train_pairs, train_labels)
        test = partition_metrics(model, test_pairs, test_labels)
        full_hidden = model.hidden(data.pairs)
        fixed_decay_contraction_applicable = not bool(
            last_update["adaptive_head_decay_active"]
        )
        if fixed_decay_contraction_applicable:
            head_contraction = empirical_gram_head_diagnostics(
                full_hidden[data.train_idx],
                width=config.width,
                head_learning_rate=config.lr_head,
                head_weight_decay=config.weight_decay_head,
            )
        else:
            head_contraction = {
                "head_empirical_gram_smoothness": math.nan,
                "head_empirical_trace_smoothness_envelope": math.nan,
                "head_contraction_stepsize_lhs": math.nan,
                "head_contraction_factor": math.nan,
            }
        head_contraction["head_fixed_decay_contraction_applicable"] = int(
            fixed_decay_contraction_applicable
        )
        alignment = fourier_alignment(full_hidden, config.p)
        agop_alignment = agop_fourier_alignment(
            full_hidden,
            model.head_weight,
            config.p,
            data.labels,
            data.train_idx,
            data.test_idx,
            ranks=config.agop_ranks,
            ridge_alphas=config.agop_ridge_alphas,
            probe_folds=config.agop_probe_folds,
            probe_seed=config.agop_probe_seed,
            random_controls=config.agop_random_controls,
        )
        row: dict[str, Any] = {
            "step": int(step),
            "train_ce": train["ce"],
            "test_ce": test["ce"],
            "train_accuracy": train["accuracy"],
            "test_accuracy": test["accuracy"],
            "train_margin_mean": train["margin_mean"],
            "test_margin_mean": test["margin_mean"],
            "train_margin_min": train["margin_min"],
            "test_margin_min": test["margin_min"],
            **theory_diagnostics,
            **all_step_diagnostics,
            **rule_diagnostics,
            **alignment,
            **agop_alignment,
            **head_contraction,
            "feature_weight_fro": float(
                torch.linalg.matrix_norm(model.feature_weight, ord="fro").item()
            ),
            "head_weight_fro": float(
                torch.linalg.matrix_norm(model.head_weight, ord="fro").item()
            ),
            **optimizer.feature_row_norm_diagnostics(),
            **last_update,
        }
        rows.append(row)

    if all_step_theory is not None:
        all_step_theory.update(model, 0)
    observe(0)
    for step in range(1, config.steps + 1):
        optimizer.zero_grad()
        loss = F.cross_entropy(model(train_pairs), train_labels)
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError(f"nonfinite training loss before step {step}")
        loss.backward()
        last_update = optimizer.step()
        last_update.update(
            {
                "training_ce_realized_one_step_change": math.nan,
                "training_ce_first_order_taylor_remainder": math.nan,
                "adaptive_phase_cumulative_abs_training_ce_realized_change": math.nan,
                "adaptive_phase_cumulative_abs_full_exact_gradient_increment_inner_product": math.nan,
                "adaptive_phase_cumulative_abs_training_ce_taylor_remainder": math.nan,
                "adaptive_phase_active_update_count": math.nan,
                "adaptive_phase_expected_update_count": math.nan,
                "adaptive_phase_formula_used_update_count": math.nan,
                "adaptive_phase_fallback_update_count": math.nan,
                "adaptive_phase_clipped_update_count": math.nan,
                "adaptive_phase_minimum_denominator": math.nan,
                "adaptive_phase_maximum_absolute_balance_residual": math.nan,
                "adaptive_phase_nonfinite_mechanism_diagnostic_count": math.nan,
                "adaptive_balance_all_step_index_mismatch_count": math.nan,
                "adaptive_balance_all_step_phase_mismatch_count": math.nan,
            }
        )
        if config.adaptive_balance_start_step >= 0:
            expected_adaptive = step > config.adaptive_balance_start_step
            index_matches = bool(
                int(last_update["adaptive_head_decay_source_state_step"])
                == step - 1
                and int(last_update["adaptive_head_decay_update_number"])
                == step
            )
            active = bool(last_update["adaptive_head_decay_active"])
            if not index_matches:
                adaptive_all_step_index_mismatch_count += 1
            if active != expected_adaptive:
                adaptive_all_step_phase_mismatch_count += 1
            if active:
                adaptive_active_update_count += 1
                formula_used = bool(
                    last_update["adaptive_head_decay_formula_used"]
                )
                adaptive_formula_used_update_count += int(formula_used)
                adaptive_fallback_update_count += int(not formula_used)
                adaptive_clipped_update_count += int(
                    bool(last_update["adaptive_head_decay_was_clipped"])
                )
                denominator = float(
                    last_update["adaptive_head_decay_denominator"]
                )
                residual = float(
                    last_update["adaptive_head_decay_balance_residual"]
                )
                if math.isfinite(denominator) and math.isfinite(residual):
                    adaptive_minimum_denominator = min(
                        adaptive_minimum_denominator, denominator
                    )
                    adaptive_maximum_absolute_balance_residual = max(
                        adaptive_maximum_absolute_balance_residual,
                        abs(residual),
                    )
                else:
                    adaptive_nonfinite_mechanism_diagnostic_count += 1
            with torch.no_grad():
                post_update_loss = F.cross_entropy(
                    model(train_pairs), train_labels
                )
            if not bool(torch.isfinite(post_update_loss)):
                raise FloatingPointError(
                    f"nonfinite training loss after step {step}"
                )
            realized_change = float((post_update_loss - loss.detach()).item())
            exact_first_order = float(
                last_update["full_exact_gradient_increment_inner_product"]
            )
            taylor_remainder = realized_change - exact_first_order
            last_update["training_ce_realized_one_step_change"] = realized_change
            last_update[
                "training_ce_first_order_taylor_remainder"
            ] = taylor_remainder
            if bool(last_update["adaptive_head_decay_active"]):
                adaptive_cumulative_abs_realized_change += abs(realized_change)
                adaptive_cumulative_abs_exact_first_order += abs(exact_first_order)
                adaptive_cumulative_abs_taylor_remainder += abs(taylor_remainder)
            last_update[
                "adaptive_phase_cumulative_abs_training_ce_realized_change"
            ] = adaptive_cumulative_abs_realized_change
            last_update[
                "adaptive_phase_cumulative_abs_full_exact_gradient_increment_inner_product"
            ] = adaptive_cumulative_abs_exact_first_order
            last_update[
                "adaptive_phase_cumulative_abs_training_ce_taylor_remainder"
            ] = adaptive_cumulative_abs_taylor_remainder
            last_update["adaptive_phase_active_update_count"] = (
                adaptive_active_update_count
            )
            last_update["adaptive_phase_expected_update_count"] = max(
                0, step - config.adaptive_balance_start_step
            )
            last_update["adaptive_phase_formula_used_update_count"] = (
                adaptive_formula_used_update_count
            )
            last_update["adaptive_phase_fallback_update_count"] = (
                adaptive_fallback_update_count
            )
            last_update["adaptive_phase_clipped_update_count"] = (
                adaptive_clipped_update_count
            )
            last_update["adaptive_phase_minimum_denominator"] = (
                adaptive_minimum_denominator
                if adaptive_active_update_count
                else math.nan
            )
            last_update[
                "adaptive_phase_maximum_absolute_balance_residual"
            ] = (
                adaptive_maximum_absolute_balance_residual
                if adaptive_active_update_count
                else math.nan
            )
            last_update[
                "adaptive_phase_nonfinite_mechanism_diagnostic_count"
            ] = adaptive_nonfinite_mechanism_diagnostic_count
            last_update[
                "adaptive_balance_all_step_index_mismatch_count"
            ] = adaptive_all_step_index_mismatch_count
            last_update[
                "adaptive_balance_all_step_phase_mismatch_count"
            ] = adaptive_all_step_phase_mismatch_count
        if all_step_theory is not None:
            all_step_theory.update(
                model,
                step,
                fixed_decay_contraction_applicable=not bool(
                    last_update["adaptive_head_decay_active"]
                ),
            )
        if step % config.log_every == 0 or step == config.steps:
            observe(step)

    summary = summarize(rows, config)
    if output_dir is not None:
        output_path = Path(output_dir)
        if output_path.exists():
            raise FileExistsError(f"refusing to overwrite existing output: {output_path}")
        _write_outputs(output_path, config, data, rows, summary)
    return {
        "config": config,
        "data": data,
        "model": model,
        "trajectory": rows,
        "summary": summary,
    }


def run_id(config: RunConfig) -> str:
    architecture_suffix = ""
    if config.activation != "square" or config.feature_geometry != "euclidean":
        architecture_suffix = (
            f"_act{config.activation}_geom{config.feature_geometry}"
        )
    adaptive_suffix = ""
    if config.adaptive_balance_start_step >= 0:
        minimum = (
            "none"
            if config.adaptive_balance_decay_min is None
            else f"{config.adaptive_balance_decay_min:g}"
        )
        maximum = (
            "none"
            if config.adaptive_balance_decay_max is None
            else f"{config.adaptive_balance_decay_max:g}"
        )
        adaptive_suffix = (
            f"_adaptive{config.adaptive_balance_start_step}"
            f"_floor{config.adaptive_balance_denominator_floor:g}"
            f"_clip{minimum}_{maximum}"
        )
    return (
        f"{config.optimizer}_p{config.p}_rho{config.train_fraction:g}_"
        f"k{config.width}_lf{config.lr_feature:g}_lh{config.lr_head:g}_"
        f"wd{config.weight_decay_feature:g}_{config.weight_decay_head:g}_"
        f"mom{config.momentum:g}{architecture_suffix}{adaptive_suffix}_"
        f"seed{config.seed}_split{config.split_seed}"
    )
