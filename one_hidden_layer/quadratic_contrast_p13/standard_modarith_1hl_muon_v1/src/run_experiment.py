#!/usr/bin/env python3

from __future__ import annotations

import argparse
import itertools
import json
from dataclasses import asdict
from pathlib import Path
from typing import Callable, TypeVar

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from modarith_1hl import (
    ACTIVATIONS,
    FEATURE_GEOMETRIES,
    OPTIMIZERS,
    RunConfig,
    run_experiment,
    run_id,
)


T = TypeVar("T")


def csv_values(text: str, convert: Callable[[str], T]) -> list[T]:
    values = [convert(piece.strip()) for piece in text.split(",") if piece.strip()]
    if not values:
        raise argparse.ArgumentTypeError("expected a nonempty comma-separated list")
    return values


def add_shared(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--p", type=int, default=31)
    parser.add_argument("--train-fraction", type=float, default=0.3)
    parser.add_argument("--split-seed", type=int, default=1)
    parser.add_argument("--steps", type=int, default=2_000)
    parser.add_argument("--activation", choices=ACTIVATIONS, default="square")
    parser.add_argument(
        "--feature-geometry", choices=FEATURE_GEOMETRIES, default="euclidean"
    )
    parser.add_argument("--lr-head", type=float, default=0.01)
    parser.add_argument("--momentum", type=float, default=0.0)
    parser.add_argument("--weight-decay-feature", type=float, default=0.0)
    parser.add_argument("--weight-decay-head", type=float, default=0.0)
    parser.add_argument(
        "--adaptive-balance-start-step",
        type=int,
        default=-1,
        help=(
            "completed warmup state T_P; ordinary updates 1 through T_P "
            "produce theta_T_P, and adaptive head decay begins on update "
            "T_P+1 (theta_T_P to theta_(T_P+1)); 0 begins on update 1; "
            "-1 disables it (default)"
        ),
    )
    parser.add_argument(
        "--adaptive-balance-denominator-floor",
        type=float,
        default=1.0e-12,
        help=(
            "use the configured fixed head decay when -<G_V,V> does not "
            "exceed this floor"
        ),
    )
    parser.add_argument(
        "--adaptive-balance-decay-min",
        type=float,
        default=None,
        help="optional active-phase lower clip for adaptive head decay",
    )
    parser.add_argument(
        "--adaptive-balance-decay-max",
        type=float,
        default=None,
        help="optional active-phase upper clip for adaptive head decay",
    )
    parser.add_argument("--feature-init-scale", type=float, default=2.0 ** -0.5)
    parser.add_argument("--head-init-scale", type=float, default=0.05)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float64")
    parser.add_argument("--strict-polar-rank", action="store_true")
    parser.add_argument("--train-accuracy-threshold", type=float, default=0.99)
    parser.add_argument("--test-accuracy-threshold", type=float, default=0.90)
    parser.add_argument("--alignment-threshold", type=float, default=0.80)
    parser.add_argument("--minimum-delay-steps", type=int, default=100)
    parser.add_argument(
        "--agop-ranks",
        default="",
        help=(
            "comma-separated leading-AGOP ranks; empty uses 4,8,16 plus the "
            "clipped p-1 rule rank"
        ),
    )
    parser.add_argument(
        "--agop-ridge-alphas",
        default="1e-5,1e-4,1e-3,1e-2,1e-1,1",
        help="positive train-only inner-CV ridge candidates",
    )
    parser.add_argument("--agop-probe-folds", type=int, default=3)
    parser.add_argument("--agop-probe-seed", type=int, default=314159)
    parser.add_argument("--agop-random-controls", type=int, default=4)
    parser.add_argument(
        "--theory-gradient-observer",
        action="store_true",
        help=(
            "at logged checkpoints compare train/full raw gradients and "
            "feature-induced Fourier logit velocities"
        ),
    )
    parser.add_argument(
        "--rule-margin-observer",
        action="store_true",
        help=(
            "at logged checkpoints fit addition/difference rule coefficients "
            "and audit the uniform held-out margin condition"
        ),
    )


def shared_kwargs(args: argparse.Namespace) -> dict[str, object]:
    names = (
        "p",
        "train_fraction",
        "split_seed",
        "steps",
        "activation",
        "feature_geometry",
        "lr_head",
        "momentum",
        "weight_decay_feature",
        "weight_decay_head",
        "adaptive_balance_start_step",
        "adaptive_balance_denominator_floor",
        "adaptive_balance_decay_min",
        "adaptive_balance_decay_max",
        "feature_init_scale",
        "head_init_scale",
        "log_every",
        "device",
        "dtype",
        "strict_polar_rank",
        "train_accuracy_threshold",
        "test_accuracy_threshold",
        "alignment_threshold",
        "minimum_delay_steps",
        "agop_probe_folds",
        "agop_probe_seed",
        "agop_random_controls",
        "theory_gradient_observer",
        "rule_margin_observer",
    )
    result = {name: getattr(args, name) for name in names}
    result["agop_ranks"] = (
        tuple(csv_values(args.agop_ranks, int)) if args.agop_ranks.strip() else ()
    )
    result["agop_ridge_alphas"] = tuple(
        csv_values(args.agop_ridge_alphas, float)
    )
    return result


def plot_trajectory(rows: list[dict], destination: Path, title: str) -> None:
    step = np.asarray([row["step"] for row in rows])
    train_ce = np.asarray([row["train_ce"] for row in rows])
    test_ce = np.asarray([row["test_ce"] for row in rows])
    train_accuracy = np.asarray([row["train_accuracy"] for row in rows])
    test_accuracy = np.asarray([row["test_accuracy"] for row in rows])
    alignment = np.asarray([row["fourier_sum_subspace_alignment"] for row in rows])
    difference_alignment = np.asarray(
        [row["fourier_difference_subspace_alignment"] for row in rows]
    )
    headfit_risk = np.asarray([row["fourier_sum_headfit_risk"] for row in rows])
    low_agop_prefix = (
        "agop_top4" if "agop_top4_fourier_sum_purity" in rows[0] else "agop_top_rule"
    )
    rule_agop_prefix = "agop_top_rule"

    def optional(name: str) -> np.ndarray | None:
        if any(name not in row or row[name] is None for row in rows):
            return None
        return np.asarray([float(row[name]) for row in rows])

    agop_rank_label = "4" if low_agop_prefix == "agop_top4" else "rule"
    probe_accuracy = optional(f"{low_agop_prefix}_heldout_modular_accuracy")
    sum_coverage = optional(f"{rule_agop_prefix}_fourier_sum_coverage")
    difference_coverage = optional(f"{rule_agop_prefix}_fourier_difference_coverage")
    sum_purity = optional(f"{low_agop_prefix}_fourier_sum_purity")
    difference_purity = optional(f"{low_agop_prefix}_fourier_difference_purity")
    probe_r2 = optional(f"{low_agop_prefix}_heldout_fourier_r2")
    purity_uplift = optional(
        f"{low_agop_prefix}_top_minus_random_fourier_sum_purity"
    )
    r2_uplift = optional(
        f"{low_agop_prefix}_top_minus_random_heldout_fourier_r2"
    )
    energy_fraction = optional(f"{low_agop_prefix}_energy_fraction")
    relative_eigengap = optional(f"{low_agop_prefix}_relative_eigengap")

    fig, axes = plt.subplots(2, 2, figsize=(11.2, 7.6), sharex=True)
    ax = axes[0, 0]
    ax.plot(step, train_ce, label="train CE", color="#3569b0")
    ax.plot(step, test_ce, label="test CE", color="#23966f")
    ax.set_yscale("log")
    ax.set_ylabel("cross-entropy")
    ax.set_title("A. Predictive loss")
    ax.legend(frameon=False)

    ax = axes[0, 1]
    ax.plot(step, train_accuracy, label="train", color="#3569b0")
    ax.plot(step, test_accuracy, label="test", color="#23966f")
    if probe_accuracy is not None:
        ax.plot(
            step,
            probe_accuracy,
            label=f"top-{agop_rank_label} train-only probe",
            color="#d37a28",
            linestyle="--",
        )
    ax.set_ylim(-0.02, 1.02)
    ax.set_ylabel("accuracy")
    ax.set_title("B. Standard grokking and held-out probe")
    ax.legend(frameon=False, fontsize=8)

    ax = axes[1, 0]
    ax.plot(step, alignment, label="full-span sum availability", color="#7a4db3")
    ax.plot(step, headfit_risk, label="full-span oracle risk", color="#d37a28")
    ax.plot(
        step,
        difference_alignment,
        label="full-span difference control",
        color="#7b8794",
        linestyle="--",
    )
    if sum_coverage is not None:
        ax.plot(
            step,
            sum_coverage,
            label="top-rule held-out sum coverage",
            color="#a34f93",
            linewidth=1.8,
        )
    if difference_coverage is not None:
        ax.plot(
            step,
            difference_coverage,
            label="top-rule held-out difference coverage",
            color="#7b8794",
            linestyle=":",
        )
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel("optimization step")
    ax.set_ylabel("normalized value")
    ax.set_title("C. Availability versus head-selected use")
    ax.legend(frameon=False, fontsize=7)

    ax = axes[1, 1]
    panel_d = (
        (purity_uplift, "sum-purity uplift vs random", "#7a4db3", "-"),
        (r2_uplift, "held-out R2 uplift vs random", "#23966f", "--"),
        (energy_fraction, "AGOP energy fraction", "#3569b0", ":"),
        (
            relative_eigengap,
            "relative eigengap",
            "#d37a28",
            "-.",
        ),
    )
    minimum = 0.0
    for values, label, color, linestyle in panel_d:
        if values is not None:
            ax.plot(step, values, label=label, color=color, linestyle=linestyle)
            minimum = min(minimum, float(np.nanmin(values)))
    ax.axhline(0.0, color="#555555", linewidth=0.8, alpha=0.5)
    ax.set_ylim(max(-1.0, minimum - 0.05), 1.02)
    ax.set_xlabel("optimization step")
    ax.set_ylabel("uplift / spectrum fraction")
    ax.set_title(f"D. Top-{agop_rank_label} evidence above random")
    ax.legend(frameon=False, fontsize=7)

    for ax in axes.flat:
        ax.grid(alpha=0.2, linewidth=0.6)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(destination / "trajectory.png", dpi=220, bbox_inches="tight")
    fig.savefig(destination / "trajectory.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_title(config: RunConfig) -> str:
    architecture = ""
    if config.activation != "square" or config.feature_geometry != "euclidean":
        architecture = f", {config.activation}/{config.feature_geometry}"
    return (
        f"{config.optimizer}: p={config.p}, train={config.train_fraction:g}, "
        f"k={config.width}, eta=({config.lr_feature:g}, {config.lr_head:g}), "
        f"wd=({config.weight_decay_feature:g}, {config.weight_decay_head:g})"
        f"{architecture}"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    single = subparsers.add_parser("single")
    add_shared(single)
    single.add_argument("--seed", type=int, default=0)
    single.add_argument("--width", type=int, default=128)
    single.add_argument("--optimizer", choices=OPTIMIZERS, default="polar_muon")
    single.add_argument("--lr-feature", type=float, default=0.01)
    single.add_argument("--output-dir", type=Path, required=True)

    grid = subparsers.add_parser("grid")
    add_shared(grid)
    grid.add_argument("--seeds", default="0")
    grid.add_argument("--widths", default="64,128")
    grid.add_argument("--optimizers", default="polar_muon,fro_ngd")
    grid.add_argument(
        "--activations",
        default="",
        help="comma-separated activation grid; empty uses --activation",
    )
    grid.add_argument(
        "--feature-geometries",
        default="",
        help="comma-separated feature-geometry grid; empty uses --feature-geometry",
    )
    grid.add_argument("--lr-features", default="0.003,0.01,0.03")
    grid.add_argument(
        "--head-to-feature-ratios",
        default="1",
        help="comma-separated head/feature learning-rate ratios",
    )
    grid.add_argument(
        "--weight-decays",
        default="0",
        help="comma-separated common decoupled weight decays for both matrices",
    )
    grid.add_argument("--max-runs", type=int, default=64)
    grid.add_argument("--output-dir", type=Path, required=True)
    return parser


def run_single(args: argparse.Namespace) -> dict:
    config = RunConfig(
        seed=args.seed,
        width=args.width,
        optimizer=args.optimizer,
        lr_feature=args.lr_feature,
        **shared_kwargs(args),
    )
    result = run_experiment(config, args.output_dir)
    plot_trajectory(result["trajectory"], args.output_dir, plot_title(config))
    return {"config": asdict(config), "summary": result["summary"]}


def run_grid(args: argparse.Namespace) -> dict:
    seeds = csv_values(args.seeds, int)
    widths = csv_values(args.widths, int)
    optimizers = csv_values(args.optimizers, str)
    activations = (
        csv_values(args.activations, str)
        if args.activations.strip()
        else [args.activation]
    )
    feature_geometries = (
        csv_values(args.feature_geometries, str)
        if args.feature_geometries.strip()
        else [args.feature_geometry]
    )
    rates = csv_values(args.lr_features, float)
    ratios = csv_values(args.head_to_feature_ratios, float)
    decays = csv_values(args.weight_decays, float)
    invalid = sorted(set(optimizers) - set(OPTIMIZERS))
    if invalid:
        raise ValueError(f"invalid optimizers: {invalid}")
    invalid_activations = sorted(set(activations) - set(ACTIVATIONS))
    if invalid_activations:
        raise ValueError(f"invalid activations: {invalid_activations}")
    invalid_geometries = sorted(set(feature_geometries) - set(FEATURE_GEOMETRIES))
    if invalid_geometries:
        raise ValueError(f"invalid feature geometries: {invalid_geometries}")
    if "row_sphere" in feature_geometries and any(decay != 0.0 for decay in decays):
        raise ValueError(
            "row_sphere grids require --weight-decays 0 because identifiable "
            "feature-row norms are fixed"
        )
    combinations = list(
        itertools.product(
            optimizers,
            widths,
            rates,
            ratios,
            decays,
            seeds,
            activations,
            feature_geometries,
        )
    )
    if len(combinations) > args.max_runs:
        raise ValueError(f"grid has {len(combinations)} runs, above --max-runs={args.max_runs}")
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {args.output_dir}")
    args.output_dir.mkdir(parents=True)
    records = []
    common = shared_kwargs(args)
    common.pop("lr_head")
    common.pop("weight_decay_feature")
    common.pop("weight_decay_head")
    common.pop("activation")
    common.pop("feature_geometry")
    for index, (
        optimizer,
        width,
        rate,
        ratio,
        decay,
        seed,
        activation,
        feature_geometry,
    ) in enumerate(
        combinations, start=1
    ):
        config = RunConfig(
            seed=seed,
            width=width,
            optimizer=optimizer,
            lr_feature=rate,
            lr_head=rate * ratio,
            weight_decay_feature=decay,
            weight_decay_head=decay,
            activation=activation,
            feature_geometry=feature_geometry,
            **common,
        )
        child = args.output_dir / run_id(config)
        print(f"[{index}/{len(combinations)}] {child.name}", flush=True)
        result = run_experiment(config, child)
        plot_trajectory(result["trajectory"], child, plot_title(config))
        records.append(
            {"run_id": child.name, "config": asdict(config), "summary": result["summary"]}
        )
    payload = {"schema_version": 1, "run_count": len(records), "runs": records}
    with (args.output_dir / "grid_summary.json").open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    return payload


def main() -> None:
    args = build_parser().parse_args()
    payload = run_single(args) if args.command == "single" else run_grid(args)
    print(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
