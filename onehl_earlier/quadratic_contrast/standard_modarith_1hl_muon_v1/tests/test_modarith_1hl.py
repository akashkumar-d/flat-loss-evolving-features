from __future__ import annotations

import json
import math
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from modarith_1hl import (
    AllStepTheoryBookkeeper,
    EndToEndMatrixOptimizer,
    QuadraticOneHiddenLayer,
    RunConfig,
    _train_only_fourier_probe,
    agop_fourier_alignment,
    empirical_gram_head_diagnostics,
    feature_induced_logit_velocity,
    fourier_alignment,
    fourier_logit_velocity_projection,
    hidden_state_agop,
    make_data,
    numerical_column_basis,
    output_gauge_basis,
    partition_metrics,
    real_fourier_bases,
    retract_rows_to_fixed_norms,
    rule_margin_diagnostics,
    rowwise_tangent_projection,
    run_experiment,
    run_id,
    support_polar_factor,
    transformed_direction,
    theory_gradient_diagnostics,
)
from run_experiment import build_parser, shared_kwargs


def tiny_config(**updates: object) -> RunConfig:
    base = RunConfig(
        seed=7,
        split_seed=11,
        p=5,
        train_fraction=0.4,
        width=7,
        steps=3,
        optimizer="polar_muon",
        lr_feature=0.01,
        lr_head=0.01,
        log_every=1,
        minimum_delay_steps=1,
        agop_random_controls=1,
    )
    return replace(base, **updates)


def test_random_entry_split_is_disjoint_deterministic_and_complete() -> None:
    device = torch.device("cpu")
    first = make_data(7, 0.3, 19, device)
    second = make_data(7, 0.3, 19, device)
    np.testing.assert_array_equal(first.train_idx_numpy, second.train_idx_numpy)
    assert first.split_fingerprint == second.split_fingerprint
    assert not set(first.train_idx_numpy) & set(first.test_idx_numpy)
    assert len(first.train_idx_numpy) + len(first.test_idx_numpy) == 49


def test_support_polar_and_matched_ngd_norms_agree_at_full_rank() -> None:
    generator = torch.Generator().manual_seed(3)
    gradient = torch.randn(4, 9, generator=generator, dtype=torch.float64)
    polar, diagnostics = support_polar_factor(gradient, strict_rank=True)
    np.testing.assert_allclose(
        (polar @ polar.T).numpy(), np.eye(4), rtol=0.0, atol=1.0e-12
    )
    polar_direction, _ = transformed_direction(
        gradient, "polar_muon", strict_rank=True
    )
    ngd_direction, _ = transformed_direction(gradient, "fro_ngd", strict_rank=True)
    assert diagnostics["rank"] == 4
    assert torch.linalg.matrix_norm(polar_direction).item() == pytest.approx(2.0)
    assert torch.linalg.matrix_norm(ngd_direction).item() == pytest.approx(2.0)


def test_support_polar_retained_nuclear_sum_matches_thresholded_direction() -> None:
    gradient = torch.zeros(128, 2, dtype=torch.float64)
    gradient[0, 0] = 1.0
    gradient[1, 1] = 1.0e-14

    polar, diagnostics = support_polar_factor(gradient, strict_rank=False)
    pairing = float(torch.sum(gradient * polar).item())

    assert diagnostics["rank"] == 1
    assert diagnostics["nuclear_norm"] == pytest.approx(
        1.0 + 1.0e-14, rel=0.0, abs=2.0e-16
    )
    assert diagnostics["retained_nuclear_norm"] == pytest.approx(
        pairing, rel=0.0, abs=2.0e-16
    )
    assert diagnostics["discarded_nuclear_norm"] == pytest.approx(
        1.0e-14, rel=0.0, abs=2.0e-28
    )
    assert diagnostics["nuclear_norm"] > diagnostics["retained_nuclear_norm"]


def test_perfect_sum_and_difference_codes_have_unit_support_fraction() -> None:
    p = 7
    a = torch.arange(p, dtype=torch.float64).repeat_interleave(p)
    b = torch.arange(p, dtype=torch.float64).repeat(p)
    columns_sum = []
    columns_difference = []
    for frequency in range(1, (p + 1) // 2):
        sum_angle = 2.0 * torch.pi * frequency * (a + b) / p
        difference_angle = 2.0 * torch.pi * frequency * (a - b) / p
        columns_sum.extend((torch.cos(sum_angle), torch.sin(sum_angle)))
        columns_difference.extend(
            (torch.cos(difference_angle), torch.sin(difference_angle))
        )
    sum_metrics = fourier_alignment(torch.stack(columns_sum, dim=1), p)
    difference_metrics = fourier_alignment(
        torch.stack(columns_difference, dim=1), p
    )
    assert sum_metrics["fourier_addition_fraction"] == pytest.approx(1.0, abs=1e-12)
    assert sum_metrics["fourier_addition_covered_fraction"] == pytest.approx(
        1.0, abs=1e-12
    )
    assert sum_metrics["fourier_sum_subspace_alignment"] == pytest.approx(
        1.0, abs=1e-12
    )
    assert sum_metrics["fourier_difference_subspace_alignment"] == pytest.approx(
        0.0, abs=1e-12
    )
    assert sum_metrics["fourier_sum_headfit_risk"] == pytest.approx(0.0, abs=1e-12)
    assert difference_metrics["fourier_difference_fraction"] == pytest.approx(
        1.0, abs=1e-12
    )
    assert difference_metrics["fourier_difference_subspace_alignment"] == pytest.approx(
        1.0, abs=1e-12
    )
    assert difference_metrics["fourier_sum_subspace_alignment"] == pytest.approx(
        0.0, abs=1e-12
    )


def test_model_is_an_exact_lookup_implementation_of_concatenated_one_hot() -> None:
    model = QuadraticOneHiddenLayer(
        5, 6, 0.3, 0.1, device=torch.device("cpu"), dtype=torch.float64
    )
    pairs = torch.tensor([[0, 1], [3, 4], [2, 2]], dtype=torch.long)
    one_hot = torch.nn.functional.one_hot(pairs, 5).reshape(len(pairs), 10).double()
    dense_hidden = (one_hot @ model.feature_weight.T).square()
    torch.testing.assert_close(model.hidden(pairs), dense_hidden)


def test_default_architecture_is_exactly_explicit_square_euclidean() -> None:
    device = torch.device("cpu")
    torch.manual_seed(151)
    implicit = QuadraticOneHiddenLayer(
        5, 6, 0.3, 0.1, device=device, dtype=torch.float64
    )
    torch.manual_seed(151)
    explicit = QuadraticOneHiddenLayer(
        5,
        6,
        0.3,
        0.1,
        device=device,
        dtype=torch.float64,
        activation="square",
    )
    pairs = torch.cartesian_prod(torch.arange(5), torch.arange(5))
    assert torch.equal(implicit.feature_weight, explicit.feature_weight)
    assert torch.equal(implicit.head_weight, explicit.head_weight)
    assert torch.equal(implicit.hidden(pairs), explicit.hidden(pairs))
    assert torch.equal(implicit(pairs), explicit(pairs))

    config = tiny_config()
    assert config.activation == "square"
    assert config.feature_geometry == "euclidean"
    assert run_id(config) == (
        "polar_muon_p5_rho0.4_k7_lf0.01_lh0.01_"
        "wd0_0_mom0_seed7_split11"
    )


@pytest.mark.parametrize("activation", ["relu", "softplus"])
def test_nonsquare_activation_forward_matches_torch(activation: str) -> None:
    model = QuadraticOneHiddenLayer(
        5,
        4,
        0.3,
        0.1,
        device=torch.device("cpu"),
        dtype=torch.float64,
        activation=activation,
    )
    with torch.no_grad():
        model.feature_weight.copy_(
            torch.linspace(-1.7, 1.3, model.feature_weight.numel()).reshape_as(
                model.feature_weight
            )
        )
        model.project_gauges_()
    pairs = torch.tensor([[0, 1], [3, 4], [2, 2]], dtype=torch.long)
    one_hot = torch.nn.functional.one_hot(pairs, 5).reshape(len(pairs), 10).double()
    preactivation = one_hot @ model.feature_weight.T
    expected = (
        torch.nn.functional.relu(preactivation)
        if activation == "relu"
        else torch.nn.functional.softplus(preactivation)
    )
    torch.testing.assert_close(model.hidden(pairs), expected, rtol=0.0, atol=0.0)


@pytest.mark.parametrize("activation", ["square", "relu", "softplus"])
def test_feature_logit_velocity_matches_exact_autograd_jvp(activation: str) -> None:
    generator = torch.Generator().manual_seed(153)
    p, width = 5, 6
    model = QuadraticOneHiddenLayer(
        p,
        width,
        0.3,
        0.1,
        device=torch.device("cpu"),
        dtype=torch.float64,
        activation=activation,
    )
    pairs = torch.cartesian_prod(torch.arange(p), torch.arange(p))
    direction = torch.randn(
        model.feature_weight.shape, generator=generator, dtype=torch.float64
    )

    def centered_logits(weight: torch.Tensor) -> torch.Tensor:
        first = weight[:, pairs[:, 0]].T
        second = weight[:, p + pairs[:, 1]].T
        preactivation = first + second
        if activation == "square":
            hidden = preactivation.square()
        elif activation == "relu":
            hidden = torch.nn.functional.relu(preactivation)
        else:
            hidden = torch.nn.functional.softplus(preactivation)
        logits = hidden @ model.head_weight.T / np.sqrt(width)
        return logits - logits.mean(dim=1, keepdim=True)

    _, expected = torch.autograd.functional.jvp(
        centered_logits,
        (model.feature_weight,),
        (direction,),
        create_graph=False,
    )
    actual = feature_induced_logit_velocity(model, pairs, direction)
    torch.testing.assert_close(actual, expected, rtol=2.0e-14, atol=2.0e-14)


def test_fourier_logit_velocity_projection_separates_sum_and_difference() -> None:
    p = 7
    sum_basis, difference_basis, _ = real_fourier_bases(
        p, torch.device("cpu"), torch.float64
    )
    zero = torch.zeros(p * p, 1, dtype=torch.float64)
    sum_velocity = torch.cat((sum_basis, zero), dim=1)
    difference_velocity = torch.cat((difference_basis, zero), dim=1)
    sum_metrics = fourier_logit_velocity_projection(sum_velocity, p)
    difference_metrics = fourier_logit_velocity_projection(
        difference_velocity, p
    )
    assert sum_metrics["fourier_sum_fraction"] == pytest.approx(1.0, abs=1e-12)
    assert sum_metrics["fourier_difference_fraction"] == pytest.approx(
        0.0, abs=1e-12
    )
    assert difference_metrics["fourier_difference_fraction"] == pytest.approx(
        1.0, abs=1e-12
    )
    assert difference_metrics["fourier_sum_fraction"] == pytest.approx(
        0.0, abs=1e-12
    )


def test_rule_margin_diagnostics_recovers_exact_mixture_and_ignores_logit_gauge() -> None:
    p = 5
    pairs = torch.cartesian_prod(torch.arange(p), torch.arange(p))
    labels = ((pairs[:, 0] + pairs[:, 1]) % p).long()
    difference_labels = ((pairs[:, 0] - pairs[:, 1]) % p).long()
    baseline = torch.full((p * p, p), -1.0 / p, dtype=torch.float64)
    sum_teacher = baseline.scatter(1, labels[:, None], 1.0 - 1.0 / p)
    difference_teacher = baseline.scatter(
        1, difference_labels[:, None], 1.0 - 1.0 / p
    )
    offsets = torch.linspace(-3.0, 2.0, p * p, dtype=torch.float64)[:, None]
    logits = 2.5 * sum_teacher + difference_teacher + offsets
    metrics = rule_margin_diagnostics(
        logits, pairs, labels, torch.arange(3, p * p), p
    )
    assert metrics["rule_addition_coefficient"] == pytest.approx(2.5, abs=1e-14)
    assert metrics["rule_difference_coefficient"] == pytest.approx(1.0, abs=1e-14)
    assert metrics["rule_kappa"] == pytest.approx(3.5, abs=1e-14)
    assert metrics["rule_tau"] == pytest.approx(2.5 / 3.5, abs=1e-14)
    assert metrics["rule_xi"] == pytest.approx(np.log(2.5), abs=1e-14)
    assert metrics["rule_ideal_sum_vs_difference_margin"] == pytest.approx(
        1.5, abs=1e-14
    )
    assert metrics["rule_full_grid_relative_residual_fro"] == pytest.approx(
        0.0, abs=1e-14
    )
    assert metrics["rule_heldout_margin_discrepancy_max"] == pytest.approx(
        0.0, abs=1e-14
    )
    assert metrics["rule_heldout_actual_margin_min"] == pytest.approx(
        1.5, abs=1e-14
    )
    assert metrics["rule_margin_interface_valid"] == 1.0
    assert metrics["rule_xi_bad_test_threshold"] == pytest.approx(0.0, abs=1e-14)
    assert metrics["rule_xi_good_test_threshold"] == pytest.approx(0.0, abs=1e-14)


def test_theory_gradient_observer_is_exact_for_full_training_table_and_no_grad_side_effect() -> None:
    p = 5
    config = tiny_config(
        p=p,
        width=7,
        steps=0,
        optimizer="fro_feature_gd_head",
        activation="softplus",
        feature_geometry="row_sphere",
    )
    model = QuadraticOneHiddenLayer(
        p,
        7,
        0.3,
        0.1,
        device=torch.device("cpu"),
        dtype=torch.float64,
        activation="softplus",
    )
    matrix_optimizer = EndToEndMatrixOptimizer(model, config)
    pairs = torch.cartesian_prod(torch.arange(p), torch.arange(p))
    labels = ((pairs[:, 0] + pairs[:, 1]) % p).long()
    loss = torch.nn.functional.cross_entropy(model(pairs[:9]), labels[:9])
    loss.backward()
    feature_grad_before = model.feature_weight.grad.clone()
    head_grad_before = model.head_weight.grad.clone()

    metrics = theory_gradient_diagnostics(
        model,
        pairs,
        labels,
        torch.arange(p * p),
        config=config,
        optimizer_state=matrix_optimizer,
    )
    assert metrics["theory_train_vs_full_feature_raw_gradient_cosine"] == pytest.approx(
        1.0, abs=2.0e-15
    )
    assert metrics["theory_train_vs_full_head_raw_gradient_cosine"] == pytest.approx(
        1.0, abs=2.0e-15
    )
    assert metrics[
        "theory_train_vs_full_feature_logit_velocity_cosine"
    ] == pytest.approx(1.0, abs=2.0e-15)
    assert metrics[
        "theory_train_vs_full_optimizer_feature_direction_cosine"
    ] == pytest.approx(1.0, abs=2.0e-15)
    assert metrics[
        "theory_train_vs_full_optimizer_feature_logit_velocity_cosine"
    ] == pytest.approx(1.0, abs=2.0e-15)
    assert metrics[
        "theory_train_optimizer_feature_update_rule_cosine"
    ] == pytest.approx(
        metrics["theory_full_optimizer_feature_update_rule_cosine"],
        abs=2.0e-15,
    )
    assert metrics[
        "theory_train_vs_full_optimizer_posttangent_feature_direction_cosine"
    ] == pytest.approx(1.0, abs=2.0e-15)
    assert metrics[
        "theory_train_vs_full_optimizer_posttangent_feature_logit_velocity_cosine"
    ] == pytest.approx(1.0, abs=2.0e-15)
    assert metrics[
        "theory_train_optimizer_posttangent_feature_update_rule_cosine"
    ] == pytest.approx(
        metrics[
            "theory_full_optimizer_posttangent_feature_update_rule_cosine"
        ],
        abs=2.0e-15,
    )
    assert all(np.isfinite(value) for value in metrics.values())
    assert 0.0 <= metrics[
        "theory_full_feature_logit_velocity_fourier_sum_fraction"
    ] <= 1.0
    assert 0.0 <= metrics[
        "theory_full_feature_logit_velocity_fourier_difference_fraction"
    ] <= 1.0
    assert torch.equal(model.feature_weight.grad, feature_grad_before)
    assert torch.equal(model.head_weight.grad, head_grad_before)


def test_partition_minimum_margin_matches_true_versus_best_false_class() -> None:
    generator = torch.Generator().manual_seed(227)
    p, width = 5, 6
    model = QuadraticOneHiddenLayer(
        p,
        width,
        0.4,
        0.2,
        device=torch.device("cpu"),
        dtype=torch.float64,
    )
    with torch.no_grad():
        model.feature_weight.copy_(
            torch.randn(
                model.feature_weight.shape, generator=generator, dtype=torch.float64
            )
        )
        model.head_weight.copy_(
            torch.randn(
                model.head_weight.shape, generator=generator, dtype=torch.float64
            )
        )
        model.project_gauges_()
    pairs = torch.cartesian_prod(torch.arange(p), torch.arange(p))
    labels = ((pairs[:, 0] + pairs[:, 1]) % p).long()
    logits = model(pairs)
    false_logits = logits.masked_fill(F.one_hot(labels, p).bool(), -torch.inf)
    margins = logits[torch.arange(p * p), labels] - false_logits.max(dim=1).values

    metrics = partition_metrics(model, pairs, labels)
    assert metrics["margin_mean"] == pytest.approx(margins.mean().item(), abs=0.0)
    assert metrics["margin_min"] == pytest.approx(margins.min().item(), abs=0.0)


def test_empirical_gram_head_diagnostics_match_theorem_formulas() -> None:
    generator = torch.Generator().manual_seed(229)
    hidden = torch.randn(11, 7, generator=generator, dtype=torch.float64)
    width = hidden.shape[1]
    learning_rate = 0.37
    regularization = 0.013
    metrics = empirical_gram_head_diagnostics(
        hidden,
        width=width,
        head_learning_rate=learning_rate,
        head_weight_decay=regularization,
    )
    gram = hidden.T @ hidden / len(hidden)
    smoothness = regularization + torch.linalg.eigvalsh(gram)[-1].item() / (
        2.0 * width
    )
    trace_envelope = regularization + hidden.square().sum().item() / (
        2.0 * len(hidden) * width
    )
    expected_q = max(
        abs(1.0 - learning_rate * regularization),
        abs(1.0 - learning_rate * smoothness),
    )
    assert metrics["head_empirical_gram_smoothness"] == pytest.approx(
        smoothness, abs=2.0e-15
    )
    assert metrics["head_empirical_trace_smoothness_envelope"] == pytest.approx(
        trace_envelope, abs=2.0e-15
    )
    assert metrics["head_empirical_trace_smoothness_envelope"] >= metrics[
        "head_empirical_gram_smoothness"
    ]
    assert metrics["head_contraction_stepsize_lhs"] == pytest.approx(
        learning_rate * (smoothness + regularization), abs=2.0e-15
    )
    assert metrics["head_contraction_factor"] == pytest.approx(
        expected_q, abs=2.0e-15
    )


def test_all_step_bookkeeper_freezes_witness_and_accumulates_delay_bounds() -> None:
    p = 3
    data = make_data(p, 0.5, 233, torch.device("cpu"))

    class ExactRuleStub(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.p = p
            self.width = p
            self.head_weight = torch.nn.Parameter(
                math.sqrt(p) * torch.eye(p, dtype=torch.float64)
            )
            self.register_buffer(
                "hidden_scale", torch.ones(p, dtype=torch.float64)
            )

        def hidden(self, pairs: torch.Tensor) -> torch.Tensor:
            labels = torch.remainder(pairs[:, 0] + pairs[:, 1], p)
            return F.one_hot(labels, p).to(torch.float64) * self.hidden_scale

    model = ExactRuleStub()
    learning_rate = 0.4
    regularization = 0.02
    bookkeeper = AllStepTheoryBookkeeper(
        data,
        width=p,
        head_learning_rate=learning_rate,
        head_weight_decay=regularization,
    )
    bookkeeper.update(model, 0)
    first = bookkeeper.snapshot()
    assert first["theory_all_step_first_exact_train_fit_step"] == 0
    assert first["theory_all_step_postfit_train_margin_min_running"] == pytest.approx(
        1.0, abs=0.0
    )
    assert first["theory_all_step_witness_margin"] == pytest.approx(1.0, abs=0.0)
    assert first["theory_all_step_witness_cumulative_u_abs"] == 0.0
    assert first["theory_all_step_witness_cumulative_u_hybrid"] == 0.0

    witness_index = int(first["theory_all_step_witness_flat_index"])
    target = int(first["theory_all_step_witness_target_class"])
    competitor = int(first["theory_all_step_witness_competitor_class"])
    hidden = model.hidden(data.pairs[witness_index][None, :])[0]
    old_hidden = hidden.clone()
    old_difference = (
        model.head_weight[target] - model.head_weight[competitor]
    ).detach().clone()
    with torch.no_grad():
        perturbation = torch.tensor(
            [[0.2, -0.1, 0.05], [-0.3, 0.15, 0.1], [0.07, -0.04, -0.2]],
            dtype=torch.float64,
        )
        model.head_weight.add_(perturbation)
        model.hidden_scale.mul_(
            torch.tensor([1.05, 0.96, 1.08], dtype=torch.float64)
        )
    hidden = model.hidden(data.pairs[witness_index][None, :])[0]
    new_difference = (
        model.head_weight[target] - model.head_weight[competitor]
    ).detach()
    delta_difference = new_difference - old_difference
    delta_hidden = hidden - old_hidden
    expected_abs = (
        torch.linalg.vector_norm(delta_difference)
        * torch.linalg.vector_norm(old_hidden)
        + torch.linalg.vector_norm(new_difference)
        * torch.linalg.vector_norm(delta_hidden)
    ).item() / math.sqrt(p)
    expected_hybrid = (
        torch.dot(delta_difference, old_hidden)
        + torch.linalg.vector_norm(new_difference)
        * torch.linalg.vector_norm(delta_hidden)
    ).item() / math.sqrt(p)
    expected_margin = torch.dot(new_difference, hidden).item() / math.sqrt(p)

    bookkeeper.update(model, 1)
    second = bookkeeper.snapshot()
    assert second["theory_all_step_first_exact_train_fit_step"] == 0
    assert second["theory_all_step_witness_flat_index"] == witness_index
    assert second["theory_all_step_witness_target_class"] == target
    assert second["theory_all_step_witness_competitor_class"] == competitor
    assert second["theory_all_step_witness_cumulative_u_abs"] == pytest.approx(
        expected_abs, abs=2.0e-15
    )
    assert second["theory_all_step_witness_cumulative_u_hybrid"] == pytest.approx(
        expected_hybrid, abs=2.0e-15
    )
    assert second["theory_all_step_witness_margin"] == pytest.approx(
        expected_margin, abs=2.0e-15
    )
    assert second[
        "theory_all_step_witness_hybrid_margin_upper_envelope"
    ] + 2.0e-15 >= expected_margin
    assert second["theory_all_step_witness_abs_margin_upper_envelope"] + 2.0e-15 >= (
        expected_margin
    )

    train_hidden = model.hidden(data.pairs[data.train_idx])
    trace_smoothness = regularization + train_hidden.square().sum().item() / (
        2.0 * len(data.train_idx) * p
    )
    trace_smoothness = max(
        trace_smoothness, regularization + 1.0 / (2.0 * p)
    )
    assert second["theory_all_step_head_trace_smoothness_max"] == pytest.approx(
        trace_smoothness, abs=2.0e-15
    )
    assert second[
        "theory_all_step_head_trace_contraction_stepsize_lhs_max"
    ] == pytest.approx(
        learning_rate * (trace_smoothness + regularization), abs=2.0e-15
    )


def test_transformed_theory_cone_matches_manual_partial_split_geometry() -> None:
    config = tiny_config(
        optimizer="polar_feature_gd_head",
        activation="relu",
        feature_geometry="row_sphere",
        steps=0,
    )
    device = torch.device("cpu")
    torch.manual_seed(config.seed)
    model = QuadraticOneHiddenLayer(
        config.p,
        config.width,
        config.feature_init_scale,
        config.head_init_scale,
        device=device,
        dtype=torch.float64,
        activation=config.activation,
    )
    matrix_optimizer = EndToEndMatrixOptimizer(model, config)
    data = make_data(config.p, config.train_fraction, config.split_seed, device)
    metrics = theory_gradient_diagnostics(
        model,
        data.pairs,
        data.labels,
        data.train_idx,
        config=config,
        optimizer_state=matrix_optimizer,
    )

    def raw_feature_gradient(indices: torch.Tensor) -> torch.Tensor:
        loss = torch.nn.functional.cross_entropy(
            model(data.pairs[indices]), data.labels[indices]
        )
        return torch.autograd.grad(loss, model.feature_weight)[0] @ model.input_basis

    point = model.feature_weight.detach() @ model.input_basis
    train_gradient = rowwise_tangent_projection(
        raw_feature_gradient(data.train_idx), point
    )
    full_gradient = rowwise_tangent_projection(
        raw_feature_gradient(torch.arange(config.p * config.p)), point
    )
    train_direction, _ = transformed_direction(
        train_gradient, "polar_muon", strict_rank=False
    )
    full_direction, _ = transformed_direction(
        full_gradient, "polar_muon", strict_rank=False
    )
    train_posttangent = rowwise_tangent_projection(train_direction, point)
    full_posttangent = rowwise_tangent_projection(full_direction, point)
    train_velocity = feature_induced_logit_velocity(
        model, data.pairs, train_direction @ model.input_basis.T
    )
    full_velocity = feature_induced_logit_velocity(
        model, data.pairs, full_direction @ model.input_basis.T
    )
    train_posttangent_velocity = feature_induced_logit_velocity(
        model, data.pairs, train_posttangent @ model.input_basis.T
    )
    full_posttangent_velocity = feature_induced_logit_velocity(
        model, data.pairs, full_posttangent @ model.input_basis.T
    )
    train_velocity -= train_velocity.mean(dim=0, keepdim=True)
    full_velocity -= full_velocity.mean(dim=0, keepdim=True)
    train_posttangent_velocity -= train_posttangent_velocity.mean(
        dim=0, keepdim=True
    )
    full_posttangent_velocity -= full_posttangent_velocity.mean(
        dim=0, keepdim=True
    )
    rule = torch.nn.functional.one_hot(data.labels, config.p).double() - 1.0 / config.p

    def cosine(first: torch.Tensor, second: torch.Tensor) -> float:
        first = first.reshape(-1)
        second = second.reshape(-1)
        return float(
            torch.dot(first, second)
            / (torch.linalg.vector_norm(first) * torch.linalg.vector_norm(second))
        )

    assert metrics[
        "theory_train_vs_full_optimizer_feature_direction_cosine"
    ] == pytest.approx(cosine(train_direction, full_direction), abs=2.0e-14)
    assert metrics[
        "theory_train_vs_full_optimizer_feature_logit_velocity_cosine"
    ] == pytest.approx(cosine(train_velocity, full_velocity), abs=2.0e-14)
    assert metrics[
        "theory_train_optimizer_feature_update_rule_cosine"
    ] == pytest.approx(cosine(-train_velocity, rule), abs=2.0e-14)
    assert metrics[
        "theory_full_optimizer_feature_update_rule_cosine"
    ] == pytest.approx(cosine(-full_velocity, rule), abs=2.0e-14)
    assert metrics["theory_train_optimizer_feature_direction_fro"] == pytest.approx(
        torch.linalg.matrix_norm(train_direction).item(), abs=2.0e-14
    )
    assert metrics[
        "theory_train_vs_full_optimizer_posttangent_feature_direction_cosine"
    ] == pytest.approx(cosine(train_posttangent, full_posttangent), abs=2.0e-14)
    assert metrics[
        "theory_train_vs_full_optimizer_posttangent_feature_logit_velocity_cosine"
    ] == pytest.approx(
        cosine(train_posttangent_velocity, full_posttangent_velocity),
        abs=2.0e-14,
    )
    assert metrics[
        "theory_train_optimizer_posttangent_feature_update_rule_cosine"
    ] == pytest.approx(cosine(-train_posttangent_velocity, rule), abs=2.0e-14)
    assert metrics[
        "theory_full_optimizer_posttangent_feature_update_rule_cosine"
    ] == pytest.approx(cosine(-full_posttangent_velocity, rule), abs=2.0e-14)
    assert metrics[
        "theory_train_optimizer_posttangent_feature_direction_fro"
    ] == pytest.approx(
        torch.linalg.matrix_norm(train_posttangent).item(), abs=2.0e-14
    )
    train_posttangent_projection = fourier_logit_velocity_projection(
        train_posttangent_velocity, config.p
    )
    for name, expected in train_posttangent_projection.items():
        assert metrics[
            f"theory_train_optimizer_posttangent_feature_logit_velocity_{name}"
        ] == pytest.approx(expected, abs=2.0e-14)
    assert not torch.allclose(train_direction, train_posttangent)


def test_finite_head_entry_fields_match_literal_tangent_certificate() -> None:
    config = tiny_config(
        optimizer="polar_feature_gd_head",
        activation="square",
        feature_geometry="row_sphere",
        steps=0,
    )
    device = torch.device("cpu")
    torch.manual_seed(config.seed)
    model = QuadraticOneHiddenLayer(
        config.p,
        config.width,
        config.feature_init_scale,
        config.head_init_scale,
        device=device,
        dtype=torch.float64,
        activation=config.activation,
    )
    optimizer = EndToEndMatrixOptimizer(model, config)
    data = make_data(config.p, config.train_fraction, config.split_seed, device)
    metrics = theory_gradient_diagnostics(
        model,
        data.pairs,
        data.labels,
        data.train_idx,
        config=config,
        optimizer_state=optimizer,
    )

    train_loss = F.cross_entropy(
        model(data.pairs[data.train_idx]), data.labels[data.train_idx]
    )
    train_gradient = torch.autograd.grad(train_loss, model.feature_weight)[0]
    teacher = F.one_hot(data.labels, config.p).double() - 1.0 / config.p
    population_correlation = (model(data.pairs) * teacher).sum() / (config.p**2)
    population_gradient = torch.autograd.grad(
        population_correlation, model.feature_weight
    )[0]
    point = model.feature_weight.detach() @ model.input_basis
    entry_j = rowwise_tangent_projection(
        population_gradient @ model.input_basis, point
    )
    entry_k = -rowwise_tangent_projection(
        train_gradient @ model.input_basis, point
    )
    polar_k, diagnostics = support_polar_factor(entry_k, strict_rank=False)
    posttangent_polar_k = rowwise_tangent_projection(polar_k, point)
    nuclear = torch.sum(entry_k * polar_k).item()
    discrepancy = torch.linalg.matrix_norm(entry_j - entry_k).item()
    rank = int(diagnostics["rank"])
    direct = torch.sum(entry_j * posttangent_polar_k).item()
    normalized = direct / (
        torch.linalg.matrix_norm(entry_j).item()
        * torch.linalg.matrix_norm(posttangent_polar_k).item()
    )

    assert metrics["theory_finite_head_entry_k_nuclear_norm"] == pytest.approx(
        nuclear, abs=2.0e-14
    )
    assert metrics["theory_finite_head_entry_k_rank"] == rank
    assert metrics["theory_finite_head_entry_j_minus_k_fro"] == pytest.approx(
        discrepancy, abs=2.0e-14
    )
    assert metrics[
        "theory_finite_head_entry_margin_lower_bound"
    ] == pytest.approx(nuclear - np.sqrt(rank) * discrepancy, abs=2.0e-14)
    assert metrics["theory_finite_head_entry_direct_cone"] == pytest.approx(
        direct, abs=2.0e-14
    )
    assert metrics[
        "theory_finite_head_entry_normalized_direct_cone"
    ] == pytest.approx(normalized, abs=2.0e-14)
    assert direct + 2.0e-14 >= metrics[
        "theory_finite_head_entry_margin_lower_bound"
    ]


def test_posttangent_direction_is_first_order_row_retraction_velocity() -> None:
    generator = torch.Generator().manual_seed(163)
    point = torch.randn(9, 12, generator=generator, dtype=torch.float64)
    point *= (
        torch.linspace(0.7, 1.4, 9, dtype=torch.float64)
        / torch.linalg.vector_norm(point, dim=1)
    )[:, None]
    gradient = torch.randn(9, 12, generator=generator, dtype=torch.float64)
    tangent_gradient = rowwise_tangent_projection(gradient, point)
    transformed, _ = transformed_direction(
        tangent_gradient, "polar_muon", strict_rank=False
    )
    posttangent = rowwise_tangent_projection(transformed, point)
    epsilon = 1.0e-7
    retracted = retract_rows_to_fixed_norms(
        point - epsilon * transformed,
        torch.linalg.vector_norm(point, dim=1),
    )
    finite_velocity = (retracted - point) / epsilon

    torch.testing.assert_close(
        finite_velocity,
        -posttangent,
        rtol=2.0e-7,
        atol=8.0e-8,
    )
    torch.testing.assert_close(
        (posttangent * point).sum(dim=1),
        torch.zeros(point.shape[0], dtype=point.dtype),
        rtol=0.0,
        atol=2.0e-15,
    )


def test_momentum_theory_counterfactual_does_not_mutate_optimizer_state() -> None:
    config = tiny_config(
        optimizer="polar_muon",
        momentum=0.7,
        theory_gradient_observer=True,
        steps=0,
    )
    device = torch.device("cpu")
    torch.manual_seed(config.seed)
    model = QuadraticOneHiddenLayer(
        config.p,
        config.width,
        config.feature_init_scale,
        config.head_init_scale,
        device=device,
        dtype=torch.float64,
    )
    matrix_optimizer = EndToEndMatrixOptimizer(model, config)
    with torch.no_grad():
        matrix_optimizer.feature_momentum.copy_(
            torch.randn_like(matrix_optimizer.feature_momentum)
        )
    momentum_before = matrix_optimizer.feature_momentum.clone()
    data = make_data(config.p, config.train_fraction, config.split_seed, device)
    theory_gradient_diagnostics(
        model,
        data.pairs,
        data.labels,
        data.train_idx,
        config=config,
        optimizer_state=matrix_optimizer,
    )
    assert torch.equal(matrix_optimizer.feature_momentum, momentum_before)


def test_rowwise_tangent_projection_is_orthogonal_and_removes_radial_rows() -> None:
    generator = torch.Generator().manual_seed(157)
    point = torch.randn(8, 11, generator=generator, dtype=torch.float64)
    gradient = torch.randn(8, 11, generator=generator, dtype=torch.float64)
    tangent = rowwise_tangent_projection(gradient, point)
    torch.testing.assert_close(
        (tangent * point).sum(dim=1),
        torch.zeros(8, dtype=torch.float64),
        rtol=0.0,
        atol=2.0e-15,
    )
    radial = torch.arange(1, 9, dtype=torch.float64)[:, None] * point
    torch.testing.assert_close(
        rowwise_tangent_projection(radial, point),
        torch.zeros_like(point),
        rtol=0.0,
        atol=2.0e-15,
    )

    targets = torch.linspace(0.5, 1.2, 8, dtype=torch.float64)
    retracted = retract_rows_to_fixed_norms(point + 0.1 * tangent, targets)
    torch.testing.assert_close(
        torch.linalg.vector_norm(retracted, dim=1),
        targets,
        rtol=2.0e-15,
        atol=2.0e-15,
    )


def test_quadratic_hidden_power_is_sum_difference_symmetric() -> None:
    p = 7
    model = QuadraticOneHiddenLayer(
        p, 9, 0.3, 0.1, device=torch.device("cpu"), dtype=torch.float64
    )
    a = torch.arange(p).repeat_interleave(p)
    b = torch.arange(p).repeat(p)
    metrics = fourier_alignment(model.hidden(torch.stack((a, b), dim=1)), p)
    assert metrics["fourier_addition_fraction"] == pytest.approx(
        metrics["fourier_difference_fraction"], abs=1e-12
    )


def test_row_sphere_configuration_rejects_incompatible_scale_changes() -> None:
    with pytest.raises(ValueError, match="weight_decay_feature must be zero"):
        tiny_config(
            feature_geometry="row_sphere", weight_decay_feature=1.0e-3
        ).validate()
    with pytest.raises(ValueError, match="positive feature_init_scale"):
        tiny_config(
            feature_geometry="row_sphere", feature_init_scale=0.0
        ).validate()


def test_head_only_configuration_rejects_feature_decay() -> None:
    with pytest.raises(ValueError, match="head_only freezes feature weights"):
        tiny_config(
            optimizer="head_only", weight_decay_feature=1.0e-3
        ).validate()


def test_row_sphere_update_uses_tangent_gradient_before_fro_transform() -> None:
    config = tiny_config(
        optimizer="fro_feature_gd_head",
        feature_geometry="row_sphere",
        steps=0,
    )
    device = torch.device("cpu")
    torch.manual_seed(config.seed)
    model = QuadraticOneHiddenLayer(
        config.p,
        config.width,
        config.feature_init_scale,
        config.head_init_scale,
        device=device,
        dtype=torch.float64,
    )
    matrix_optimizer = EndToEndMatrixOptimizer(model, config)
    data = make_data(config.p, config.train_fraction, config.split_seed, device)
    loss = torch.nn.functional.cross_entropy(
        model(data.pairs[data.train_idx]), data.labels[data.train_idx]
    )
    loss.backward()

    point = model.feature_weight.detach() @ model.input_basis
    gradient = model.feature_weight.grad.detach() @ model.input_basis
    tangent = rowwise_tangent_projection(gradient, point)
    direction, _ = transformed_direction(
        tangent, "fro_ngd", strict_rank=False
    )
    expected = retract_rows_to_fixed_norms(
        point - config.lr_feature * direction,
        torch.linalg.vector_norm(point, dim=1),
    )
    matrix_optimizer.step()
    actual = model.feature_weight.detach() @ model.input_basis
    torch.testing.assert_close(actual, expected, rtol=2.0e-15, atol=2.0e-15)


@pytest.mark.parametrize(
    "optimizer", ["polar_feature_gd_head", "fro_feature_gd_head", "gd"]
)
def test_row_sphere_optimizer_preserves_initialized_identifiable_norms(
    optimizer: str,
) -> None:
    config = tiny_config(
        optimizer=optimizer,
        activation="relu",
        feature_geometry="row_sphere",
        steps=0,
    )
    config.validate()
    device = torch.device("cpu")
    torch.manual_seed(config.seed)
    model = QuadraticOneHiddenLayer(
        config.p,
        config.width,
        config.feature_init_scale,
        config.head_init_scale,
        device=device,
        dtype=torch.float64,
        activation=config.activation,
    )
    matrix_optimizer = EndToEndMatrixOptimizer(model, config)
    data = make_data(config.p, config.train_fraction, config.split_seed, device)
    initial_norms = matrix_optimizer.feature_row_norm_targets.clone()

    for _ in range(4):
        matrix_optimizer.zero_grad()
        loss = torch.nn.functional.cross_entropy(
            model(data.pairs[data.train_idx]), data.labels[data.train_idx]
        )
        loss.backward()
        diagnostics = matrix_optimizer.step()
        identifiable = model.feature_weight @ model.input_basis
        torch.testing.assert_close(
            torch.linalg.vector_norm(identifiable, dim=1),
            initial_norms,
            rtol=3.0e-15,
            atol=3.0e-15,
        )
        assert diagnostics["feature_tangent_max_abs_cosine"] < 2.0e-14
        assert diagnostics["feature_radial_gradient_fro"] >= 0.0
        assert diagnostics["feature_retraction_correction_fro"] >= 0.0
        assert diagnostics["feature_actual_step_fro"] > 0.0
        assert (
            diagnostics["feature_post_retraction_row_norm_max_abs_error"]
            < 2.0e-15
        )

    null = torch.cat(
        (torch.ones(config.p), -torch.ones(config.p))
    ).to(dtype=torch.float64)
    torch.testing.assert_close(
        model.feature_weight @ null,
        torch.zeros(config.width, dtype=torch.float64),
        rtol=0.0,
        atol=2.0e-14,
    )


def test_architecture_cli_and_config_round_trip(tmp_path: Path) -> None:
    parser = build_parser()
    args = parser.parse_args(
        [
            "single",
            "--activation",
            "softplus",
            "--feature-geometry",
            "row_sphere",
            "--weight-decay-feature",
            "0",
            "--theory-gradient-observer",
            "--rule-margin-observer",
            "--seed",
            "23",
            "--width",
            "17",
            "--optimizer",
            "fro_feature_gd_head",
            "--lr-feature",
            "0.03",
            "--output-dir",
            str(tmp_path / "single"),
        ]
    )
    config = RunConfig(
        seed=args.seed,
        width=args.width,
        optimizer=args.optimizer,
        lr_feature=args.lr_feature,
        **shared_kwargs(args),
    )
    config.validate()
    assert config.activation == "softplus"
    assert config.feature_geometry == "row_sphere"
    assert config.theory_gradient_observer is True
    assert config.rule_margin_observer is True
    assert RunConfig(**asdict(config)) == config
    assert "actsoftplus_geomrow_sphere" in run_id(config)

    grid_args = parser.parse_args(
        [
            "grid",
            "--activations",
            "square,relu",
            "--feature-geometries",
            "euclidean,row_sphere",
            "--output-dir",
            str(tmp_path / "grid"),
        ]
    )
    assert grid_args.activations == "square,relu"
    assert grid_args.feature_geometries == "euclidean,row_sphere"


def test_hidden_state_agop_matches_centered_logit_jacobian_exactly() -> None:
    generator = torch.Generator().manual_seed(101)
    p, width = 7, 5
    head = torch.randn(p, width, generator=generator, dtype=torch.float64)
    hidden = torch.randn(width, generator=generator, dtype=torch.float64)

    def centered_logits(value: torch.Tensor) -> torch.Tensor:
        logits = head @ value / np.sqrt(width)
        return logits - logits.mean()

    jacobian = torch.autograd.functional.jacobian(centered_logits, hidden)
    expected = jacobian.T @ jacobian
    torch.testing.assert_close(
        hidden_state_agop(head), expected, rtol=1.0e-13, atol=1.0e-13
    )


def test_agop_subspace_rank_uses_allweights_relative_cutoff() -> None:
    matrix = torch.diag(torch.tensor([1.0, 1.0e-11], dtype=torch.float64))
    assert numerical_column_basis(matrix).shape[1] == 2
    assert numerical_column_basis(matrix, relative_tolerance=1.0e-10).shape[1] == 1


def synthetic_fourier_observer_inputs(
    p: int = 7,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    sum_basis, _, _ = real_fourier_bases(p, torch.device("cpu"), torch.float64)
    width = p - 1
    output_basis = output_gauge_basis(p, torch.device("cpu"), torch.float64)
    scales = torch.arange(width, 0, -1, dtype=torch.float64)
    head = output_basis @ torch.diag(scales)
    a = torch.arange(p).repeat_interleave(p)
    b = torch.arange(p).repeat(p)
    labels = (a + b) % p
    train_idx = torch.arange(2 * p, dtype=torch.long)
    test_idx = torch.arange(2 * p, p * p, dtype=torch.long)
    return sum_basis, head, labels.long(), train_idx, test_idx


def test_agop_selected_fourier_purity_coverage_and_train_only_probe() -> None:
    p = 7
    hidden, head, labels, train_idx, test_idx = synthetic_fourier_observer_inputs(p)
    metrics = agop_fourier_alignment(
        hidden,
        head,
        p,
        labels,
        train_idx,
        test_idx,
        ranks=(2,),
        ridge_alphas=(1.0e-10,),
    )

    assert metrics["agop_numerical_rank"] == p - 1
    assert metrics["agop_top2_fourier_sum_purity"] == pytest.approx(
        1.0, abs=1.0e-12
    )
    assert metrics["agop_top2_fourier_sum_coverage"] == pytest.approx(
        2.0 / (p - 1), abs=1.0e-12
    )
    assert metrics["agop_top2_full_grid_fourier_sum_purity"] == pytest.approx(
        1.0, abs=1.0e-12
    )
    assert metrics[
        "agop_top2_full_grid_fourier_difference_coverage"
    ] == pytest.approx(0.0, abs=1.0e-12)
    assert metrics["agop_top_rule_fourier_sum_purity"] == pytest.approx(
        1.0, abs=1.0e-12
    )
    assert metrics["agop_top_rule_fourier_sum_coverage"] == pytest.approx(
        1.0, abs=1.0e-12
    )
    assert metrics["agop_top_rule_heldout_fourier_r2"] > 1.0 - 1.0e-16
    assert metrics["agop_top_rule_heldout_modular_accuracy"] == 1.0
    _, _, codebook = real_fourier_bases(
        p, torch.device("cpu"), torch.float64
    )
    centered_hidden = hidden - hidden.mean(dim=0, keepdim=True)
    expected_full_hidden_probe = _train_only_fourier_probe(
        centered_hidden,
        labels,
        train_idx,
        test_idx,
        codebook,
        (1.0e-10,),
        3,
        314159,
    )
    for suffix in (
        "heldout_modular_accuracy",
        "heldout_fourier_r2",
        "ridge_alpha",
        "inner_cv_r2",
    ):
        assert metrics[f"full_hidden_train_only_refit_{suffix}"] == pytest.approx(
            expected_full_hidden_probe[suffix], abs=0.0
        )


def test_agop_fourier_observer_is_orthogonal_coordinate_and_logit_gauge_invariant() -> None:
    generator = torch.Generator().manual_seed(203)
    p, width = 7, 5
    hidden = torch.randn(p * p, width, generator=generator, dtype=torch.float64)
    head = torch.randn(p, width, generator=generator, dtype=torch.float64)
    head -= head.mean(dim=0, keepdim=True)
    rotation, _ = torch.linalg.qr(
        torch.randn(width, width, generator=generator, dtype=torch.float64)
    )
    gauge = torch.randn(width, generator=generator, dtype=torch.float64)
    a = torch.arange(p).repeat_interleave(p)
    b = torch.arange(p).repeat(p)
    labels = ((a + b) % p).long()
    train_idx = torch.arange(0, p * p, 3, dtype=torch.long)
    mask = torch.ones(p * p, dtype=torch.bool)
    mask[train_idx] = False
    test_idx = torch.arange(p * p, dtype=torch.long)[mask]
    kwargs = {
        "ranks": (2, 4),
        "ridge_alphas": (1.0e-4, 1.0e-2),
        "probe_seed": 17,
    }
    original = agop_fourier_alignment(
        hidden, head, p, labels, train_idx, test_idx, **kwargs
    )
    rotated = agop_fourier_alignment(
        hidden @ rotation,
        head @ rotation,
        p,
        labels,
        train_idx,
        test_idx,
        **kwargs,
    )
    shifted = agop_fourier_alignment(
        hidden,
        head + torch.ones(p, 1, dtype=torch.float64) @ gauge[None, :],
        p,
        labels,
        train_idx,
        test_idx,
        **kwargs,
    )
    assert original.keys() == rotated.keys() == shifted.keys()
    for name in original:
        if isinstance(original[name], int):
            assert original[name] == shifted[name]
            if "random" not in name and not name.startswith(
                "full_hidden_train_only_refit_"
            ):
                assert original[name] == rotated[name]
        else:
            assert shifted[name] == pytest.approx(original[name], abs=2.0e-10)
            if "random" not in name and not name.startswith(
                "full_hidden_train_only_refit_"
            ):
                assert rotated[name] == pytest.approx(original[name], abs=2.0e-10)


def test_probe_selection_does_not_use_heldout_labels() -> None:
    p = 7
    hidden, head, labels, train_idx, test_idx = synthetic_fourier_observer_inputs(p)
    altered_labels = labels.clone()
    altered_labels[test_idx] = (altered_labels[test_idx] + 1) % p
    kwargs = {
        "ranks": (2,),
        "ridge_alphas": (1.0e-10, 1.0e-4, 1.0e-2),
        "probe_folds": 3,
        "probe_seed": 29,
    }
    original = agop_fourier_alignment(
        hidden, head, p, labels, train_idx, test_idx, **kwargs
    )
    altered = agop_fourier_alignment(
        hidden, head, p, altered_labels, train_idx, test_idx, **kwargs
    )
    for rank in (2, p - 1):
        prefix = f"agop_top{rank}"
        assert altered[f"{prefix}_ridge_alpha"] == original[f"{prefix}_ridge_alpha"]
        assert altered[f"{prefix}_inner_cv_r2"] == pytest.approx(
            original[f"{prefix}_inner_cv_r2"], abs=0.0
        )
        assert altered[f"{prefix}_fourier_sum_purity"] == pytest.approx(
            original[f"{prefix}_fourier_sum_purity"], abs=0.0
        )
    assert original["agop_top_rule_heldout_modular_accuracy"] == 1.0
    assert altered["agop_top_rule_heldout_modular_accuracy"] == 0.0
    assert altered["full_hidden_train_only_refit_ridge_alpha"] == original[
        "full_hidden_train_only_refit_ridge_alpha"
    ]
    assert altered["full_hidden_train_only_refit_inner_cv_r2"] == pytest.approx(
        original["full_hidden_train_only_refit_inner_cv_r2"], abs=0.0
    )


def test_random_hidden_controls_are_seeded_matched_and_nontrivial() -> None:
    p = 7
    sum_basis, difference_basis, _ = real_fourier_bases(
        p, torch.device("cpu"), torch.float64
    )
    hidden = torch.cat((sum_basis, difference_basis), dim=1)
    output_basis = output_gauge_basis(p, torch.device("cpu"), torch.float64)
    scales = torch.arange(p - 1, 0, -1, dtype=torch.float64)
    head = torch.cat(
        (
            output_basis @ torch.diag(scales),
            torch.zeros(p, p - 1, dtype=torch.float64),
        ),
        dim=1,
    )
    a = torch.arange(p).repeat_interleave(p)
    b = torch.arange(p).repeat(p)
    labels = ((a + b) % p).long()
    train_idx = torch.arange(2 * p, dtype=torch.long)
    test_idx = torch.arange(2 * p, p * p, dtype=torch.long)
    kwargs = {
        "ranks": (p - 1,),
        "ridge_alphas": (1.0e-8,),
        "probe_seed": 311,
        "random_controls": 8,
    }
    first = agop_fourier_alignment(
        hidden, head, p, labels, train_idx, test_idx, **kwargs
    )
    second = agop_fourier_alignment(
        hidden, head, p, labels, train_idx, test_idx, **kwargs
    )
    assert first == second
    prefix = f"agop_top{p - 1}"
    assert first[f"{prefix}_fourier_sum_purity"] == pytest.approx(
        1.0, abs=1.0e-12
    )
    assert 0.0 < first[f"{prefix}_random_mean_fourier_sum_purity"] < 0.9
    assert first[f"{prefix}_top_minus_random_fourier_sum_purity"] > 0.1


@pytest.mark.parametrize(
    "optimizer",
    [
        "polar_muon",
        "fro_ngd",
        "polar_feature_gd_head",
        "fro_feature_gd_head",
        "gd",
    ],
)
def test_end_to_end_run_updates_both_matrices_and_is_finite(optimizer: str) -> None:
    result = run_experiment(tiny_config(optimizer=optimizer))
    rows = result["trajectory"]
    assert len(rows) == 4
    assert rows[-1]["feature_update_fro"] > 0.0
    assert rows[-1]["head_update_fro"] > 0.0
    assert all(np.isfinite(row["train_ce"]) for row in rows)
    assert all(np.isfinite(row["train_margin_min"]) for row in rows)
    assert all(np.isfinite(row["test_margin_min"]) for row in rows)
    assert all(
        row["head_empirical_trace_smoothness_envelope"]
        >= row["head_empirical_gram_smoothness"]
        for row in rows
    )
    assert all(
        0.0 <= row["full_hidden_train_only_refit_heldout_modular_accuracy"] <= 1.0
        for row in rows
    )
    assert all(0.0 <= row["fourier_addition_fraction"] <= 1.0 for row in rows)
    assert all(0.0 <= row["fourier_sum_subspace_alignment"] <= 1.0 for row in rows)
    assert all(0.0 <= row["fourier_addition_covered_fraction"] <= 1.0 for row in rows)
    assert all(0.0 <= row["agop_top_rule_fourier_sum_purity"] <= 1.0 for row in rows)
    assert all(0.0 <= row["agop_top_rule_fourier_sum_coverage"] <= 1.0 for row in rows)
    assert all(0.0 <= row["agop_top_rule_energy_fraction"] <= 1.0 for row in rows)
    assert all(
        row["fourier_sum_full_hidden_span_coverage"]
        == row["fourier_sum_subspace_alignment"]
        for row in rows
    )
    assert all(
        row["fourier_sum_oracle_headfit_residual"]
        == row["fourier_sum_headfit_risk"]
        for row in rows
    )


def test_head_decay_diagnostics_reconstruct_the_exact_head_step() -> None:
    learning_rate = 0.2
    decay = 0.03
    result = run_experiment(
        tiny_config(
            steps=1,
            optimizer="polar_feature_gd_head",
            lr_head=learning_rate,
            weight_decay_head=decay,
        )
    )
    initial, updated = result["trajectory"]
    expected_decay_update = learning_rate * decay * initial["head_weight_fro"]
    assert updated["head_decay_update_fro"] == pytest.approx(
        expected_decay_update, rel=2e-14, abs=2e-14
    )
    assert updated["head_actual_step_fro"] == pytest.approx(
        learning_rate * updated["head_combined_direction_fro"],
        rel=2e-14,
        abs=2e-14,
    )
    assert updated["head_decay_balance_ratio"] == pytest.approx(
        updated["head_update_fro"] / expected_decay_update,
        rel=2e-14,
        abs=2e-14,
    )
    gradient_norm = updated["head_direction_fro"]
    decay_direction_norm = updated["head_decay_update_fro"] / learning_rate
    reconstructed_squared_norm = (
        gradient_norm**2
        + decay_direction_norm**2
        + 2.0
        * gradient_norm
        * decay_direction_norm
        * updated["head_direction_decay_cosine"]
    )
    assert updated["head_combined_direction_fro"] ** 2 == pytest.approx(
        reconstructed_squared_norm, rel=2e-13, abs=2e-13
    )


def test_signed_first_order_ce_diagnostics_match_literal_block_derivatives() -> None:
    config = tiny_config(
        steps=0,
        optimizer="polar_feature_gd_head",
        activation="square",
        feature_geometry="row_sphere",
        lr_feature=0.013,
        lr_head=0.17,
        weight_decay_head=0.04,
    )
    device = torch.device("cpu")
    torch.manual_seed(config.seed)
    model = QuadraticOneHiddenLayer(
        config.p,
        config.width,
        config.feature_init_scale,
        config.head_init_scale,
        device=device,
        dtype=torch.float64,
        activation=config.activation,
    )
    optimizer = EndToEndMatrixOptimizer(model, config)
    data = make_data(config.p, config.train_fraction, config.split_seed, device)
    loss = F.cross_entropy(
        model(data.pairs[data.train_idx]), data.labels[data.train_idx]
    )
    loss.backward()

    feature_gradient = model.feature_weight.grad @ model.input_basis
    feature_point = model.feature_weight.detach() @ model.input_basis
    feature_tangent = rowwise_tangent_projection(
        feature_gradient, feature_point
    )
    feature_direction, _ = transformed_direction(
        feature_tangent, "polar_muon", strict_rank=False
    )
    expected_nuclear = torch.linalg.matrix_norm(
        feature_tangent, ord="nuc"
    ).item()
    expected_feature_change = torch.sum(
        feature_tangent * (-config.lr_feature * feature_direction)
    ).item()

    head_gradient = model.output_basis.T @ model.head_weight.grad
    head_direction, _ = transformed_direction(
        head_gradient, "gd", strict_rank=False
    )
    head_combined = (
        model.output_basis @ head_direction
        + config.weight_decay_head * model.head_weight.detach()
    )
    expected_head_dot = torch.sum(
        (model.output_basis @ head_gradient) * head_combined
    ).item()
    expected_head_change = -config.lr_head * expected_head_dot

    diagnostics = optimizer.step()

    assert diagnostics[
        "feature_tangent_gradient_nuclear_norm"
    ] == pytest.approx(expected_nuclear, rel=2.0e-14, abs=2.0e-14)
    assert diagnostics[
        "feature_pre_retraction_signed_first_order_ce_change"
    ] == pytest.approx(expected_feature_change, rel=2.0e-14, abs=2.0e-14)
    assert expected_feature_change == pytest.approx(
        -config.lr_feature * expected_nuclear,
        rel=2.0e-14,
        abs=2.0e-14,
    )
    assert diagnostics[
        "head_data_gradient_dot_combined_direction"
    ] == pytest.approx(expected_head_dot, rel=2.0e-14, abs=2.0e-14)
    assert diagnostics[
        "head_signed_first_order_ce_change"
    ] == pytest.approx(expected_head_change, rel=2.0e-14, abs=2.0e-14)
    assert diagnostics[
        "total_signed_first_order_ce_change"
    ] == pytest.approx(
        expected_feature_change + expected_head_change,
        rel=2.0e-14,
        abs=2.0e-14,
    )


def test_signed_first_order_instrumentation_preserves_legacy_update() -> None:
    config = tiny_config(
        steps=0,
        optimizer="polar_feature_gd_head",
        activation="square",
        feature_geometry="row_sphere",
        lr_feature=0.021,
        lr_head=0.13,
        weight_decay_head=0.025,
    )
    device = torch.device("cpu")
    torch.manual_seed(config.seed)
    model = QuadraticOneHiddenLayer(
        config.p,
        config.width,
        config.feature_init_scale,
        config.head_init_scale,
        device=device,
        dtype=torch.float64,
        activation=config.activation,
    )
    optimizer = EndToEndMatrixOptimizer(model, config)
    data = make_data(config.p, config.train_fraction, config.split_seed, device)
    loss = F.cross_entropy(
        model(data.pairs[data.train_idx]), data.labels[data.train_idx]
    )
    loss.backward()

    point = model.feature_weight.detach() @ model.input_basis
    feature_gradient = model.feature_weight.grad @ model.input_basis
    feature_tangent = rowwise_tangent_projection(feature_gradient, point)
    feature_direction, _ = transformed_direction(
        feature_tangent, "polar_muon", strict_rank=False
    )
    expected_feature = retract_rows_to_fixed_norms(
        point - config.lr_feature * feature_direction,
        optimizer.feature_row_norm_targets,
    ) @ model.input_basis.T

    head_gradient = model.output_basis.T @ model.head_weight.grad
    head_direction, _ = transformed_direction(
        head_gradient, "gd", strict_rank=False
    )
    expected_head = model.head_weight.detach().clone()
    expected_head.mul_(1.0 - config.lr_head * config.weight_decay_head)
    expected_head.add_(
        model.output_basis @ head_direction, alpha=-config.lr_head
    )
    expected_head.sub_(expected_head.mean(dim=0, keepdim=True))

    optimizer.step()

    assert torch.equal(model.feature_weight, expected_feature)
    assert torch.equal(model.head_weight, expected_head)


def test_signed_first_order_fields_have_initial_observation_defaults() -> None:
    result = run_experiment(
        tiny_config(
            steps=0,
            optimizer="polar_feature_gd_head",
            activation="square",
            feature_geometry="row_sphere",
        )
    )
    initial = result["trajectory"][0]
    for key in (
        "feature_tangent_gradient_nuclear_norm",
        "feature_tangent_gradient_retained_nuclear_norm",
        "feature_data_direction_pairing",
        "feature_pre_retraction_signed_first_order_ce_change",
        "feature_retraction_gradient_inner_product",
        "feature_exact_gradient_increment_inner_product",
        "head_data_gradient_dot_combined_direction",
        "head_signed_first_order_ce_change",
        "total_signed_first_order_ce_change",
        "head_exact_gradient_increment_inner_product",
        "full_exact_gradient_increment_inner_product",
        "training_ce_realized_one_step_change",
        "training_ce_first_order_taylor_remainder",
    ):
        assert math.isnan(initial[key])


@pytest.mark.parametrize("feature_geometry", ["euclidean", "row_sphere"])
def test_head_only_bitwise_freezes_features_and_updates_head(
    feature_geometry: str,
) -> None:
    initial = run_experiment(
        tiny_config(
            optimizer="head_only",
            steps=0,
            lr_head=0.1,
            weight_decay_feature=0.0,
            weight_decay_head=0.01,
            feature_geometry=feature_geometry,
        )
    )
    trained = run_experiment(
        tiny_config(
            optimizer="head_only",
            steps=4,
            lr_head=0.1,
            weight_decay_feature=0.0,
            weight_decay_head=0.01,
            feature_geometry=feature_geometry,
        )
    )

    assert torch.equal(
        initial["model"].feature_weight,
        trained["model"].feature_weight,
    )
    assert not torch.equal(
        initial["model"].head_weight,
        trained["model"].head_weight,
    )
    update_rows = trained["trajectory"][1:]
    assert update_rows
    assert all(row["feature_gradient_fro"] > 0.0 for row in update_rows)
    assert all(row["feature_direction_fro"] == 0.0 for row in update_rows)
    assert all(row["feature_update_fro"] == 0.0 for row in update_rows)
    assert all(row["feature_actual_step_fro"] == 0.0 for row in update_rows)
    assert all(row["head_update_fro"] > 0.0 for row in update_rows)
    assert all(
        row["feature_weight_fro"]
        == trained["trajectory"][0]["feature_weight_fro"]
        for row in update_rows
    )


def test_head_only_cli_choice_round_trips(tmp_path: Path) -> None:
    args = build_parser().parse_args(
        [
            "single",
            "--optimizer",
            "head_only",
            "--weight-decay-feature",
            "0",
            "--output-dir",
            str(tmp_path / "head-only"),
        ]
    )
    config = RunConfig(
        seed=args.seed,
        width=args.width,
        optimizer=args.optimizer,
        lr_feature=args.lr_feature,
        **shared_kwargs(args),
    )
    config.validate()
    assert config.optimizer == "head_only"
    assert config.weight_decay_feature == 0.0


@pytest.mark.parametrize("activation", ["relu", "softplus"])
@pytest.mark.parametrize("feature_geometry", ["euclidean", "row_sphere"])
def test_observers_are_compatible_with_all_new_architectures(
    activation: str,
    feature_geometry: str,
) -> None:
    result = run_experiment(
        tiny_config(
            steps=1,
            optimizer="fro_feature_gd_head",
            activation=activation,
            feature_geometry=feature_geometry,
        )
    )
    assert result["model"].activation == activation
    for row in result["trajectory"]:
        assert np.isfinite(row["fourier_sum_subspace_alignment"])
        assert np.isfinite(row["agop_trace"])
        assert np.isfinite(row["agop_top_rule_fourier_sum_purity"])
        assert np.isfinite(row["agop_top_rule_heldout_fourier_r2"])
        assert np.isfinite(row["feature_identifiable_row_norm_mean"])
    if feature_geometry == "row_sphere":
        assert max(
            row["feature_row_norm_max_relative_error"]
            for row in result["trajectory"]
        ) < 5.0e-15
        assert np.isfinite(
            result["trajectory"][-1]["feature_retraction_correction_fro"]
        )


def test_logging_cadence_is_an_observer_only() -> None:
    dense = run_experiment(tiny_config(log_every=1, steps=4))
    sparse = run_experiment(tiny_config(log_every=3, steps=4))
    torch.testing.assert_close(
        dense["model"].feature_weight, sparse["model"].feature_weight, rtol=0.0, atol=0.0
    )
    torch.testing.assert_close(
        dense["model"].head_weight, sparse["model"].head_weight, rtol=0.0, atol=0.0
    )


def test_theory_gradient_observer_is_opt_in_and_update_invariant() -> None:
    disabled = run_experiment(tiny_config(steps=2))
    enabled = run_experiment(
        tiny_config(steps=2, theory_gradient_observer=True)
    )
    torch.testing.assert_close(
        disabled["model"].feature_weight,
        enabled["model"].feature_weight,
        rtol=0.0,
        atol=0.0,
    )
    torch.testing.assert_close(
        disabled["model"].head_weight,
        enabled["model"].head_weight,
        rtol=0.0,
        atol=0.0,
    )
    assert not any(
        key.startswith("theory_") for key in disabled["trajectory"][0]
    )
    assert all(
        "theory_train_vs_full_feature_raw_gradient_cosine" in row
        for row in enabled["trajectory"]
    )
    assert all(
        "theory_finite_head_entry_margin_lower_bound" in row
        for row in enabled["trajectory"]
    )


def test_rule_margin_observer_is_opt_in_and_update_invariant() -> None:
    disabled = run_experiment(tiny_config(steps=2))
    enabled = run_experiment(tiny_config(steps=2, rule_margin_observer=True))
    torch.testing.assert_close(
        disabled["model"].feature_weight,
        enabled["model"].feature_weight,
        rtol=0.0,
        atol=0.0,
    )
    torch.testing.assert_close(
        disabled["model"].head_weight,
        enabled["model"].head_weight,
        rtol=0.0,
        atol=0.0,
    )
    assert not any(key.startswith("rule_") for key in disabled["trajectory"][0])
    assert all(
        "rule_heldout_margin_discrepancy_max" in row
        for row in enabled["trajectory"]
    )


def test_machine_readable_outputs_refuse_overwrite(tmp_path: Path) -> None:
    destination = tmp_path / "run"
    run_experiment(tiny_config(steps=1), destination)
    assert (destination / "run.json").exists()
    assert (destination / "trajectory.csv").exists()
    assert (destination / "trajectory.npz").exists()
    with pytest.raises(FileExistsError):
        run_experiment(tiny_config(steps=1), destination)


def test_adaptive_certificate_fields_round_trip_machine_outputs(
    tmp_path: Path,
) -> None:
    destination = tmp_path / "adaptive"
    run_experiment(
        tiny_config(
            steps=2,
            optimizer="polar_feature_gd_head",
            feature_geometry="row_sphere",
            adaptive_balance_start_step=0,
            adaptive_balance_denominator_floor=0.0,
        ),
        destination,
    )
    payload = json.loads((destination / "run.json").read_text(encoding="utf-8"))
    archive = np.load(destination / "trajectory.npz")
    fields = (
        "feature_tangent_gradient_retained_nuclear_norm",
        "feature_data_direction_pairing",
        "feature_retraction_gradient_inner_product",
        "full_exact_gradient_increment_inner_product",
        "training_ce_realized_one_step_change",
        "training_ce_first_order_taylor_remainder",
        "adaptive_phase_cumulative_abs_training_ce_realized_change",
        "adaptive_phase_cumulative_abs_full_exact_gradient_increment_inner_product",
        "adaptive_phase_cumulative_abs_training_ce_taylor_remainder",
        "adaptive_phase_active_update_count",
        "adaptive_phase_expected_update_count",
        "adaptive_phase_formula_used_update_count",
        "adaptive_phase_fallback_update_count",
        "adaptive_phase_clipped_update_count",
        "adaptive_phase_minimum_denominator",
        "adaptive_phase_maximum_absolute_balance_residual",
        "adaptive_phase_nonfinite_mechanism_diagnostic_count",
        "adaptive_balance_all_step_index_mismatch_count",
        "adaptive_balance_all_step_phase_mismatch_count",
        "adaptive_head_decay_source_state_step",
        "adaptive_head_decay_update_number",
    )
    for field in fields:
        assert field in payload["trajectory"][0]
        assert field in archive.files
        expected = np.asarray(
            [
                np.nan if row[field] is None else float(row[field])
                for row in payload["trajectory"]
            ]
        )
        np.testing.assert_allclose(
            archive[field], expected, rtol=0.0, atol=0.0, equal_nan=True
        )


def _adaptive_step_fixture(
    **updates: object,
) -> tuple[
    RunConfig,
    QuadraticOneHiddenLayer,
    EndToEndMatrixOptimizer,
    object,
]:
    config = tiny_config(
        steps=0,
        optimizer="polar_feature_gd_head",
        activation="square",
        feature_geometry="row_sphere",
        lr_feature=0.013,
        lr_head=0.17,
        weight_decay_head=0.025,
        adaptive_balance_start_step=0,
        adaptive_balance_denominator_floor=0.0,
        **updates,
    )
    config.validate()
    device = torch.device("cpu")
    torch.manual_seed(config.seed)
    model = QuadraticOneHiddenLayer(
        config.p,
        config.width,
        config.feature_init_scale,
        config.head_init_scale,
        device=device,
        dtype=torch.float64,
        activation=config.activation,
    )
    optimizer = EndToEndMatrixOptimizer(model, config)
    data = make_data(config.p, config.train_fraction, config.split_seed, device)
    optimizer.zero_grad()
    F.cross_entropy(
        model(data.pairs[data.train_idx]), data.labels[data.train_idx]
    ).backward()
    return config, model, optimizer, data


def test_adaptive_head_decay_uses_exact_balance_formula() -> None:
    config, model, optimizer, _ = _adaptive_step_fixture()

    feature_gradient = model.feature_weight.grad @ model.input_basis
    feature_point = model.feature_weight.detach() @ model.input_basis
    feature_tangent = rowwise_tangent_projection(feature_gradient, feature_point)
    feature_nuclear = torch.linalg.matrix_norm(feature_tangent, ord="nuc").item()
    feature_polar, feature_polar_diagnostics = support_polar_factor(
        feature_tangent, strict_rank=False
    )
    feature_pairing = torch.sum(feature_tangent * feature_polar).item()
    head_gradient = model.output_basis.T @ model.head_weight.grad
    head_point = model.output_basis.T @ model.head_weight.detach()
    denominator = -torch.sum(head_gradient * head_point).item()
    assert denominator > 0.0
    expected = (
        torch.sum(head_gradient.square()).item()
        + (config.lr_feature / config.lr_head) * feature_pairing
    ) / denominator

    diagnostics = optimizer.step()

    assert diagnostics["adaptive_head_decay_active"] == 1
    assert diagnostics["adaptive_head_decay_formula_used"] == 1
    assert diagnostics["adaptive_head_decay_was_clipped"] == 0
    assert diagnostics["adaptive_head_decay_denominator"] == pytest.approx(
        denominator, rel=2.0e-14, abs=2.0e-14
    )
    assert diagnostics["adaptive_head_decay_unclipped_lambda"] == pytest.approx(
        expected, rel=2.0e-14, abs=2.0e-14
    )
    assert diagnostics["adaptive_head_decay_lambda"] == pytest.approx(
        expected, rel=2.0e-14, abs=2.0e-14
    )
    assert diagnostics["feature_tangent_gradient_nuclear_norm"] == pytest.approx(
        feature_nuclear, rel=2.0e-14, abs=2.0e-14
    )
    assert diagnostics[
        "feature_tangent_gradient_retained_nuclear_norm"
    ] == pytest.approx(
        feature_polar_diagnostics["retained_nuclear_norm"],
        rel=2.0e-14,
        abs=2.0e-14,
    )
    assert diagnostics["feature_data_direction_pairing"] == pytest.approx(
        feature_pairing, rel=2.0e-14, abs=2.0e-14
    )


def test_adaptive_balance_uses_retained_support_polar_pairing_at_cutoff() -> None:
    config = RunConfig(
        seed=3,
        split_seed=2,
        p=3,
        train_fraction=0.6,
        width=128,
        steps=0,
        optimizer="polar_feature_gd_head",
        lr_feature=0.01,
        lr_head=1.0e-6,
        weight_decay_feature=0.0,
        weight_decay_head=0.0,
        feature_geometry="row_sphere",
        adaptive_balance_start_step=0,
        adaptive_balance_denominator_floor=0.0,
    )
    config.validate()
    torch.manual_seed(config.seed)
    model = QuadraticOneHiddenLayer(
        config.p,
        config.width,
        config.feature_init_scale,
        config.head_init_scale,
        device=torch.device("cpu"),
        dtype=torch.float64,
        activation="square",
    )
    optimizer = EndToEndMatrixOptimizer(model, config)
    feature_point = model.feature_weight.detach() @ model.input_basis

    candidate = torch.arange(
        1.0, feature_point.shape[1] + 1.0, dtype=torch.float64
    )
    first = candidate - (
        torch.dot(candidate, feature_point[0])
        / torch.dot(feature_point[0], feature_point[0])
    ) * feature_point[0]
    first /= torch.linalg.vector_norm(first)
    constraints = torch.stack((feature_point[1], first))
    _, _, vh = torch.linalg.svd(constraints, full_matrices=True)
    second = vh[-1]
    tangent_gradient = torch.zeros_like(feature_point)
    tangent_gradient[0] = first
    tangent_gradient[1] = 1.0e-14 * second
    model.feature_weight.grad = tangent_gradient @ model.input_basis.T

    head_point = model.output_basis.T @ model.head_weight.detach()
    head_gradient = -head_point
    model.head_weight.grad = model.output_basis @ head_gradient
    denominator = -torch.sum(head_gradient * head_point).item()
    head_squared = torch.sum(head_gradient.square()).item()

    polar, polar_diagnostics = support_polar_factor(tangent_gradient)
    retained_pairing = torch.sum(tangent_gradient * polar).item()
    expected = (
        head_squared
        + (config.lr_feature / config.lr_head) * retained_pairing
    ) / denominator
    incorrectly_using_full_nuclear = (
        head_squared
        + (config.lr_feature / config.lr_head)
        * polar_diagnostics["nuclear_norm"]
    ) / denominator

    diagnostics = optimizer.step()

    assert polar_diagnostics["rank"] == 1
    assert polar_diagnostics["discarded_nuclear_norm"] == pytest.approx(1.0e-14)
    assert abs(incorrectly_using_full_nuclear - expected) > 1.0e-11
    assert diagnostics["feature_data_direction_pairing"] == pytest.approx(
        retained_pairing, rel=0.0, abs=2.0e-14
    )
    assert diagnostics["adaptive_head_decay_lambda"] == pytest.approx(
        expected, rel=2.0e-14, abs=2.0e-12
    )
    assert diagnostics["total_signed_first_order_ce_change"] == pytest.approx(
        0.0, abs=2.0e-14
    )


def test_unclipped_adaptive_decay_cancels_first_order_ce_change() -> None:
    config, model, optimizer, _ = _adaptive_step_fixture()
    feature_before = model.feature_weight.detach().clone()
    head_before = model.head_weight.detach().clone()
    feature_gradient = model.feature_weight.grad.detach().clone()
    head_gradient = model.head_weight.grad.detach().clone()
    diagnostics = optimizer.step()

    feature_exact = torch.sum(
        feature_gradient * (model.feature_weight.detach() - feature_before)
    ).item()
    head_exact = torch.sum(
        head_gradient * (model.head_weight.detach() - head_before)
    ).item()
    full_exact = feature_exact + head_exact

    assert diagnostics["adaptive_head_decay_formula_used"] == 1
    assert diagnostics["adaptive_head_decay_balance_residual"] == pytest.approx(
        0.0, abs=2.0e-14
    )
    assert diagnostics["total_signed_first_order_ce_change"] == pytest.approx(
        -config.lr_head
        * diagnostics["adaptive_head_decay_balance_residual"],
        rel=2.0e-13,
        abs=2.0e-14,
    )
    assert diagnostics[
        "feature_exact_gradient_increment_inner_product"
    ] == pytest.approx(feature_exact, rel=2.0e-13, abs=2.0e-14)
    assert diagnostics[
        "head_exact_gradient_increment_inner_product"
    ] == pytest.approx(head_exact, rel=2.0e-13, abs=2.0e-14)
    assert diagnostics[
        "full_exact_gradient_increment_inner_product"
    ] == pytest.approx(full_exact, rel=2.0e-13, abs=2.0e-14)
    assert diagnostics[
        "full_exact_gradient_increment_inner_product"
    ] == pytest.approx(
        diagnostics["total_signed_first_order_ce_change"]
        + diagnostics["feature_retraction_gradient_inner_product"],
        rel=2.0e-10,
        abs=2.0e-14,
    )
    assert diagnostics["adaptive_head_decay_source_state_step"] == 0
    assert diagnostics["adaptive_head_decay_update_number"] == 1


def test_adaptive_run_logs_realized_ce_taylor_identity_and_running_budgets() -> None:
    result = run_experiment(
        tiny_config(
            steps=4,
            log_every=1,
            optimizer="polar_feature_gd_head",
            feature_geometry="row_sphere",
            lr_feature=0.002,
            lr_head=0.05,
            weight_decay_head=0.025,
            adaptive_balance_start_step=3,
            adaptive_balance_denominator_floor=0.0,
        )
    )
    rows = result["trajectory"]
    cumulative_realized = 0.0
    cumulative_first_order = 0.0
    cumulative_remainder = 0.0
    active_count = 0
    formula_count = 0
    fallback_count = 0
    clipped_count = 0
    minimum_denominator = math.inf
    maximum_absolute_residual = 0.0

    assert rows[0][
        "adaptive_phase_cumulative_abs_training_ce_realized_change"
    ] == 0.0
    for field in (
        "adaptive_phase_active_update_count",
        "adaptive_phase_expected_update_count",
        "adaptive_phase_formula_used_update_count",
        "adaptive_phase_fallback_update_count",
        "adaptive_phase_clipped_update_count",
        "adaptive_phase_nonfinite_mechanism_diagnostic_count",
        "adaptive_balance_all_step_index_mismatch_count",
        "adaptive_balance_all_step_phase_mismatch_count",
    ):
        assert rows[0][field] == 0
    assert math.isnan(rows[0]["adaptive_phase_minimum_denominator"])
    assert math.isnan(
        rows[0]["adaptive_phase_maximum_absolute_balance_residual"]
    )
    assert [row["adaptive_head_decay_active"] for row in rows[1:]] == [
        0,
        0,
        0,
        1,
    ]
    for update_number, row in enumerate(rows[1:], start=1):
        realized = row["train_ce"] - rows[update_number - 1]["train_ce"]
        first_order = row["full_exact_gradient_increment_inner_product"]
        remainder = realized - first_order
        assert row["adaptive_head_decay_source_state_step"] == update_number - 1
        assert row["adaptive_head_decay_update_number"] == update_number
        assert row["training_ce_realized_one_step_change"] == pytest.approx(
            realized, rel=2.0e-13, abs=2.0e-14
        )
        assert row["training_ce_first_order_taylor_remainder"] == pytest.approx(
            remainder, rel=2.0e-13, abs=2.0e-14
        )
        assert realized == pytest.approx(
            first_order + row["training_ce_first_order_taylor_remainder"],
            rel=2.0e-13,
            abs=2.0e-14,
        )
        if row["adaptive_head_decay_active"]:
            active_count += 1
            formula_count += row["adaptive_head_decay_formula_used"]
            fallback_count += 1 - row["adaptive_head_decay_formula_used"]
            clipped_count += row["adaptive_head_decay_was_clipped"]
            minimum_denominator = min(
                minimum_denominator,
                row["adaptive_head_decay_denominator"],
            )
            maximum_absolute_residual = max(
                maximum_absolute_residual,
                abs(row["adaptive_head_decay_balance_residual"]),
            )
            cumulative_realized += abs(realized)
            cumulative_first_order += abs(first_order)
            cumulative_remainder += abs(remainder)
        assert row[
            "adaptive_phase_cumulative_abs_training_ce_realized_change"
        ] == pytest.approx(cumulative_realized, rel=2.0e-13, abs=2.0e-14)
        assert row[
            "adaptive_phase_cumulative_abs_full_exact_gradient_increment_inner_product"
        ] == pytest.approx(cumulative_first_order, rel=2.0e-13, abs=2.0e-14)
        assert row[
            "adaptive_phase_cumulative_abs_training_ce_taylor_remainder"
        ] == pytest.approx(cumulative_remainder, rel=2.0e-13, abs=2.0e-14)
        assert row["adaptive_phase_active_update_count"] == active_count
        assert row["adaptive_phase_expected_update_count"] == max(
            0, update_number - 3
        )
        assert row["adaptive_phase_formula_used_update_count"] == formula_count
        assert row["adaptive_phase_fallback_update_count"] == fallback_count
        assert row["adaptive_phase_clipped_update_count"] == clipped_count
        assert row[
            "adaptive_phase_nonfinite_mechanism_diagnostic_count"
        ] == 0
        assert row["adaptive_balance_all_step_index_mismatch_count"] == 0
        assert row["adaptive_balance_all_step_phase_mismatch_count"] == 0
        if active_count:
            assert row["adaptive_phase_minimum_denominator"] == pytest.approx(
                minimum_denominator, rel=2.0e-13, abs=2.0e-14
            )
            assert row[
                "adaptive_phase_maximum_absolute_balance_residual"
            ] == pytest.approx(
                maximum_absolute_residual, rel=2.0e-13, abs=2.0e-14
            )
        else:
            assert math.isnan(row["adaptive_phase_minimum_denominator"])
            assert math.isnan(
                row["adaptive_phase_maximum_absolute_balance_residual"]
            )


def test_adaptive_all_step_mechanism_counters_cover_unlogged_updates() -> None:
    result = run_experiment(
        tiny_config(
            steps=5,
            log_every=5,
            optimizer="polar_feature_gd_head",
            feature_geometry="row_sphere",
            lr_feature=0.002,
            lr_head=0.05,
            adaptive_balance_start_step=2,
            adaptive_balance_denominator_floor=0.0,
        )
    )
    assert [row["step"] for row in result["trajectory"]] == [0, 5]
    final = result["trajectory"][-1]
    assert final["adaptive_phase_expected_update_count"] == 3
    assert final["adaptive_phase_active_update_count"] == 3
    assert final["adaptive_phase_formula_used_update_count"] == 3
    assert final["adaptive_phase_fallback_update_count"] == 0
    assert final["adaptive_phase_clipped_update_count"] == 0
    assert final["adaptive_phase_minimum_denominator"] > 0.0
    assert final[
        "adaptive_phase_maximum_absolute_balance_residual"
    ] == pytest.approx(0.0, abs=2.0e-14)
    assert final["adaptive_phase_nonfinite_mechanism_diagnostic_count"] == 0
    assert final["adaptive_balance_all_step_index_mismatch_count"] == 0
    assert final["adaptive_balance_all_step_phase_mismatch_count"] == 0


def test_adaptive_decay_is_training_only_and_ignores_heldout_labels() -> None:
    config = tiny_config(
        steps=0,
        optimizer="polar_feature_gd_head",
        feature_geometry="row_sphere",
        lr_feature=0.013,
        lr_head=0.17,
        adaptive_balance_start_step=0,
        adaptive_balance_denominator_floor=0.0,
    )
    device = torch.device("cpu")
    data = make_data(config.p, config.train_fraction, config.split_seed, device)
    alternate_labels = data.labels.clone()
    alternate_labels[data.test_idx] = torch.remainder(
        alternate_labels[data.test_idx] + 1, config.p
    )
    assert torch.equal(
        alternate_labels[data.train_idx], data.labels[data.train_idx]
    )
    assert not torch.equal(
        alternate_labels[data.test_idx], data.labels[data.test_idx]
    )

    models = []
    optimizers = []
    for _ in range(2):
        torch.manual_seed(config.seed)
        model = QuadraticOneHiddenLayer(
            config.p,
            config.width,
            config.feature_init_scale,
            config.head_init_scale,
            device=device,
            dtype=torch.float64,
            activation=config.activation,
        )
        models.append(model)
        optimizers.append(EndToEndMatrixOptimizer(model, config))

    diagnostics = []
    for model, optimizer, labels in zip(
        models, optimizers, (data.labels, alternate_labels), strict=True
    ):
        optimizer.zero_grad()
        F.cross_entropy(
            model(data.pairs[data.train_idx]), labels[data.train_idx]
        ).backward()
        diagnostics.append(optimizer.step())

    assert diagnostics[0]["adaptive_head_decay_formula_used"] == 1
    assert diagnostics[0]["adaptive_head_decay_lambda"] == diagnostics[1][
        "adaptive_head_decay_lambda"
    ]
    assert torch.equal(models[0].feature_weight, models[1].feature_weight)
    assert torch.equal(models[0].head_weight, models[1].head_weight)


def test_adaptive_decay_warmup_fallback_clipping_and_default_are_explicit() -> None:
    default = tiny_config(
        optimizer="polar_feature_gd_head",
        feature_geometry="row_sphere",
        steps=2,
        weight_decay_head=0.025,
    )
    disabled = replace(
        default,
        adaptive_balance_start_step=-1,
        adaptive_balance_decay_min=10.0,
        adaptive_balance_decay_max=11.0,
    )
    default_result = run_experiment(default)
    disabled_result = run_experiment(disabled)
    assert default.adaptive_balance_start_step == -1
    assert torch.equal(
        default_result["model"].feature_weight,
        disabled_result["model"].feature_weight,
    )
    assert torch.equal(
        default_result["model"].head_weight,
        disabled_result["model"].head_weight,
    )
    assert all(
        row["adaptive_head_decay_active"] == 0
        for row in default_result["trajectory"]
    )
    assert default_result["trajectory"][-1][
        "adaptive_head_decay_lambda"
    ] == pytest.approx(default.weight_decay_head)

    warmup = run_experiment(
        replace(
            default,
            steps=3,
            adaptive_balance_start_step=2,
            adaptive_balance_denominator_floor=1.0e100,
            adaptive_balance_decay_min=0.0,
            adaptive_balance_decay_max=0.01,
        )
    )
    first, second, third = warmup["trajectory"][1:]
    assert first["adaptive_head_decay_active"] == 0
    assert first["adaptive_head_decay_source_state_step"] == 0
    assert first["adaptive_head_decay_update_number"] == 1
    assert first["adaptive_head_decay_lambda"] == pytest.approx(0.025)
    assert second["adaptive_head_decay_active"] == 0
    assert second["adaptive_head_decay_source_state_step"] == 1
    assert second["adaptive_head_decay_update_number"] == 2
    assert second["adaptive_head_decay_lambda"] == pytest.approx(0.025)
    assert third["adaptive_head_decay_active"] == 1
    assert third["adaptive_head_decay_source_state_step"] == 2
    assert third["adaptive_head_decay_update_number"] == 3
    assert third["adaptive_head_decay_formula_used"] == 0
    assert third["adaptive_head_decay_was_clipped"] == 1
    assert third["adaptive_head_decay_unclipped_lambda"] == pytest.approx(0.025)
    assert third["adaptive_head_decay_lambda"] == pytest.approx(0.01)
    assert third["adaptive_phase_active_update_count"] == 1
    assert third["adaptive_phase_expected_update_count"] == 1
    assert third["adaptive_phase_formula_used_update_count"] == 0
    assert third["adaptive_phase_fallback_update_count"] == 1
    assert third["adaptive_phase_clipped_update_count"] == 1
    assert third["adaptive_phase_minimum_denominator"] == pytest.approx(
        third["adaptive_head_decay_denominator"]
    )
    assert third[
        "adaptive_phase_maximum_absolute_balance_residual"
    ] == pytest.approx(abs(third["adaptive_head_decay_balance_residual"]))
    assert third["adaptive_phase_nonfinite_mechanism_diagnostic_count"] == 0
    assert third["adaptive_balance_all_step_index_mismatch_count"] == 0
    assert third["adaptive_balance_all_step_phase_mismatch_count"] == 0

    immediate_zero = run_experiment(
        replace(default, steps=1, adaptive_balance_start_step=0)
    )
    one_state_warmup = run_experiment(
        replace(default, steps=2, adaptive_balance_start_step=1)
    )
    assert immediate_zero["trajectory"][1]["adaptive_head_decay_active"] == 1
    assert one_state_warmup["trajectory"][1]["adaptive_head_decay_active"] == 0
    assert one_state_warmup["trajectory"][2]["adaptive_head_decay_active"] == 1
    assert one_state_warmup["trajectory"][2][
        "adaptive_head_decay_source_state_step"
    ] == 1


@pytest.mark.parametrize(
    "updates,match",
    [
        ({"adaptive_balance_start_step": -2}, "start_step"),
        ({"adaptive_balance_start_step": 0, "optimizer": "gd"}, "optimizer"),
        (
            {
                "adaptive_balance_start_step": 0,
                "optimizer": "polar_feature_gd_head",
                "feature_geometry": "euclidean",
            },
            "row_sphere",
        ),
        (
            {
                "adaptive_balance_start_step": 0,
                "optimizer": "polar_feature_gd_head",
                "feature_geometry": "row_sphere",
                "momentum": 0.5,
            },
            "zero momentum",
        ),
        ({"adaptive_balance_denominator_floor": -1.0}, "denominator_floor"),
        ({"adaptive_balance_decay_min": -0.1}, "decay_min"),
        (
            {
                "adaptive_balance_decay_min": 0.2,
                "adaptive_balance_decay_max": 0.1,
            },
            "cannot exceed",
        ),
    ],
)
def test_adaptive_head_decay_validation(
    updates: dict[str, object], match: str
) -> None:
    config = tiny_config(**updates)
    with pytest.raises(ValueError, match=match):
        config.validate()


def test_adaptive_head_decay_cli_round_trips(tmp_path: Path) -> None:
    args = build_parser().parse_args(
        [
            "single",
            "--optimizer",
            "polar_feature_gd_head",
            "--feature-geometry",
            "row_sphere",
            "--adaptive-balance-start-step",
            "5000",
            "--adaptive-balance-denominator-floor",
            "1e-10",
            "--adaptive-balance-decay-min",
            "0.001",
            "--adaptive-balance-decay-max",
            "0.2",
            "--output-dir",
            str(tmp_path / "adaptive"),
        ]
    )
    config = RunConfig(
        seed=args.seed,
        width=args.width,
        optimizer=args.optimizer,
        lr_feature=args.lr_feature,
        **shared_kwargs(args),
    )
    config.validate()
    assert config.adaptive_balance_start_step == 5000
    assert config.adaptive_balance_denominator_floor == pytest.approx(1.0e-10)
    assert config.adaptive_balance_decay_min == pytest.approx(0.001)
    assert config.adaptive_balance_decay_max == pytest.approx(0.2)
    assert "_adaptive5000_" in run_id(config)


def test_adaptive_phase_cannot_masquerade_as_fixed_decay_certificate() -> None:
    result = run_experiment(
        tiny_config(
            steps=3,
            log_every=1,
            optimizer="polar_feature_gd_head",
            feature_geometry="row_sphere",
            adaptive_balance_start_step=2,
            adaptive_balance_denominator_floor=0.0,
            theory_gradient_observer=True,
        )
    )
    initial, warmup_one, warmup_two, adaptive = result["trajectory"]
    for row in (initial, warmup_one, warmup_two):
        assert row["head_fixed_decay_contraction_applicable"] == 1
        assert row[
            "theory_all_step_head_fixed_decay_contraction_applicable"
        ] == 1
        assert np.isfinite(row["head_contraction_factor"])
        assert np.isfinite(row["theory_all_step_head_trace_smoothness_max"])

    assert adaptive["adaptive_head_decay_active"] == 1
    assert adaptive["head_fixed_decay_contraction_applicable"] == 0
    assert adaptive[
        "theory_all_step_head_fixed_decay_contraction_applicable"
    ] == 0
    assert math.isnan(adaptive["head_empirical_gram_smoothness"])
    assert math.isnan(adaptive["head_contraction_factor"])
    assert math.isnan(adaptive["theory_all_step_head_trace_smoothness_max"])
    assert math.isnan(
        adaptive[
            "theory_all_step_head_trace_contraction_stepsize_lhs_max"
        ]
    )
    assert np.isfinite(
        adaptive["theory_train_vs_full_feature_raw_gradient_cosine"]
    )
    assert "theory_all_step_postfit_train_margin_min_running" in adaptive
