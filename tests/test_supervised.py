from collections.abc import Mapping
from dataclasses import replace
from typing import cast

import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pandas as pd
import pytest
from jax import tree
from tjax import RngStream

import cem.demos.supervised.problem as supervised_problem
import cem.demos.supervised.solution as supervised_solution
from cem.commands.demos import DemoEnum, demo_registry
from cem.demos.supervised.demo import (
    supervised_bike_sharing_demand_demo,
    supervised_cpu_activity_demo,
    supervised_elevators_demo,
    supervised_iris_demo,
)
from cem.demos.supervised.problem import SupervisedProblem
from cem.demos.supervised.solution import (
    DatasetKind,
    PerceptronSupervisedModel,
    PhasorProjectionKind,
    PhasorSupervisedModel,
    SupervisedSolver,
)
from cem.phasor import (
    FrequencyGatedProjection,
    FrequencyMobiusProjection,
    FrequencyPhaseActivation,
    MobiusSummation,
    PhasorTargetConfiguration,
    RecurrentPhaseFocusing,
)
from cem.phasor.frequency import frequency_base_for_domain_width
from cem.phasor.telemetry import SpectralLossTelemetry
from cem.structure.graph import LearnableParameter
from cem.structure.plotter import Demo
from cem.structure.solution import (
    ExecutionPacket,
    InferenceResults,
    LossTelemetry,
    SolutionState,
    Telemetries,
    TrainingResults,
)

_EXPECTED_HF_TEST_FEATURES = 2
_EXPECTED_HF_TEST_ROWS = 5
_PARAMETER_COUNT_RELATIVE_TOLERANCE = 0.02


def _small_supervised_problem() -> SupervisedProblem:
    rng = np.random.default_rng(0)
    x = rng.normal(size=(32, 4))
    y = 0.5 * x[:, 0] - 0.25 * x[:, 1] + 0.1 * rng.normal(size=32)
    df = pd.DataFrame({f"x_{i}": x[:, i] for i in range(x.shape[1])})
    df["target"] = y
    return supervised_problem.problem_from_numeric_dataframe(df, max_rows=16, seed=0)


def _small_multi_target_problem() -> SupervisedProblem:
    rng = np.random.default_rng(1)
    x = rng.normal(size=(32, 4))
    df = pd.DataFrame({f"x_{i}": x[:, i] for i in range(x.shape[1])})
    df["target_0"] = 0.5 * x[:, 0] - 0.25 * x[:, 1]
    df["target_1"] = -0.3 * x[:, 2] + 0.2 * x[:, 3]
    return supervised_problem.problem_from_numeric_dataframe(
        df,
        max_rows=None,
        seed=0,
        n_targets=2,
    )


def _problem_with_targets(targets: np.ndarray) -> SupervisedProblem:
    features = np.linspace(-1.0, 1.0, targets.shape[0])
    df = pd.DataFrame({"feature": features, "target": targets})
    return supervised_problem.problem_from_numeric_dataframe(df, max_rows=None, seed=0)


def _problem_with_feature_count(n_features: int) -> SupervisedProblem:
    rng = np.random.default_rng(n_features)
    x = rng.normal(size=(32, n_features))
    df = pd.DataFrame(
        {
            **{f"x_{i}": x[:, i] for i in range(n_features)},
            "target": rng.normal(size=x.shape[0]),
        }
    )
    return supervised_problem.problem_from_numeric_dataframe(df, max_rows=None, seed=0)


def _real_parameter_count(model: object) -> int:
    parameters = [
        leaf
        for leaf in tree.leaves(model, is_leaf=lambda x: isinstance(x, LearnableParameter))
        if isinstance(leaf, LearnableParameter)
    ]
    return sum(
        parameter.value.size
        * (2 if jnp.issubdtype(parameter.value.dtype, jnp.complexfloating) else 1)
        for parameter in parameters
    )


def test_phasor_supervised_multi_target_infer_splits_target_fields(
    streams: Mapping[str, RngStream],
) -> None:
    problem = _small_multi_target_problem()
    model = PhasorSupervisedModel.create(
        problem,
        n_frequencies=10,
        hidden_size=8,
        streams=streams,
    )
    observation = problem.create_data_source().initial_problem_state(jr.key(0))

    result = model.infer(observation, None, streams=streams, inference=False)
    config = result.configurations["target"]
    assert isinstance(config, PhasorTargetConfiguration)

    assert tuple(config.loss) == ("y_0", "y_1")
    assert config.score.shape == (problem.n_targets * 10,)
    assert jnp.isfinite(result.loss)


def test_phasor_supervised_model_enables_mobius_projection_dropout(
    streams: Mapping[str, RngStream],
) -> None:
    model = PhasorSupervisedModel.create(
        _small_multi_target_problem(),
        n_frequencies=4,
        hidden_size=8,
        streams=streams,
    )

    assert jnp.allclose(model.link.dropout_rate.value, 0.1)


def test_phasor_supervised_model_supports_dense_mobius_candidates(
    streams: Mapping[str, RngStream],
) -> None:
    model = PhasorSupervisedModel.create(
        _small_multi_target_problem(),
        n_frequencies=4,
        hidden_size=8,
        projection_kind=PhasorProjectionKind.dense,
        streams=streams,
    )

    assert isinstance(model.link, FrequencyMobiusProjection)
    assert isinstance(model.link.value, MobiusSummation)


def test_phasor_supervised_model_supports_dense_signed_weights(
    streams: Mapping[str, RngStream],
) -> None:
    model = PhasorSupervisedModel.create(
        _small_multi_target_problem(),
        n_frequencies=4,
        hidden_size=8,
        projection_kind=PhasorProjectionKind.dense,
        streams=streams,
    )

    assert isinstance(model.link, FrequencyMobiusProjection)
    assert isinstance(model.link.value, MobiusSummation)
    expected_parameter_count = 4 + 2 * 8 * 4 + 2 * 8 * 2
    assert _real_parameter_count(model) == expected_parameter_count


def test_phasor_supervised_model_supports_phase_activation(
    streams: Mapping[str, RngStream],
) -> None:
    model = PhasorSupervisedModel.create(
        _small_multi_target_problem(),
        n_frequencies=4,
        hidden_size=8,
        projection_kind=PhasorProjectionKind.phase_activated,
        streams=streams,
    )

    assert isinstance(model.link, FrequencyMobiusProjection)
    assert isinstance(model.link.activation, FrequencyPhaseActivation)
    expected_parameter_count = 4 + 2 * 8 * 4 + 2 * 8 + 2 * 8 * 2
    assert _real_parameter_count(model) == expected_parameter_count


def test_phasor_supervised_model_supports_gated_projection(
    streams: Mapping[str, RngStream],
) -> None:
    model = PhasorSupervisedModel.create(
        _small_multi_target_problem(),
        n_frequencies=4,
        hidden_size=8,
        mobius_rank=2,
        projection_kind=PhasorProjectionKind.gated,
        streams=streams,
    )

    assert isinstance(model.link, FrequencyGatedProjection)
    expected_parameter_count = 4 + 4 * 2 * (8 + 4) + 8 + 2 * 8 * 2
    assert _real_parameter_count(model) == expected_parameter_count


def test_phasor_supervised_model_supports_recurrent_phase_focusing(
    streams: Mapping[str, RngStream],
) -> None:
    recurrent_steps = 3
    problem = _small_multi_target_problem()
    model = PhasorSupervisedModel.create(
        problem,
        n_frequencies=4,
        hidden_size=8,
        recurrent_steps=recurrent_steps,
        projection_kind=PhasorProjectionKind.recurrent_phase_focusing,
        streams=streams,
    )
    observation = problem.create_data_source().initial_problem_state(jr.key(0))

    assert isinstance(model.link, RecurrentPhaseFocusing)
    assert model.link.iterations == recurrent_steps
    assert model.target.frequencies.value.shape == (1,)
    result = model.infer(observation, None, streams=streams, inference=True)
    config = result.configurations["target"]
    assert isinstance(config, PhasorTargetConfiguration)
    assert config.score.shape == (problem.n_targets,)
    assert jnp.isfinite(result.loss)


@pytest.mark.parametrize(
    (
        "n_features",
        "n_frequencies",
        "phasor_hidden",
        "mlp_hidden",
    ),
    [
        (4, 12, 128, 220),
        (6, 6, 73, 98),
        (16, 8, 139, 85),
        (21, 4, 27, 20),
    ],
)
def test_supervised_models_have_similar_real_parameter_counts(
    n_features: int,
    n_frequencies: int,
    phasor_hidden: int,
    mlp_hidden: int,
    streams: Mapping[str, RngStream],
) -> None:
    problem = _problem_with_feature_count(n_features)
    perceptron = PerceptronSupervisedModel.create(problem, mlp_hidden, streams=streams)
    phasor = PhasorSupervisedModel.create(
        problem,
        n_frequencies,
        phasor_hidden,
        4,
        streams=streams,
    )
    perceptron_count = _real_parameter_count(perceptron)
    phasor_count = _real_parameter_count(phasor)
    assert (
        abs(perceptron_count - phasor_count) / perceptron_count
        < _PARAMETER_COUNT_RELATIVE_TOLERANCE
    )


def test_frequency_base_for_domain_width_keeps_unit_base_for_small_domains() -> None:
    problem = _problem_with_targets(np.array([-1.0, 0.0, 1.0]))
    domain_width = jnp.max(problem.y_flat) - jnp.min(problem.y_flat)
    assert jnp.allclose(frequency_base_for_domain_width(domain_width), 1.0)


def test_frequency_base_for_domain_width_scales_large_domains() -> None:
    problem = _problem_with_targets(np.concatenate([np.zeros(99), np.array([10.0])]))
    domain_width = jnp.max(problem.y_flat) - jnp.min(problem.y_flat)
    expected = 0.9 * 2.0 * jnp.pi / domain_width
    assert jnp.allclose(frequency_base_for_domain_width(domain_width), expected)


def test_frequency_base_for_domain_width_handles_constant_targets() -> None:
    problem = _problem_with_targets(np.array([3.0, 3.0, 3.0]))
    domain_width = jnp.max(problem.y_flat) - jnp.min(problem.y_flat)
    assert jnp.allclose(frequency_base_for_domain_width(domain_width), 1.0)


def test_phasor_supervised_model_scales_target_frequencies(
    streams: Mapping[str, RngStream],
) -> None:
    problem = _problem_with_targets(np.concatenate([np.zeros(99), np.array([10.0])]))
    model = PhasorSupervisedModel.create(
        problem,
        n_frequencies=4,
        hidden_size=8,
        streams=streams,
    )
    assert model.target.frequencies.value[0] < 1.0


def test_hf_tabular_regression_dataframe_loader_is_deterministic() -> None:
    df = pd.DataFrame(
        {
            "feature_a": np.arange(10, dtype=np.float64),
            "feature_b": np.arange(10, dtype=np.float64) ** 2,
            "ignored": ["x"] * 10,
            "target": np.linspace(0.0, 1.0, 10),
        }
    )
    first = supervised_problem.problem_from_numeric_dataframe(
        df,
        max_rows=_EXPECTED_HF_TEST_ROWS,
        seed=7,
    )
    second = supervised_problem.problem_from_numeric_dataframe(
        df,
        max_rows=_EXPECTED_HF_TEST_ROWS,
        seed=7,
    )

    assert first.n_features == _EXPECTED_HF_TEST_FEATURES
    assert first.n_targets == 1
    assert first.x_flat.shape == (_EXPECTED_HF_TEST_ROWS, _EXPECTED_HF_TEST_FEATURES)
    assert first.y_flat.shape == (_EXPECTED_HF_TEST_ROWS, 1)
    assert jnp.all(jnp.isfinite(first.x_flat))
    assert jnp.all(jnp.isfinite(first.y_flat))
    assert jnp.allclose(first.x_flat, second.x_flat)
    assert jnp.allclose(first.y_flat, second.y_flat)


@pytest.mark.parametrize(
    ("demo", "enum_value"),
    [
        (supervised_bike_sharing_demand_demo, DemoEnum.supervised_bike_sharing_demand),
        (supervised_elevators_demo, DemoEnum.supervised_elevators),
        (supervised_cpu_activity_demo, DemoEnum.supervised_cpu_activity),
    ],
)
def test_hf_supervised_demo_registry_and_variants(demo: Demo, enum_value: DemoEnum) -> None:
    assert demo_registry[enum_value] is demo
    assert [variant.label for variant in demo.variants] == ["perceptron", "phasor"]


@pytest.mark.parametrize(
    "dataset_kind",
    [
        DatasetKind.bike_sharing_demand,
        DatasetKind.elevators,
        DatasetKind.cpu_activity,
    ],
)
def test_hf_supervised_solver_short_training_is_finite(
    dataset_kind: DatasetKind,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        supervised_solution,
        "load_hf_tabular_regression",
        lambda _config: _small_supervised_problem(),
    )
    telemetry = LossTelemetry(selected_node="target")
    packet = ExecutionPacket(telemetries=Telemetries((telemetry,)))
    for link_kind in ("perceptron", "phasor"):
        variant = next(
            variant
            for variant in supervised_bike_sharing_demand_demo.variants
            if variant.label == link_kind
        )
        variant_solver = variant.create_solver()
        assert isinstance(variant_solver, SupervisedSolver)
        solver = replace(
            variant_solver,
            dataset_kind=dataset_kind,
            training_examples=2,
            training_batch_size=4,
            hidden_size=8,
        )
        training_results = solver.training_results(packet=packet)
        losses = training_results.telemetries[telemetry]
        assert losses.shape[0] == solver.training_examples
        assert jnp.all(jnp.isfinite(losses))


def test_supervised_training_and_inference_support_non_divisible_scan_chunks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    examples = 5
    monkeypatch.setattr(
        supervised_solution,
        "load_hf_tabular_regression",
        lambda _config: _small_supervised_problem(),
    )
    telemetry = LossTelemetry(selected_node="target")
    packet = ExecutionPacket(
        telemetries=Telemetries((telemetry,)),
        scan_chunk_size=2,
    )
    variant_solver = supervised_bike_sharing_demand_demo.variants[0].create_solver()
    assert isinstance(variant_solver, SupervisedSolver)
    solver = replace(
        variant_solver,
        training_examples=examples,
        inference_examples=examples,
        training_batch_size=4,
        inference_batch_size=4,
        hidden_size=8,
    )

    training_results, inference_results = solver.training_and_inference_result(packet=packet)

    assert training_results.count == examples
    assert inference_results.count == examples
    assert training_results.telemetries[telemetry].shape == (examples,)
    assert inference_results.telemetries[telemetry].shape == (examples,)
    assert jnp.all(jnp.isfinite(training_results.telemetries[telemetry]))
    assert jnp.all(jnp.isfinite(inference_results.telemetries[telemetry]))


def test_phasor_supervised_scan_chunks_stack_spectral_telemetry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    examples = 5
    monkeypatch.setattr(
        supervised_solution,
        "load_hf_tabular_regression",
        lambda _config: _small_supervised_problem(),
    )
    variant = supervised_bike_sharing_demand_demo.variants[1]
    telemetry = SpectralLossTelemetry(selected_node="target")
    packet = ExecutionPacket(
        telemetries=variant.all_telemetries(),
        scan_chunk_size=2,
    )
    variant_solver = variant.create_solver()
    assert isinstance(variant_solver, SupervisedSolver)
    solver = replace(
        variant_solver,
        training_examples=examples,
        training_batch_size=4,
        hidden_size=8,
    )

    training_results = solver.training_results(packet=packet)

    assert training_results.count == examples
    assert training_results.telemetries[telemetry].shape == (examples,)
    assert jnp.all(jnp.isfinite(training_results.telemetries[telemetry]))


def _training_results_with_target_losses(losses: jnp.ndarray) -> TrainingResults:
    telemetry = LossTelemetry(selected_node="target")
    return TrainingResults(
        count=losses.shape[0],
        telemetries={telemetry: losses},
        final_state=cast("SolutionState", None),
    )


def _inference_results() -> InferenceResults:
    return InferenceResults(count=0, telemetries={})


def test_supervised_demo_loss_uses_final_quarter_mean() -> None:
    variants = supervised_bike_sharing_demand_demo.variants
    high_early_loss = jnp.array([100.0, 100.0, 100.0, 100.0, 4.0, 3.0, 2.0, 1.0])
    high_late_loss = jnp.array([4.0, 3.0, 2.0, 1.0, 100.0, 100.0, 100.0, 100.0])
    hyperparameters = {"training_examples": 8, "training_batch_size": 4, "hidden_size": 8}

    low_final_loss = supervised_bike_sharing_demand_demo.demo_loss(
        [
            (
                variants[0],
                _training_results_with_target_losses(high_early_loss),
                _inference_results(),
            ),
            (
                variants[1],
                _training_results_with_target_losses(high_early_loss),
                _inference_results(),
            ),
        ],
        hyperparameters,
    )
    high_final_loss = supervised_bike_sharing_demand_demo.demo_loss(
        [
            (
                variants[0],
                _training_results_with_target_losses(high_late_loss),
                _inference_results(),
            ),
            (
                variants[1],
                _training_results_with_target_losses(high_early_loss),
                _inference_results(),
            ),
        ],
        hyperparameters,
    )

    assert high_final_loss > low_final_loss


def test_supervised_demo_loss_rejects_fewer_than_four_training_examples() -> None:
    variants = supervised_bike_sharing_demand_demo.variants
    losses = jnp.array([2.0])

    with pytest.raises(ValueError, match="at least 4 training examples"):
        supervised_bike_sharing_demand_demo.demo_loss(
            [
                (
                    variants[0],
                    _training_results_with_target_losses(losses),
                    _inference_results(),
                ),
                (
                    variants[1],
                    _training_results_with_target_losses(losses),
                    _inference_results(),
                ),
            ],
            {"training_examples": 1, "training_batch_size": 4, "hidden_size": 8},
        )


def test_supervised_demo_loss_penalizes_compute_proxy() -> None:
    variants = supervised_bike_sharing_demand_demo.variants
    losses = jnp.array([2.0, 2.0, 2.0, 2.0])
    variant_results = [
        (variants[0], _training_results_with_target_losses(losses), _inference_results()),
        (variants[1], _training_results_with_target_losses(losses), _inference_results()),
    ]

    small = supervised_bike_sharing_demand_demo.demo_loss(
        variant_results,
        {"training_examples": 4, "training_batch_size": 4, "hidden_size": 8},
    )
    large = supervised_bike_sharing_demand_demo.demo_loss(
        variant_results,
        {"training_examples": 400, "training_batch_size": 32, "hidden_size": 128},
    )

    assert large > small


def _assert_demo_second_half_loss_below(demo: Demo, thresholds: dict[str, float]) -> None:
    telemetry = LossTelemetry(selected_node="target")
    packet = ExecutionPacket(telemetries=Telemetries((telemetry,)))
    for variant in demo.variants:
        threshold = thresholds[variant.label]
        solver = variant.create_solver()
        training_results = solver.training_results(packet=packet)
        losses = training_results.telemetries[telemetry]
        second_half_losses = losses[losses.shape[0] // 2 :]
        mean_loss = jnp.mean(second_half_losses)

        assert training_results.count == solver.training_examples
        assert jnp.all(jnp.isfinite(second_half_losses))
        assert mean_loss < threshold, (
            f"{demo.name} {variant.label} second-half mean loss {float(mean_loss)} >= {threshold}"
        )


def test_supervised_iris_demo_second_half_loss_is_low() -> None:
    _assert_demo_second_half_loss_below(
        supervised_iris_demo,
        thresholds={"perceptron": 52.0, "phasor": 52.0, "phasor-spectral": 52.0},
    )
