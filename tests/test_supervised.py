from collections.abc import Mapping
from dataclasses import replace
from typing import cast

import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pandas as pd
import pytest
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
    PhasorSupervisedModel,
    SupervisedSolver,
)
from cem.perceptron.target_node import PerceptronTargetConfiguration
from cem.phasor.target_node import PhasorTargetConfiguration
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
_EXPECTED_HF_SELECTED_ROWS = 5
_EXPECTED_HF_TRAINING_ROWS = 4
_EXPECTED_HF_INFERENCE_ROWS = 1
_IRIS_LOSS_THRESHOLD = 52.0


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


def test_perceptron_supervised_multi_target_infer_splits_target_fields(
    streams: Mapping[str, RngStream],
) -> None:
    problem = _small_multi_target_problem()
    model = PerceptronSupervisedModel.create(problem, hidden_size=8, streams=streams)
    observation = problem.create_data_source().initial_problem_state(jr.key(0))

    result = model.infer(observation, None, streams=streams, inference=False)
    config = result.configurations["target"]
    assert isinstance(config, PerceptronTargetConfiguration)
    assert tuple(config.loss) == ("y_0", "y_1")
    assert jnp.isfinite(result.loss)


def test_phasor_supervised_multi_target_infer_splits_target_fields(
    streams: Mapping[str, RngStream],
) -> None:
    problem = _small_multi_target_problem()
    model = PhasorSupervisedModel.create(problem, hidden_size=8, streams=streams)
    observation = problem.create_data_source().initial_problem_state(jr.key(0))

    result = model.infer(observation, None, streams=streams, inference=False)
    config = result.configurations["target"]
    assert isinstance(config, PhasorTargetConfiguration)
    assert tuple(config.loss) == ("y_0", "y_1")
    assert config.score.shape == (problem.n_targets,)
    assert jnp.isfinite(result.loss)


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
        max_rows=_EXPECTED_HF_SELECTED_ROWS,
        seed=7,
    )
    second = supervised_problem.problem_from_numeric_dataframe(
        df,
        max_rows=_EXPECTED_HF_SELECTED_ROWS,
        seed=7,
    )

    assert first.n_features == _EXPECTED_HF_TEST_FEATURES
    assert first.n_targets == 1
    assert first.training.x_flat.shape == (
        _EXPECTED_HF_TRAINING_ROWS,
        _EXPECTED_HF_TEST_FEATURES,
    )
    assert first.training.y_flat.shape == (_EXPECTED_HF_TRAINING_ROWS, 1)
    assert first.inference.x_flat.shape == (
        _EXPECTED_HF_INFERENCE_ROWS,
        _EXPECTED_HF_TEST_FEATURES,
    )
    assert first.inference.y_flat.shape == (_EXPECTED_HF_INFERENCE_ROWS, 1)
    assert jnp.all(jnp.isfinite(first.training.x_flat))
    assert jnp.all(jnp.isfinite(first.training.y_flat))
    assert jnp.all(jnp.isfinite(first.inference.x_flat))
    assert jnp.all(jnp.isfinite(first.inference.y_flat))
    assert jnp.allclose(first.training.x_flat, second.training.x_flat)
    assert jnp.allclose(first.training.y_flat, second.training.y_flat)
    assert jnp.allclose(first.inference.x_flat, second.inference.x_flat)
    assert jnp.allclose(first.inference.y_flat, second.inference.y_flat)
    assert not jnp.any(
        jnp.all(
            first.training.x_flat == first.inference.x_flat[0][jnp.newaxis, :],
            axis=-1,
        )
    )


def test_supervised_problem_selects_training_and_inference_sources() -> None:
    problem = _small_supervised_problem()

    training = problem.create_data_source(inference=False)
    inference = problem.create_data_source(inference=True)

    assert training is problem.training
    assert inference is problem.inference


@pytest.mark.parametrize("source_name", ["training", "inference"])
def test_supervised_sources_reuse_rows_for_common_keys(source_name: str) -> None:
    first = getattr(_small_supervised_problem(), source_name)
    second = getattr(_small_supervised_problem(), source_name)
    keys = jr.split(jr.key(23), 16)

    first_states = [first.initial_problem_state(key) for key in keys]
    second_states = [second.initial_problem_state(key) for key in keys]

    for first_state, second_state in zip(first_states, second_states, strict=True):
        assert jnp.array_equal(first_state.x, second_state.x)
        assert jnp.array_equal(first_state.y, second_state.y)


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
    variant_solver = supervised_bike_sharing_demand_demo.variants[0].create_solver()
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


def test_phasor_supervised_solver_short_training_is_finite(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        supervised_solution,
        "load_hf_tabular_regression",
        lambda _config: _small_supervised_problem(),
    )
    telemetry = LossTelemetry(selected_node="target")
    packet = ExecutionPacket(telemetries=Telemetries((telemetry,)))
    variant_solver = supervised_bike_sharing_demand_demo.variants[1].create_solver()
    assert isinstance(variant_solver, SupervisedSolver)
    solver = replace(
        variant_solver,
        training_examples=2,
        training_batch_size=4,
        hidden_size=8,
    )

    training_results = solver.training_results(packet=packet)
    losses = training_results.telemetries[telemetry]

    assert losses.shape == (solver.training_examples,)
    assert jnp.all(jnp.isfinite(losses))


def test_supervised_training_and_inference_support_non_divisible_scan_chunks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    examples = 5
    data_source_modes: list[bool] = []
    original_create_data_source = SupervisedProblem.create_data_source

    def tracked_create_data_source(
        problem: SupervisedProblem,
        *,
        inference: bool = False,
    ) -> supervised_problem.SupervisedDataSource:
        data_source_modes.append(inference)
        return original_create_data_source(problem, inference=inference)

    monkeypatch.setattr(SupervisedProblem, "create_data_source", tracked_create_data_source)
    monkeypatch.setattr(
        supervised_solution,
        "load_hf_tabular_regression",
        lambda _config: _small_supervised_problem(),
    )
    telemetry = LossTelemetry(selected_node="target")
    packet = ExecutionPacket(telemetries=Telemetries((telemetry,)), scan_chunk_size=2)
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
    assert False in data_source_modes
    assert True in data_source_modes


def _training_results_with_target_losses(losses: jnp.ndarray) -> TrainingResults:
    telemetry = LossTelemetry(selected_node="target")
    return TrainingResults(
        count=losses.shape[0],
        telemetries={telemetry: losses},
        final_state=cast("SolutionState", None),
    )


def _inference_results(losses: jnp.ndarray) -> InferenceResults:
    telemetry = LossTelemetry(selected_node="target")
    return InferenceResults(count=losses.shape[0], telemetries={telemetry: losses})


def test_supervised_demo_loss_uses_inference_loss() -> None:
    variant = supervised_bike_sharing_demand_demo.variants[0]
    training_losses = jnp.array([4.0, 3.0, 2.0, 1.0])
    low_inference_loss = jnp.array([1.0, 2.0])
    high_inference_loss = jnp.array([100.0, 100.0])
    hyperparameters = {"training_examples": 8, "training_batch_size": 4, "hidden_size": 8}

    low_loss = supervised_bike_sharing_demand_demo.demo_loss(
        [
            (
                variant,
                _training_results_with_target_losses(training_losses),
                _inference_results(low_inference_loss),
            )
        ],
        hyperparameters,
    )
    high_loss = supervised_bike_sharing_demand_demo.demo_loss(
        [
            (
                variant,
                _training_results_with_target_losses(training_losses),
                _inference_results(high_inference_loss),
            )
        ],
        hyperparameters,
    )

    assert high_loss > low_loss


def test_supervised_demo_loss_requires_inference_results() -> None:
    variant = supervised_bike_sharing_demand_demo.variants[0]
    losses = jnp.array([2.0])

    with pytest.raises(ValueError, match="requires inference results"):
        supervised_bike_sharing_demand_demo.demo_loss(
            [
                (
                    variant,
                    _training_results_with_target_losses(losses),
                    InferenceResults(count=0, telemetries={}),
                )
            ],
            {"training_examples": 1, "training_batch_size": 4, "hidden_size": 8},
        )


def test_supervised_demo_loss_penalizes_compute_proxy() -> None:
    variant = supervised_bike_sharing_demand_demo.variants[0]
    losses = jnp.array([2.0, 2.0, 2.0, 2.0])
    variant_results = [
        (
            variant,
            _training_results_with_target_losses(losses),
            _inference_results(losses),
        )
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


def test_supervised_iris_demo_second_half_loss_is_low() -> None:
    telemetry = LossTelemetry(selected_node="target")
    packet = ExecutionPacket(telemetries=Telemetries((telemetry,)))
    variant = supervised_iris_demo.variants[0]
    solver = variant.create_solver()
    training_results = solver.training_results(packet=packet)
    losses = training_results.telemetries[telemetry]
    second_half_losses = losses[losses.shape[0] // 2 :]

    assert training_results.count == solver.training_examples
    assert jnp.all(jnp.isfinite(second_half_losses))
    assert jnp.mean(second_half_losses) < _IRIS_LOSS_THRESHOLD
