from collections.abc import Mapping
from dataclasses import replace

import jax.numpy as jnp
import jax.random as jr
import pytest
from tjax import RngStream

import cem.demos.supervised.problem as supervised_problem
import cem.demos.supervised.solution as supervised_solution
from cem.demos.supervised.demo import supervised_bike_sharing_demand_demo, supervised_iris_demo
from cem.demos.supervised.problem import SupervisedProblem
from cem.demos.supervised.solution import (
    DatasetKind,
    GaussianNPNSupervisedModel,
    GaussianNPNTargetConfiguration,
    LinkKind,
    PerceptronSupervisedModel,
    SupervisedSolver,
)
from cem.perceptron.target_node import PerceptronTargetConfiguration
from cem.structure.solution import ExecutionPacket, LossTelemetry, Telemetries

from ._problems import small_multi_target_problem, small_supervised_problem

_IRIS_LOSS_THRESHOLD = 52.0


def test_perceptron_supervised_multi_target_infer_splits_target_fields(
    streams: Mapping[str, RngStream],
) -> None:
    problem = small_multi_target_problem()
    model = PerceptronSupervisedModel.create(problem, hidden_size=8, streams=streams)
    observation = problem.create_data_source().initial_problem_state(jr.key(0))

    result = model.infer(observation, None, streams=streams, inference=False)
    config = result.configurations["target"]
    assert isinstance(config, PerceptronTargetConfiguration)
    assert tuple(config.loss) == ("y_0", "y_1")
    assert jnp.isfinite(result.loss)


def test_gaussian_npn_supervised_multi_target_infer_splits_target_fields(
    streams: Mapping[str, RngStream],
) -> None:
    problem = small_multi_target_problem()
    model = GaussianNPNSupervisedModel.create(problem, hidden_size=8, streams=streams)
    observation = problem.create_data_source().initial_problem_state(jr.key(0))

    result = model.infer(observation, None, streams=streams, inference=True)
    config = result.configurations["target"]

    assert isinstance(config, GaussianNPNTargetConfiguration)
    assert tuple(config.loss) == ("y_0", "y_1")
    assert jnp.isfinite(result.loss)


@pytest.mark.parametrize("link_kind", [LinkKind.perceptron, LinkKind.natural_parameter])
def test_supervised_models_train_with_all_inputs_missing(
    link_kind: LinkKind,
    streams: Mapping[str, RngStream],
) -> None:
    """Missingness changes inputs while leaving observed targets available."""
    problem = small_multi_target_problem()
    if link_kind == LinkKind.perceptron:
        model = PerceptronSupervisedModel.create(
            problem,
            hidden_size=8,
            missing_probability=1.0,
            streams=streams,
        )
    else:
        model = GaussianNPNSupervisedModel.create(
            problem,
            hidden_size=8,
            missing_probability=1.0,
            streams=streams,
        )
    observation = problem.create_data_source().initial_problem_state(jr.key(0))

    result = model.infer(observation, None, streams=streams, inference=False)
    config = result.configurations["target"]

    assert isinstance(config, (PerceptronTargetConfiguration, GaussianNPNTargetConfiguration))
    assert tuple(config.observed_distributions) == ("y_0", "y_1")
    assert jnp.isfinite(result.loss)


def test_mask_aware_perceptron_receives_values_and_presence(
    streams: Mapping[str, RngStream],
) -> None:
    """The mask-aware baseline adds one presence input per observed value."""
    problem = small_multi_target_problem()
    model = PerceptronSupervisedModel.create(
        problem,
        hidden_size=8,
        missing_probability=0.5,
        random_missing_values=False,
        include_missingness_mask=True,
        streams=streams,
    )

    assert model.link.layers[0].weight.value.shape == (8, 2 * problem.n_features)

    observation = problem.create_data_source().initial_problem_state(jr.key(0))
    result = model.infer(observation, None, streams=streams, inference=False)
    assert jnp.isfinite(result.loss)


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
        lambda _config: small_supervised_problem(),
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


def test_gaussian_npn_supervised_solver_short_training_is_finite(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        supervised_solution,
        "load_hf_tabular_regression",
        lambda _config: small_supervised_problem(),
    )
    telemetry = LossTelemetry(selected_node="target")
    packet = ExecutionPacket(telemetries=Telemetries((telemetry,)))
    solver = SupervisedSolver(
        dataset_kind=DatasetKind.bike_sharing_demand,
        link_kind=supervised_solution.LinkKind.natural_parameter,
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
        lambda _config: small_supervised_problem(),
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
