from __future__ import annotations

from collections.abc import Mapping

import jax.numpy as jnp
import jax.random as jr
from tjax import RngStream

from cem.demos.supervised.demo import (
    supervised_iris_demo,
    supervised_iris_spectral_demo,
    supervised_synthetic_regression_demo,
)
from cem.demos.supervised.problem import load_synthetic_regression
from cem.demos.supervised.solution import PhasorSupervisedModel
from cem.phasor import PhasorTargetConfiguration
from cem.structure.plotter import Demo
from cem.structure.solution import ExecutionPacket, LossTelemetry, Telemetries


def test_phasor_supervised_multi_target_infer_splits_target_fields(
    streams: Mapping[str, RngStream],
) -> None:
    problem = load_synthetic_regression()
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
    assert config.score.data.shape == (problem.n_targets * 10,)
    assert jnp.isfinite(result.loss)


def _assert_demo_second_half_loss_below(demo: Demo, threshold: float) -> None:
    telemetry = LossTelemetry(selected_node="target")
    packet = ExecutionPacket(telemetries=Telemetries((telemetry,)))
    for variant in demo.variants:
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
    _assert_demo_second_half_loss_below(supervised_iris_demo, threshold=50.0)


def test_supervised_iris_spectral_demo_second_half_loss_is_low() -> None:
    _assert_demo_second_half_loss_below(supervised_iris_spectral_demo, threshold=52.0)


def test_supervised_synthetic_regression_demo_second_half_loss_is_low() -> None:
    _assert_demo_second_half_loss_below(supervised_synthetic_regression_demo, threshold=120.0)
