from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace

import jax.numpy as jnp
import jax.random as jr
import matplotlib as mpl
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
    supervised_synthetic_regression_demo,
)
from cem.demos.supervised.problem import SupervisedProblem, load_synthetic_regression
from cem.demos.supervised.solution import DatasetKind, PhasorSupervisedModel
from cem.phasor import PhasorTargetConfiguration
from cem.structure.plotter import Demo
from cem.structure.solution import (
    ExecutionPacket,
    LossTelemetry,
    Telemetries,
)

mpl.use("Agg")


_EXPECTED_HF_TEST_FEATURES = 2
_EXPECTED_HF_TEST_ROWS = 5


def _small_supervised_problem() -> SupervisedProblem:
    rng = np.random.default_rng(0)
    x = rng.normal(size=(32, 4))
    y = 0.5 * x[:, 0] - 0.25 * x[:, 1] + 0.1 * rng.normal(size=32)
    df = pd.DataFrame({f"x_{i}": x[:, i] for i in range(x.shape[1])})
    df["target"] = y
    return supervised_problem.problem_from_numeric_dataframe(df, max_rows=16, seed=0)


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
    assert config.score.shape == (problem.n_targets * 10,)
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
        solver = replace(
            variant.create_solver(),
            dataset_kind=dataset_kind,
            training_examples=2,
            training_batch_size=4,
            hidden_size=8,
        )
        training_results = solver.training_results(packet=packet)
        losses = training_results.telemetries[telemetry]
        assert losses.shape[0] == solver.training_examples
        assert jnp.all(jnp.isfinite(losses))


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


def test_supervised_synthetic_regression_demo_second_half_loss_is_low() -> None:
    _assert_demo_second_half_loss_below(
        supervised_synthetic_regression_demo,
        thresholds={"perceptron": 120.0, "phasor": 230.0, "phasor-spectral": 120.0},
    )
