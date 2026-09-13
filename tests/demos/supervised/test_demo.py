from typing import cast

import jax.numpy as jnp
import pytest

from cem.commands.demos import DemoEnum, demo_registry
from cem.demos.supervised.demo import (
    supervised_bike_sharing_demand_demo,
    supervised_cpu_activity_demo,
    supervised_elevators_demo,
)
from cem.structure.plotter import Demo
from cem.structure.solution import (
    InferenceResults,
    LossTelemetry,
    SolutionState,
    TrainingResults,
)


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
    assert [variant.label for variant in demo.variants] == ["perceptron", "natural_parameter"]


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
        {
            "training_examples": 4,
            "training_batch_size": 4,
            "perceptron.hidden_size": 8,
        },
    )
    large = supervised_bike_sharing_demand_demo.demo_loss(
        variant_results,
        {
            "training_examples": 400,
            "training_batch_size": 32,
            "perceptron.hidden_size": 128,
        },
    )

    assert large > small
