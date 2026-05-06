"""Supervised learning variant."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, override

import jax.numpy as jnp

from cem.structure.plotter import Demo, Plotter, Variant
from cem.structure.solution import InferenceResults, Telemetries, TrainingResults
from cem.structure.solution.loss_telemetry import LossTelemetry
from cem.structure.solver import Solver

from .plotter import SupervisedTrainingLossPlotter
from .problem import SupervisedProblem
from .solution import (
    DatasetKind,
    LinkKind,
    SupervisedSolver,
)

_PLATEAU_WEIGHT = 10.0
_COMPUTE_WEIGHT = 0.1
_LOSS_EPSILON = 1e-12


class SupervisedVariant(Variant):
    """Variant for supervised learning on tabular datasets."""

    def __init__(
        self,
        *,
        dataset_kind: DatasetKind,
        link_kind: LinkKind,
    ) -> None:
        self.dataset_kind = dataset_kind
        self.link_kind = link_kind
        self.label = link_kind.name

    @override
    def create_solver(self) -> Solver[SupervisedProblem]:
        return SupervisedSolver(
            dataset_kind=self.dataset_kind,
            link_kind=self.link_kind,
        )

    @override
    def plotters(self) -> Sequence[Plotter]:
        return [SupervisedTrainingLossPlotter()]

    @override
    def extra_telemetries(self) -> Telemetries:
        return Telemetries()

    @override
    def shared_hyperparameter_names(self) -> frozenset[str]:
        return frozenset(
            {
                "training_examples",
                "training_batch_size",
                "hidden_size",
            }
        )


class SupervisedDemo(Demo):
    """Supervised demo scored jointly across perceptron and phasor variants."""

    def demo_loss(
        self,
        variant_results: Sequence[tuple[Variant, TrainingResults, InferenceResults]],
        hyperparameters: dict[str, Any],
    ) -> float:
        telemetry = LossTelemetry(selected_node="target")
        final_losses: list[jnp.ndarray] = []
        plateau_gaps: list[jnp.ndarray] = []
        for _variant, training_results, _inference_results in variant_results:
            losses = training_results.telemetries[telemetry]
            plateau_gap, final_mean = _plateau_gap_and_final_mean(losses)
            final_losses.append(final_mean)
            plateau_gaps.append(plateau_gap)
        default_solver = self._default_solver()
        solver = default_solver.populate_from_hyperparameters(hyperparameters)
        compute_penalty = _COMPUTE_WEIGHT * (
            solver.compute_proxy() / default_solver.compute_proxy()
        )
        return float(
            jnp.max(jnp.asarray(final_losses))
            + _PLATEAU_WEIGHT * jnp.max(jnp.asarray(plateau_gaps))
            + compute_penalty
        )

    def _default_solver(self) -> SupervisedSolver:
        solver = self.variants[0].create_solver()
        assert isinstance(solver, SupervisedSolver)
        return solver


# TODO: Move this to a shared demo-loss utility once another demo needs plateau scoring.
SUPERVISED_MIN_TRAINING_EXAMPLES = 4


def _plateau_gap_and_final_mean(losses: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
    if losses.shape[0] < SUPERVISED_MIN_TRAINING_EXAMPLES:
        msg = "supervised demo plateau scoring requires at least 4 training examples"
        raise ValueError(msg)
    quarter = max(1, losses.shape[0] // 4)
    final_mean = jnp.mean(losses[-quarter:])
    second_last_mean = jnp.mean(losses[-2 * quarter : -quarter])
    plateau_gap = jnp.abs(final_mean - second_last_mean) / (second_last_mean + _LOSS_EPSILON)
    return plateau_gap, final_mean


supervised_iris_demo = SupervisedDemo(
    name="supervised-iris",
    variants=[
        SupervisedVariant(dataset_kind=DatasetKind.iris, link_kind=LinkKind.perceptron),
        SupervisedVariant(dataset_kind=DatasetKind.iris, link_kind=LinkKind.phasor),
    ],
    tuned_hyperparameters={
        "hidden_size": 30,
        "perceptron.learning_rate": 0.30476574430714515,
        "phasor.learning_rate": 0.4256399175470551,
        "phasor.n_frequencies": 12,
        "training_examples": 724,
    },
)

supervised_bike_sharing_demand_demo = SupervisedDemo(
    name="supervised-bike-sharing-demand",
    variants=[
        SupervisedVariant(
            dataset_kind=DatasetKind.bike_sharing_demand,
            link_kind=LinkKind.perceptron,
        ),
        SupervisedVariant(
            dataset_kind=DatasetKind.bike_sharing_demand,
            link_kind=LinkKind.phasor,
        ),
    ],
    tuned_hyperparameters={
        "hidden_size": 24,
        "perceptron.learning_rate": 0.4071429559809199,
        "phasor.learning_rate": 0.6781247399981128,
        "phasor.n_frequencies": 15,
        "training_examples": 677,
    },
)

supervised_elevators_demo = SupervisedDemo(
    name="supervised-elevators",
    variants=[
        SupervisedVariant(dataset_kind=DatasetKind.elevators, link_kind=LinkKind.perceptron),
        SupervisedVariant(dataset_kind=DatasetKind.elevators, link_kind=LinkKind.phasor),
    ],
    tuned_hyperparameters={
        "hidden_size": 81,
        "perceptron.learning_rate": 0.0016592994855346016,
        "phasor.learning_rate": 0.030034085627541237,
        "phasor.n_frequencies": 15,
        "training_examples": 414,
    },
)

supervised_cpu_activity_demo = SupervisedDemo(
    name="supervised-cpu-activity",
    variants=[
        SupervisedVariant(dataset_kind=DatasetKind.cpu_activity, link_kind=LinkKind.perceptron),
        SupervisedVariant(dataset_kind=DatasetKind.cpu_activity, link_kind=LinkKind.phasor),
    ],
    tuned_hyperparameters={
        "hidden_size": 125,
        "perceptron.learning_rate": 0.000302148338976774,
        "phasor.learning_rate": 0.0023416881957209408,
        "phasor.n_frequencies": 16,
        "training_examples": 471,
    },
)
