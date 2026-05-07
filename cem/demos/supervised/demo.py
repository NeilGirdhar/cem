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

_COMPUTE_WEIGHT = 0.1


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
        for _variant, training_results, _inference_results in variant_results:
            losses = training_results.telemetries[telemetry]
            final_losses.append(_final_quarter_mean(losses))
        default_solver = self._default_solver()
        solver = default_solver.populate_from_hyperparameters(hyperparameters)
        compute_penalty = _COMPUTE_WEIGHT * (
            solver.compute_proxy() / default_solver.compute_proxy()
        )
        return float(jnp.max(jnp.asarray(final_losses)) + compute_penalty)

    def _default_solver(self) -> SupervisedSolver:
        solver = self.variants[0].create_solver()
        assert isinstance(solver, SupervisedSolver)
        return solver


SUPERVISED_MIN_TRAINING_EXAMPLES = 4


def _final_quarter_mean(losses: jnp.ndarray) -> jnp.ndarray:
    if losses.shape[0] < SUPERVISED_MIN_TRAINING_EXAMPLES:
        msg = "supervised demo scoring requires at least 4 training examples"
        raise ValueError(msg)
    quarter = max(1, losses.shape[0] // 4)
    return jnp.mean(losses[-quarter:])


supervised_iris_demo = SupervisedDemo(
    name="supervised-iris",
    variants=[
        SupervisedVariant(dataset_kind=DatasetKind.iris, link_kind=LinkKind.perceptron),
        SupervisedVariant(dataset_kind=DatasetKind.iris, link_kind=LinkKind.phasor),
    ],
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
)

supervised_elevators_demo = SupervisedDemo(
    name="supervised-elevators",
    variants=[
        SupervisedVariant(dataset_kind=DatasetKind.elevators, link_kind=LinkKind.perceptron),
        SupervisedVariant(dataset_kind=DatasetKind.elevators, link_kind=LinkKind.phasor),
    ],
)

supervised_cpu_activity_demo = SupervisedDemo(
    name="supervised-cpu-activity",
    variants=[
        SupervisedVariant(dataset_kind=DatasetKind.cpu_activity, link_kind=LinkKind.perceptron),
        SupervisedVariant(dataset_kind=DatasetKind.cpu_activity, link_kind=LinkKind.phasor),
    ],
)
