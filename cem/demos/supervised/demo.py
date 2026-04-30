"""Supervised learning variant."""

from __future__ import annotations

from collections.abc import Sequence
from typing import override

import jax.numpy as jnp

from cem.structure.plotter import Demo, Plotter, Variant
from cem.structure.solution import InferenceResults, Telemetries, TrainingResults
from cem.structure.solution.loss_telemetry import LossTelemetry
from cem.structure.solver import Solver

from .plotter import SupervisedTrainingLossPlotter
from .problem import SupervisedProblem
from .solution import DatasetKind, LinkKind, SupervisedSolver


class SupervisedVariant(Variant):
    """Variant for supervised learning on tabular datasets."""

    def __init__(
        self,
        *,
        dataset_kind: DatasetKind,
        link_kind: LinkKind,
        use_spectral_loss: bool = False,
    ) -> None:
        self.dataset_kind = dataset_kind
        self.link_kind = link_kind
        self.use_spectral_loss = use_spectral_loss
        if link_kind == LinkKind.phasor and use_spectral_loss:
            self.label = "phasor-spectral"
        else:
            self.label = link_kind.name

    @override
    def create_solver(self) -> Solver[SupervisedProblem]:
        return SupervisedSolver(
            dataset_kind=self.dataset_kind,
            link_kind=self.link_kind,
            use_spectral_loss=self.use_spectral_loss,
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

    @override
    def demo_loss(
        self, training_results: TrainingResults, inference_results: InferenceResults
    ) -> float:
        telemetry = LossTelemetry(selected_node="target")
        losses = training_results.telemetries[telemetry]
        return float(jnp.mean(losses))


supervised_iris_demo = Demo(
    name="supervised-iris",
    variants=[
        SupervisedVariant(dataset_kind=DatasetKind.iris, link_kind=LinkKind.perceptron),
        SupervisedVariant(dataset_kind=DatasetKind.iris, link_kind=LinkKind.phasor),
    ],
)

supervised_iris_spectral_demo = Demo(
    name="supervised-iris-spectral",
    variants=[
        SupervisedVariant(dataset_kind=DatasetKind.iris, link_kind=LinkKind.perceptron),
        SupervisedVariant(
            dataset_kind=DatasetKind.iris,
            link_kind=LinkKind.phasor,
            use_spectral_loss=True,
        ),
    ],
)

supervised_synthetic_regression_demo = Demo(
    name="supervised-synthetic-regression",
    variants=[
        SupervisedVariant(
            dataset_kind=DatasetKind.synthetic_regression, link_kind=LinkKind.perceptron
        ),
        SupervisedVariant(dataset_kind=DatasetKind.synthetic_regression, link_kind=LinkKind.phasor),
    ],
)
