"""Supervised learning demo."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, override

import jax.numpy as jnp

from cem.structure.plotter import Demo, Plotter
from cem.structure.solution import InferenceResults, Telemetries, TrainingResults
from cem.structure.solution.loss_telemetry import LossTelemetry
from cem.structure.solver import Solver

from .plotter import SupervisedLossPlotter
from .solution import SupervisedSolver


class SupervisedDemo(Demo):
    """Demo for supervised learning on tabular datasets."""

    name = "supervised"
    title = "Supervised Learning"

    @classmethod
    @override
    def create_solver(cls) -> Solver[Any]:
        return SupervisedSolver()

    @classmethod
    @override
    def plotters(cls) -> Sequence[Plotter]:
        return [SupervisedLossPlotter()]

    @classmethod
    @override
    def extra_telemetries(cls) -> Telemetries:
        return Telemetries()

    @classmethod
    @override
    def demo_loss(
        cls, training_results: TrainingResults, inference_results: InferenceResults
    ) -> float:
        telemetry = LossTelemetry(selected_node="target")
        losses = inference_results.telemetries[telemetry]
        return float(jnp.mean(losses))
