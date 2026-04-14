"""AFP IV demo."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, override

import jax.numpy as jnp

from cem.structure.plotter import Demo, Plotter
from cem.structure.solution import InferenceResults, Telemetries, TrainingResults
from cem.structure.solver import Solver

from .plotter import AFPLossPlotter, AFPTelemetry
from .solution import AFPSolver


class AFPDemo(Demo):
    """Demo for adversarial factor purification on the synthetic IV problem."""

    name = "afp"
    title = "AFP IV"

    @override
    def create_solver(self) -> Solver[Any]:
        return AFPSolver()

    @override
    def plotters(self) -> Sequence[Plotter]:
        return [AFPLossPlotter()]

    @override
    def extra_telemetries(self) -> Telemetries:
        return Telemetries()

    @override
    def demo_loss(
        self, training_results: TrainingResults, inference_results: InferenceResults
    ) -> float:
        telemetry = AFPTelemetry(selected_node="afp")
        config = inference_results.telemetries[telemetry]
        return float(jnp.mean(config.recon_loss))
