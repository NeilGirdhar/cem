from __future__ import annotations

import jax.numpy as jnp

from cem.structure.plotter import Demo, Plotter
from cem.structure.solution import InferenceResults, Telemetries, TrainingResults
from cem.structure.solver import Solver

from .plotter import AFPLossPlotter, AFPTelemetry
from .problem import IVProblem
from .solver import AFPIVSolver


class AFPIVDemo(Demo):
    name = "afp"
    title = "AFP IV: adversarial factor purification on a synthetic IV problem."

    @classmethod
    def create_solver(cls) -> Solver[IVProblem]:
        return AFPIVSolver(
            title="AFP IV",
            inference_examples=64,
            inference_batch_size=64,
            training_examples=256,
            training_batch_size=32,
            learning_rate=3e-3,
        )

    @classmethod
    def demo_loss(
        cls, training_results: TrainingResults, inference_results: InferenceResults
    ) -> float:
        telemetry = AFPTelemetry(selected_node="afp")
        losses = inference_results.telemetries[telemetry]
        return float(jnp.mean(losses.reconstruction))

    @classmethod
    def plotters(cls) -> list[Plotter]:
        return [AFPLossPlotter(selected_node="afp")]

    @classmethod
    def extra_telemetries(cls) -> Telemetries:
        return Telemetries((AFPTelemetry(selected_node="afp"),))
