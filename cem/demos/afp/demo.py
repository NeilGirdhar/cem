"""AFP IV variant."""

from __future__ import annotations

from collections.abc import Sequence
from typing import override

import jax.numpy as jnp

from cem.demos.afp.problem import IVProblem
from cem.structure.plotter import Demo, Plotter, Variant
from cem.structure.solution import InferenceResults, Telemetries, TrainingResults
from cem.structure.solver import Solver

from .plotter import AFPLossPlotter, AFPTelemetry
from .solution import AFPSolver


class AFPVariant(Variant):
    """Variant for adversarial factor purification on the synthetic IV problem."""

    @override
    def create_solver(self) -> Solver[IVProblem]:
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
        # Shapes after telemetry stacking:
        # recon_loss, exo_loss, endo_loss: (inference_examples, inference_batch_size)
        # Each term is already a summed per-example objective.
        per_example_objective = config.recon_loss + config.exo_loss + config.endo_loss
        return float(jnp.mean(per_example_objective))


afp_demo = Demo(name="afp", variants=[AFPVariant()])
