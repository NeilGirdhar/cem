"""AFP synthetic IV demo."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, override

import jax.numpy as jnp

from cem.demos.afp.problem import IVProblem
from cem.structure.plotter import Demo, Plotter, Variant
from cem.structure.solution import InferenceResults, Telemetries, TrainingResults
from cem.structure.solver import Solver

from .plotter import AFPGammaRecoveryPlotter, AFPLossPlotter, AFPTelemetry
from .solution import AFPSolver


class AFPVariant(Variant):
    """Variant for adversarial factor purification on the parameterized IV problem."""

    def _create_afp_solver(self) -> AFPSolver:
        return AFPSolver(
            n_instruments=4,
            n_confounders=3,
            n_treatments=3,
            n_outcomes=1,
        )

    @override
    def create_solver(self) -> Solver[IVProblem]:
        return self._create_afp_solver()

    @override
    def plotters(self) -> Sequence[Plotter]:
        return [AFPLossPlotter(), AFPGammaRecoveryPlotter(solver=self._create_afp_solver())]

    @override
    def extra_telemetries(self) -> Telemetries:
        return Telemetries()


class AFPDemo(Demo):
    """AFP demo scored from the full set of variant results."""

    def demo_loss(
        self,
        variant_results: Sequence[tuple[Variant, TrainingResults, InferenceResults]],
        hyperparameters: dict[str, Any],
    ) -> float:
        del hyperparameters
        _variant, training_results, inference_results = variant_results[0]
        del inference_results
        telemetry = AFPTelemetry(selected_node="afp")
        config = training_results.telemetries[telemetry]
        # Shapes after telemetry stacking:
        # recon_loss, exo_loss, endo_loss: (training_examples, training_batch_size)
        # Each term is already a summed per-example objective.
        per_example_objective = config.recon_loss + config.exo_loss + config.endo_loss
        return float(jnp.mean(per_example_objective))


afp_synthetic_iv_demo = AFPDemo(name="afp-synthetic-iv", variants=[AFPVariant()])
