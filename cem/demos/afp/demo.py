"""AFP synthetic IV demo."""

from collections.abc import Sequence
from typing import Any, override

import jax.numpy as jnp

from cem.demos.afp.problem import IVProblem
from cem.structure.plotter import Demo, Plotter, Variant
from cem.structure.solution import InferenceResults, Telemetries, TrainingResults
from cem.structure.solver import Solver

from .plotter import (
    AFPCausalExoPlotter,
    AFPConfoundedEndoPlotter,
    AFPCrossEnvPlotter,
    AFPLossPlotter,
    AFPTelemetry,
)
from .solution import AFPSolver


class AFPVariant(Variant):
    """Variant for adversarial factor purification on the parameterized IV problem."""

    def _create_afp_solver(self) -> AFPSolver:
        return AFPSolver(
            n_instruments=4,
            n_confounders=3,
            n_candidate_confounders=3,
            n_outcomes=1,
            n_environments=2,
        )

    @override
    def create_solver(self) -> Solver[IVProblem]:
        return self._create_afp_solver()

    @override
    def plotters(self) -> Sequence[Plotter]:
        return [
            AFPLossPlotter(),
            AFPCausalExoPlotter(solver=self._create_afp_solver()),
            AFPConfoundedEndoPlotter(solver=self._create_afp_solver()),
            AFPCrossEnvPlotter(solver=self._create_afp_solver()),
        ]

    @override
    def extra_telemetries(self) -> Telemetries:
        return Telemetries()


class AFPDemo(Demo):
    """AFP demo scored on a leak-rectified mean of the three training losses."""

    def demo_loss(
        self,
        variant_results: Sequence[tuple[Variant, TrainingResults, InferenceResults]],
        hyperparameters: dict[str, Any],
    ) -> float:
        """Mean over training of ``recon + relu(exo) + relu(endo)``.

        ``recon`` is non-negative by construction.  The adversarial terms
        (``exo``, ``endo``) are signed gains over the uniform-predictor baseline:
        positive means the critic extracted information (a leak we want to
        penalize), negative means the producer pushed past the critic (which
        does not by itself help causal identification).  ReLU-clipping the
        adversarial terms keeps the meta-objective monotone with respect to
        information leak while closing the "drive adversarial loss strongly
        negative" loophole that a plain sum allowed.
        """
        del hyperparameters
        _variant, training_results, _ = variant_results[0]
        telemetry = AFPTelemetry(selected_node="afp")
        config = training_results.telemetries[telemetry]
        per_step = (
            config.recon_loss
            + jnp.maximum(0.0, config.exo_loss)
            + jnp.maximum(0.0, config.endo_loss)
        )
        return float(jnp.mean(per_step))


afp_synthetic_iv_demo = AFPDemo(name="afp-synthetic-iv", variants=[AFPVariant()])
