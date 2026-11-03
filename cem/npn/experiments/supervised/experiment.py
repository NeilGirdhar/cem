"""Shared :class:`Experiment` subclass for the supervised experiments."""

from collections.abc import Mapping, Sequence
from typing import Any, override

import jax.numpy as jnp

from cem.structure.plotter import OptimizableExperiment, Run, Variant
from cem.structure.solution.loss_telemetry import LossTelemetry

from .models import SupervisedSolver
from .plotter import SupervisedTrainingLossPlotter

_COMPUTE_WEIGHT = 0.1
_SHARED_HYPERPARAMETER_NAMES = frozenset({"training_examples", "training_batch_size"})


class SupervisedExperiment(OptimizableExperiment):
    """Supervised experiment scored across its model variants with a compute penalty."""

    def __init__(self, *, name: str, variants: Sequence[Variant]) -> None:
        super().__init__(
            name=name,
            variants=variants,
            plotters=[SupervisedTrainingLossPlotter()],
            shared_hyperparameter_names=_SHARED_HYPERPARAMETER_NAMES,
        )

    @override
    def experiment_loss(
        self,
        runs: Sequence[Run],
        hyperparameters: Mapping[str, Any],
    ) -> float:
        telemetry = LossTelemetry(selected_node="target")
        penalized_losses: list[jnp.ndarray] = []
        for run in runs:
            if run.inference_results.count < 1:
                msg = "supervised experiment scoring requires inference results"
                raise ValueError(msg)
            default_solver = self.variant_for(run).solver
            solver = self.solver_for(run, hyperparameters)
            assert isinstance(default_solver, SupervisedSolver)
            assert isinstance(solver, SupervisedSolver)
            compute_penalty = _COMPUTE_WEIGHT * (
                solver.compute_proxy() / default_solver.compute_proxy()
            )
            inference_loss = jnp.mean(run.inference_results.telemetries[telemetry])
            penalized_losses.append(inference_loss + compute_penalty)
        return float(jnp.max(jnp.asarray(penalized_losses)))
