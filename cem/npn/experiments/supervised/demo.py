"""Shared :class:`Demo` subclass for the supervised experiments."""

from collections.abc import Sequence
from typing import Any, override

import jax.numpy as jnp

from cem.structure.plotter import Demo, Variant
from cem.structure.solution import InferenceResults, TrainingResults
from cem.structure.solution.loss_telemetry import LossTelemetry

from .models import SupervisedSolver
from .plotter import SupervisedTrainingLossPlotter

_COMPUTE_WEIGHT = 0.1
_SHARED_HYPERPARAMETER_NAMES = frozenset({"training_examples", "training_batch_size"})


class SupervisedDemo(Demo):
    """Supervised demo scored across its model variants with a compute penalty."""

    def __init__(self, *, name: str, variants: Sequence[Variant]) -> None:
        super().__init__(
            name=name,
            variants=variants,
            plotters=[SupervisedTrainingLossPlotter()],
            shared_hyperparameter_names=_SHARED_HYPERPARAMETER_NAMES,
        )

    @override
    def demo_loss(
        self,
        variant_results: Sequence[tuple[Variant, TrainingResults, InferenceResults]],
        hyperparameters: dict[str, Any],
    ) -> float:
        telemetry = LossTelemetry(selected_node="target")
        penalized_losses: list[jnp.ndarray] = []
        for variant, _training_results, inference_results in variant_results:
            if inference_results.count < 1:
                msg = "supervised demo scoring requires inference results"
                raise ValueError(msg)
            prefix = f"{variant.label}."
            variant_hyperparameters = {
                key: value
                for key, value in hyperparameters.items()
                if key in self.shared_hyperparameter_names
            }
            variant_hyperparameters.update(
                {
                    key[len(prefix) :]: value
                    for key, value in hyperparameters.items()
                    if key.startswith(prefix)
                }
            )
            default_solver = variant.solver
            assert isinstance(default_solver, SupervisedSolver)
            solver = default_solver.populate_from_hyperparameters(variant_hyperparameters)
            compute_penalty = _COMPUTE_WEIGHT * (
                solver.compute_proxy() / default_solver.compute_proxy()
            )
            inference_loss = jnp.mean(inference_results.telemetries[telemetry])
            penalized_losses.append(inference_loss + compute_penalty)
        return float(jnp.max(jnp.asarray(penalized_losses)))
