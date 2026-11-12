"""Telemetry for comparing a Gaussian prediction's uncertainty with the observed target."""

from collections.abc import Mapping
from typing import Any, override

import jax.numpy as jnp
from efax import NormalEP
from tjax import JaxRealArray
from tjax.dataclasses import field

from cem.framework.graph.node import TargetConfiguration

from .inference import Inference, InferenceResult, TrainingResult
from .telemetry import Telemetry
from .training_solution import TrainingSolution


class PredictionTelemetry(Telemetry):
    """Record a scalar Gaussian prediction and its observed target during training or inference."""

    selected_node: str = field(static=True)
    selected_field: str = field(static=True)

    @override
    def training_snapshot(
        self,
        training_solution: TrainingSolution,
        training_result: TrainingResult,
        snapshots: Mapping[Telemetry, Any],
    ) -> JaxRealArray:
        return self._snapshot(training_result.inference_result)

    @override
    def inference_snapshot(
        self,
        inference: Inference,
        inference_result: InferenceResult,
        snapshots: Mapping[Telemetry, Any],
    ) -> JaxRealArray:
        return self._snapshot(inference_result)

    def _snapshot(self, inference_result: InferenceResult) -> JaxRealArray:
        """Return the predicted mean, predicted variance, and observed mean in that order.

        Both distributions must use Gaussian expectation parameters. The snapshot takes
        the first flattened entry from each statistic.
        """
        configuration = inference_result.model_configuration[self.selected_node]
        assert isinstance(configuration, TargetConfiguration)
        predicted = configuration.predicted_distributions[self.selected_field]
        observed = configuration.observed_distributions[self.selected_field]
        assert isinstance(predicted, NormalEP)
        assert isinstance(observed, NormalEP)
        predicted_mean = jnp.ravel(predicted.mean)[0]
        predicted_variance = jnp.ravel(predicted.second_moment)[0] - predicted_mean * predicted_mean
        observed_value = jnp.ravel(observed.mean)[0]
        return jnp.stack([predicted_mean, predicted_variance, observed_value])
