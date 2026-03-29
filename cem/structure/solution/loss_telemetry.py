from collections.abc import Mapping
from typing import Any, override

from tjax.dataclasses import field

from cem.structure.solution.inference import Inference, InferenceResult, TrainingResult

from .telemetry import Telemetry
from .training_solution import TrainingSolution


class LossTelemetry(Telemetry):
    """Telemetry that records the total loss for a selected node."""

    selected_node: str = field(static=True)

    @override
    def training_snapshot(
        self,
        training_solution: TrainingSolution,
        training_result: TrainingResult,
        snapshots: Mapping[Telemetry, Any],
    ) -> Any:
        configuration = training_result.inference_result.model_configuration[self.selected_node]
        return configuration.total_loss()

    @override
    def inference_snapshot(
        self,
        inference: Inference,
        inference_result: InferenceResult,
        snapshots: Mapping[Telemetry, Any],
    ) -> Any:
        configuration = inference_result.model_configuration[self.selected_node]
        return configuration.total_loss()
