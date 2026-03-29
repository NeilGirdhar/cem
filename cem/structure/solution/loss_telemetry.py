from collections.abc import Mapping
from typing import Any, override

from tjax.dataclasses import field

from cem.structure.model import Inference, InferenceResult, TrainingResult
from .telemetry import InferenceTelemetry, TrainingTelemetry
from .training_solution import TrainingSolution


class LossTrainingTelemetry(TrainingTelemetry):
    selected_node: str = field(static=True)

    @override
    def training_snapshot(
        self,
        training_solution: TrainingSolution,
        training_result: TrainingResult,
        snapshots: Mapping[TrainingTelemetry, Any],
    ) -> Any:
        configuration = training_result.inference_result.model_configuration[self.selected_node]
        return configuration.total_loss()


class LossInferenceTelemetry(InferenceTelemetry):
    selected_node: str = field(static=True)

    @override
    def inference_snapshot(
        self,
        inference: Inference,
        inference_result: InferenceResult,
        snapshots: Mapping[InferenceTelemetry, Any],
    ) -> Any:
        configuration = inference_result.model_configuration[self.selected_node]
        return configuration.total_loss()
