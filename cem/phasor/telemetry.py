from __future__ import annotations

from collections.abc import Mapping
from typing import override

from tjax.dataclasses import field

from cem.phasor.target_node import PhasorTargetConfiguration
from cem.structure.solution.inference import Inference, InferenceResult, TrainingResult
from cem.structure.solution.telemetry import Telemetry
from cem.structure.solution.training_solution import TrainingSolution


class SpectralLossTelemetry(Telemetry):
    """Telemetry that records the total spectral loss for a phasor target node."""

    selected_node: str = field(static=True)

    @override
    def training_snapshot(
        self,
        training_solution: TrainingSolution,
        training_result: TrainingResult,
        snapshots: Mapping[Telemetry, object],
    ) -> object | None:
        configuration = training_result.inference_result.model_configuration[self.selected_node]
        if not isinstance(configuration, PhasorTargetConfiguration):
            return None
        return configuration.total_spectral_loss()

    @override
    def inference_snapshot(
        self,
        inference: Inference,
        inference_result: InferenceResult,
        snapshots: Mapping[Telemetry, object],
    ) -> object | None:
        configuration = inference_result.model_configuration[self.selected_node]
        if not isinstance(configuration, PhasorTargetConfiguration):
            return None
        return configuration.total_spectral_loss()
