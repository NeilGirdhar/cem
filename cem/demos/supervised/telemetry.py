"""Telemetry for supervised Möbius candidate presence."""

from collections.abc import Mapping
from typing import Any, override

from tjax import JaxArray
from tjax.dataclasses import field

from cem.experimental.phasor.mobius_summation import MobiusSummationDiagnostics
from cem.experimental.phasor.target_node import PhasorTargetConfiguration
from cem.structure.solution.inference import Inference, InferenceResult, TrainingResult
from cem.structure.solution.telemetry import Telemetry
from cem.structure.solution.training_solution import TrainingSolution


class MobiusSummationTelemetry(Telemetry):
    """Record intermediate candidate presences for a selected Möbius summation."""

    selected_node: str = field(static=True, default="mobius")

    @staticmethod
    def _extract(configuration: object) -> MobiusSummationDiagnostics:
        assert isinstance(configuration, MobiusSummationDiagnostics)
        return configuration

    @override
    def training_snapshot(
        self,
        training_solution: TrainingSolution,
        training_result: TrainingResult,
        snapshots: Mapping[Telemetry, Any],
    ) -> MobiusSummationDiagnostics | None:
        configuration = training_result.inference_result.model_configuration.get(self.selected_node)
        return None if configuration is None else self._extract(configuration)

    @override
    def inference_snapshot(
        self,
        inference: Inference,
        inference_result: InferenceResult,
        snapshots: Mapping[Telemetry, Any],
    ) -> MobiusSummationDiagnostics | None:
        configuration = inference_result.model_configuration.get(self.selected_node)
        return None if configuration is None else self._extract(configuration)


class PhaseDomainTelemetry(Telemetry):
    """Record the target penalty for predictions outside the right semicircle."""

    selected_node: str = field(static=True, default="target")

    @staticmethod
    def _extract(configuration: object) -> JaxArray:
        assert isinstance(configuration, PhasorTargetConfiguration)
        return configuration.total_phase_domain_loss()

    @override
    def training_snapshot(
        self,
        training_solution: TrainingSolution,
        training_result: TrainingResult,
        snapshots: Mapping[Telemetry, Any],
    ) -> JaxArray | None:
        configuration = training_result.inference_result.model_configuration.get(self.selected_node)
        return None if configuration is None else self._extract(configuration)

    @override
    def inference_snapshot(
        self,
        inference: Inference,
        inference_result: InferenceResult,
        snapshots: Mapping[Telemetry, Any],
    ) -> JaxArray | None:
        configuration = inference_result.model_configuration.get(self.selected_node)
        return None if configuration is None else self._extract(configuration)
