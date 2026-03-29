from __future__ import annotations

from collections.abc import Mapping
from dataclasses import fields
from functools import total_ordering
from typing import Any

import equinox as eqx
from tjax.dataclasses import field

from cem.structure.model import Inference, InferenceResult, TrainingResult
from .training_solution import TrainingSolution


@total_ordering
class Telemetry(eqx.Module):
    def _key(self) -> tuple[Any, ...]:
        return (id(type(self)), *(getattr(self, f.name) for f in fields(self) if f.compare))

    def __lt__(self, other: object) -> bool:
        if not isinstance(other, Telemetry):
            return NotImplemented
        return self._key() < other._key()


class TrainingTelemetry(Telemetry):
    """Collect data during training for use in loss estimation or plotting."""

    def training_snapshot(
        self,
        training_solution: TrainingSolution,
        training_result: TrainingResult,
        snapshots: Mapping[TrainingTelemetry, Any],
    ) -> object:
        """Collect a snapshot of data from one training example."""
        raise NotImplementedError


class InferenceTelemetry(Telemetry):
    """Collect data during inference for use in loss estimation or plotting."""

    def inference_snapshot(
        self,
        inference: Inference,
        inference_result: InferenceResult,
        snapshots: Mapping[InferenceTelemetry, Any],
    ) -> object:
        """Collect a snapshot of data from one training example."""
        raise NotImplementedError


class Telemetries(eqx.Module):
    training_telemetries: tuple[TrainingTelemetry, ...] = field(default_factory=tuple)
    inference_telemetries: tuple[InferenceTelemetry, ...] = field(default_factory=tuple)

    def combine(self, other: Telemetries) -> Telemetries:
        def combine_element[T: Telemetry](x: tuple[T, ...], y: tuple[T, ...]) -> tuple[T, ...]:
            result = list(x)
            seen = set(x)
            for yi in y:
                if yi in seen:
                    continue
                seen.add(yi)
                result.append(yi)
            return tuple(result)

        return Telemetries(
            combine_element(self.training_telemetries, other.training_telemetries),
            combine_element(self.inference_telemetries, other.inference_telemetries),
        )
