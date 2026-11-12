"""Interfaces for collecting and combining training and inference snapshots."""

from collections.abc import Mapping
from dataclasses import fields
from functools import total_ordering
from typing import Any

import equinox as eqx
from tjax.dataclasses import field

from cem.framework.solution.inference import Inference, InferenceResult, TrainingResult

from .training_solution import TrainingSolution


@total_ordering
class Telemetry(eqx.Module):
    """A selectable measurement of one episode's training or inference results."""

    def _key(self) -> tuple[Any, ...]:
        return (id(type(self)), *(getattr(self, f.name) for f in fields(self) if f.compare))

    def __lt__(self, other: object) -> bool:
        if not isinstance(other, Telemetry):
            return NotImplemented
        return self._key() < other._key()

    def training_snapshot(
        self,
        training_solution: TrainingSolution,
        training_result: TrainingResult,
        snapshots: Mapping[Telemetry, Any],
    ) -> object | None:
        """Collect a training snapshot using any earlier snapshots from the same episode.

        Return none to omit the measurement.
        """
        return None

    def inference_snapshot(
        self,
        inference: Inference,
        inference_result: InferenceResult,
        snapshots: Mapping[Telemetry, Any],
    ) -> object | None:
        """Collect an inference snapshot using any earlier snapshots from the same episode.

        Return none to omit the measurement.
        """
        return None


class Telemetries(eqx.Module):
    """An ordered collection of telemetry requests."""

    telemetries: tuple[Telemetry, ...] = field(default_factory=tuple)

    def combine(self, other: Telemetries) -> Telemetries:
        """Append requests from the other collection that are absent from this one."""
        result = list(self.telemetries)
        seen = set(self.telemetries)
        for t in other.telemetries:
            if t not in seen:
                seen.add(t)
                result.append(t)
        return Telemetries(tuple(result))
