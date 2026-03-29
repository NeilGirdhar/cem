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

    def training_snapshot(
        self,
        training_solution: TrainingSolution,
        training_result: TrainingResult,
        snapshots: Mapping[Telemetry, Any],
    ) -> object | None:
        """Collect a snapshot of data from one training example."""
        return None

    def inference_snapshot(
        self,
        inference: Inference,
        inference_result: InferenceResult,
        snapshots: Mapping[Telemetry, Any],
    ) -> object | None:
        """Collect a snapshot of data from one inference example."""
        return None


class Telemetries(eqx.Module):
    telemetries: tuple[Telemetry, ...] = field(default_factory=tuple)

    def combine(self, other: Telemetries) -> Telemetries:
        result = list(self.telemetries)
        seen = set(self.telemetries)
        for t in other.telemetries:
            if t not in seen:
                seen.add(t)
                result.append(t)
        return Telemetries(tuple(result))
