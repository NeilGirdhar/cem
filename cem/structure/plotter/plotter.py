from collections.abc import Mapping, Sequence
from dataclasses import KW_ONLY, dataclass
from typing import Any

import equinox as eqx

from cem.structure.solution import InferenceResults, Telemetries, TrainingResults

type PlottedSeries = dict[str, Any]


@dataclass(frozen=True)
class Run:
    """Results from one (variant, sweep entry) training-plus-inference execution.

    Attributes:
        variant_label: Label of the variant that produced this run (empty for a
            single-variant experiment).
        sweep_entry: The sweep parameter overrides used for this run; empty when
            the experiment has no sweep.
        training_results: Metrics recorded during training.
        inference_results: Metrics recorded during inference.
    """

    variant_label: str
    sweep_entry: Mapping[str, Any]
    training_results: TrainingResults
    inference_results: InferenceResults


class Plotter(eqx.Module):
    _: KW_ONLY
    name: str = eqx.field(static=True)
    title: str = eqx.field(static=True)

    def __check_init__(self) -> None:  # ruff:ignore[bad-dunder-method-name]
        if "_" in self.name:
            msg = f"Plotter.name must use hyphens, not underscores: {self.name!r}"
            raise ValueError(msg)

    def plotted_series(self, runs: Sequence[Run]) -> PlottedSeries:
        """Pivot a complete grid of :class:`Run` objects into one chart's JSON object."""
        raise NotImplementedError

    def telemetries(self) -> Telemetries:
        return Telemetries()
