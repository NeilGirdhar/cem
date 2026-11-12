"""The run records and plotter interface shared by experiment charts."""

from collections.abc import Mapping, Sequence
from dataclasses import KW_ONLY, dataclass
from typing import Any

import equinox as eqx

from cem.framework.solution import InferenceResults, Telemetries, TrainingResults

type PlottedSeries = dict[str, Any]


@dataclass(frozen=True)
class Run:
    """Training and inference results for one solver variant and one sweep entry.

    A single-variant experiment can use an empty label. An empty sweep entry
    means the solver runs without sweep overrides.
    """

    variant_label: str
    sweep_entry: Mapping[str, Any]
    training_results: TrainingResults
    inference_results: InferenceResults


class Plotter(eqx.Module):
    """A chart definition that selects telemetry and turns experiment runs into series."""

    _: KW_ONLY
    name: str = eqx.field(static=True)
    title: str = eqx.field(static=True)

    def __check_init__(self) -> None:  # ruff:ignore[bad-dunder-method-name]
        if "_" in self.name:
            msg = f"Plotter.name must use hyphens, not underscores: {self.name!r}"
            raise ValueError(msg)

    def plotted_series(self, runs: Sequence[Run]) -> PlottedSeries:
        """Convert the experiment's grid of runs into one chart's JSON data."""
        raise NotImplementedError

    def telemetries(self) -> Telemetries:
        """Select the telemetry this chart needs. The default requests none."""
        return Telemetries()
