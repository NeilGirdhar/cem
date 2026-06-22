from __future__ import annotations

from dataclasses import KW_ONLY

import equinox as eqx

from cem.structure.solution import InferenceResults, Telemetries, TrainingResults

type PlottedSeries = dict[str, list[float]]


class Plotter(eqx.Module):
    _: KW_ONLY
    name: str = eqx.field(static=True)
    title: str = eqx.field(static=True)

    def __check_init__(self) -> None:  # noqa: PLW3201
        if "_" in self.name:
            msg = f"Plotter.name must use hyphens, not underscores: {self.name!r}"
            raise ValueError(msg)

    def plotted_series(
        self,
        training_results: TrainingResults,
        inference_results: InferenceResults,
        label: str,
    ) -> PlottedSeries:
        raise NotImplementedError

    def telemetries(self) -> Telemetries:
        return Telemetries()
