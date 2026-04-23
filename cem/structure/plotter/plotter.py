from __future__ import annotations

from dataclasses import KW_ONLY

import equinox as eqx
from matplotlib.figure import Figure

from cem.structure.solution import InferenceResults, Telemetries, TrainingResults


class Plotter(eqx.Module):
    _: KW_ONLY
    name: str = eqx.field(static=True)
    title: str = eqx.field(static=True)

    def __check_init__(self) -> None:  # noqa: PLW3201
        if "_" in self.name:
            msg = f"Plotter.name must use hyphens, not underscores: {self.name!r}"
            raise ValueError(msg)

    def plot(
        self,
        figure: Figure,
        training_results: TrainingResults,
        inference_results: InferenceResults,
        label: str,
    ) -> None:
        raise NotImplementedError

    def telemetries(self) -> Telemetries:
        return Telemetries()
