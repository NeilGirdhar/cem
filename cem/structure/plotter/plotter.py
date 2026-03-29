from __future__ import annotations

from dataclasses import KW_ONLY

import equinox as eqx
from matplotlib.figure import Figure

from cem.structure.solution import InferenceResults, Telemetries, TrainingResults


class Plotter(eqx.Module):
    _: KW_ONLY
    name: str = eqx.field(static=True)
    title: str = eqx.field(static=True)

    def plot(
        self,
        figure: Figure,
        training_results: TrainingResults,
        inference_results: InferenceResults,
        solver_index: int,
    ) -> None:
        raise NotImplementedError

    def telemetries(self) -> Telemetries:
        return Telemetries()
