"""Supervised learning plotter: training loss curves."""

from collections.abc import Sequence
from dataclasses import KW_ONLY
from typing import override

import numpy as np
from tjax.dataclasses import field

from cem.structure.plotter.plotter import PlottedSeries, Run
from cem.structure.plotter.with_smooth_graph import PlotterWithSmoothGraph, smooth_data
from cem.structure.solution import Telemetries
from cem.structure.solution.loss_telemetry import LossTelemetry

_VARIANT_TITLE = {
    "perceptron": "Perceptron",
    "natural_parameter": "Gaussian Natural-Parameter Network",
}


class _SupervisedLossPlotter(PlotterWithSmoothGraph):
    _: KW_ONLY
    selected_node: str = field(static=True, default="target")

    @override
    def telemetries(self) -> Telemetries:
        return Telemetries((LossTelemetry(selected_node=self.selected_node),))

    def _loss_series(self, losses: object) -> tuple[np.ndarray, np.ndarray]:
        loss_values = np.asarray(losses, dtype=np.float64)
        if loss_values.ndim > 1:
            loss_values = np.mean(loss_values, axis=tuple(range(1, loss_values.ndim)))
        times = np.arange(loss_values.shape[0], dtype=np.float64)
        return times, smooth_data(loss_values, self.smoothing, log_space=True)


class SupervisedTrainingLossPlotter(_SupervisedLossPlotter):
    """Plot one training loss curve per variant for the supervised experiment."""

    _: KW_ONLY
    name: str = field(static=True, default="supervised-training-loss")
    title: str = field(static=True, default="Supervised Training Loss")

    @override
    def plotted_series(self, runs: Sequence[Run]) -> PlottedSeries:
        telemetry = LossTelemetry(selected_node=self.selected_node)
        result: PlottedSeries = {}
        line_plots: dict[str, str] = {}
        for run in runs:
            label = run.variant_label
            times, losses = self._loss_series(run.training_results.telemetries[telemetry])
            result["iteration"] = times.tolist()
            prefix_title = _VARIANT_TITLE.get(label, label.title() if label else "")
            key = f"{label}.distributional_loss" if label else "distributional_loss"
            title = f"{prefix_title} Distributional Loss" if prefix_title else "Distributional Loss"
            result[key] = losses.tolist()
            line_plots[key] = title
        result["line plots"] = line_plots
        return result
