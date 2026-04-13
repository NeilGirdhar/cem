"""Supervised learning plotter: training and inference loss curves."""

from __future__ import annotations

from dataclasses import KW_ONLY
from typing import override

import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from tjax.dataclasses import field

from cem.structure.plotter.with_smooth_graph import PlotterWithSmoothGraph, smooth_data
from cem.structure.solution import InferenceResults, Telemetries, TrainingResults
from cem.structure.solution.loss_telemetry import LossTelemetry


class SupervisedLossPlotter(PlotterWithSmoothGraph):
    """Plots training and inference loss curves for the supervised demo."""

    _: KW_ONLY
    name: str = field(static=True, default="supervised_loss")
    title: str = field(static=True, default="Supervised Loss")
    selected_node: str = field(static=True, default="target")

    @override
    def telemetries(self) -> Telemetries:
        return Telemetries((LossTelemetry(selected_node=self.selected_node),))

    def _plot_axis(self, ax: Axes, losses: np.ndarray, *, split: str) -> None:
        if losses.ndim > 1:
            losses = np.mean(losses, axis=tuple(range(1, losses.ndim)))
        times = np.arange(losses.shape[0], dtype=np.float64)
        ax.plot(times, smooth_data(losses, self.smoothing), label="Loss")
        ax.set_title(split)
        ax.set_xlabel("Episode")
        ax.set_ylabel("Loss")
        ax.legend()

    @override
    def plot(
        self,
        figure: Figure,
        training_results: TrainingResults,
        inference_results: InferenceResults,
        solver_index: int,
    ) -> None:
        del solver_index
        telemetry = LossTelemetry(selected_node=self.selected_node)
        training_losses = np.asarray(training_results.telemetries[telemetry], dtype=np.float64)
        inference_losses = np.asarray(inference_results.telemetries[telemetry], dtype=np.float64)
        axes = figure.subplots(1, 2, squeeze=False)
        self._plot_axis(axes[0, 0], training_losses, split="Training")
        self._plot_axis(axes[0, 1], inference_losses, split="Inference")
