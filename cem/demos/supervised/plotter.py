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


class _SupervisedLossPlotter(PlotterWithSmoothGraph):
    _: KW_ONLY
    selected_node: str = field(static=True, default="target")

    @override
    def telemetries(self) -> Telemetries:
        return Telemetries((LossTelemetry(selected_node=self.selected_node),))

    def _plot_axis(self, ax: Axes, losses: np.ndarray, *, label: str) -> None:
        if losses.ndim > 1:
            losses = np.mean(losses, axis=tuple(range(1, losses.ndim)))
        times = np.arange(losses.shape[0], dtype=np.float64)
        ax.plot(times, smooth_data(losses, self.smoothing), label=label or "Loss")
        ax.set_xlabel("Episode")
        ax.set_ylabel("Loss")
        ax.legend()

    def _get_or_create_ax(self, figure: Figure, title: str) -> Axes:
        if axes := figure.get_axes():
            return axes[0]
        ax = figure.add_subplot()
        ax.set_title(title)
        return ax


class SupervisedTrainingLossPlotter(_SupervisedLossPlotter):
    """Plots training loss curves for the supervised demo."""

    _: KW_ONLY
    name: str = field(static=True, default="supervised-training-loss")
    title: str = field(static=True, default="Supervised Training Loss")

    @override
    def plot(
        self,
        figure: Figure,
        training_results: TrainingResults,
        inference_results: InferenceResults,
        label: str,
    ) -> None:
        telemetry = LossTelemetry(selected_node=self.selected_node)
        losses = np.asarray(training_results.telemetries[telemetry], dtype=np.float64)
        ax = self._get_or_create_ax(figure, "Training")
        self._plot_axis(ax, losses, label=label)


class SupervisedInferenceLossPlotter(_SupervisedLossPlotter):
    """Plots inference loss curves for the supervised demo."""

    _: KW_ONLY
    name: str = field(static=True, default="supervised-inference-loss")
    title: str = field(static=True, default="Supervised Inference Loss")

    @override
    def plot(
        self,
        figure: Figure,
        training_results: TrainingResults,
        inference_results: InferenceResults,
        label: str,
    ) -> None:
        telemetry = LossTelemetry(selected_node=self.selected_node)
        losses = np.asarray(inference_results.telemetries[telemetry], dtype=np.float64)
        ax = self._get_or_create_ax(figure, "Inference")
        self._plot_axis(ax, losses, label=label)
