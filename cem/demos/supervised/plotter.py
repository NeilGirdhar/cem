"""Supervised learning plotter: training loss curves."""

from __future__ import annotations

from dataclasses import KW_ONLY
from typing import override

import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from tjax.dataclasses import field

from cem.phasor.telemetry import SpectralLossTelemetry
from cem.structure.plotter.with_smooth_graph import PlotterWithSmoothGraph, smooth_data
from cem.structure.solution import InferenceResults, Telemetries, TrainingResults
from cem.structure.solution.loss_telemetry import LossTelemetry


class _SupervisedLossPlotter(PlotterWithSmoothGraph):
    _: KW_ONLY
    selected_node: str = field(static=True, default="target")

    @override
    def telemetries(self) -> Telemetries:
        return Telemetries(
            (
                LossTelemetry(selected_node=self.selected_node),
                SpectralLossTelemetry(selected_node=self.selected_node),
            )
        )

    def _plot_axis(self, ax: Axes, losses: np.ndarray, *, label: str) -> None:
        if losses.ndim > 1:
            losses = np.mean(losses, axis=tuple(range(1, losses.ndim)))
        times = np.arange(losses.shape[0], dtype=np.float64)
        ax.plot(times, smooth_data(losses, self.smoothing, log_space=True), label=label or "Loss")
        ax.set_xlabel("Episode")
        ax.set_ylabel("Loss")
        ax.set_yscale("log")
        ax.legend(title="Variant / Loss")

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
        spectral_telemetry = SpectralLossTelemetry(selected_node=self.selected_node)
        has_spectral = spectral_telemetry in training_results.telemetries
        ax = self._get_or_create_ax(figure, self.title)
        telemetry = LossTelemetry(selected_node=self.selected_node)
        losses = np.asarray(training_results.telemetries[telemetry], dtype=np.float64)
        variant_label = label.capitalize()
        dist_label = f"{variant_label} / Distributional"
        self._plot_axis(ax, losses, label=dist_label)
        if has_spectral:
            spectral_losses = np.asarray(
                training_results.telemetries[spectral_telemetry], dtype=np.float64
            )
            self._plot_axis(ax, spectral_losses, label=f"{variant_label} / Spectral")
