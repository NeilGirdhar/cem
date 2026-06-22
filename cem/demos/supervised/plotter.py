"""Supervised learning plotter: training loss curves."""

from __future__ import annotations

from dataclasses import KW_ONLY
from typing import override

import numpy as np
from tjax.dataclasses import field

from cem.phasor.telemetry import SpectralLossTelemetry
from cem.structure.plotter.plotter import LinePlotTitles, PlottedSeries
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

    def _loss_series(self, losses: object) -> tuple[np.ndarray, np.ndarray]:
        loss_values = np.asarray(losses, dtype=np.float64)
        if loss_values.ndim > 1:
            loss_values = np.mean(loss_values, axis=tuple(range(1, loss_values.ndim)))
        times = np.arange(loss_values.shape[0], dtype=np.float64)
        return times, smooth_data(loss_values, self.smoothing, log_space=True)


class SupervisedTrainingLossPlotter(_SupervisedLossPlotter):
    """Plots training loss curves for the supervised demo."""

    _: KW_ONLY
    name: str = field(static=True, default="supervised-training-loss")
    title: str = field(static=True, default="Supervised Training Loss")

    @override
    def line_plot_titles(self, label: str) -> LinePlotTitles:
        prefix = {
            "perceptron": "Perceptron",
            "phasor": "Phasor",
        }.get(label, label.title())
        prefix = f"{prefix} " if prefix else ""
        return {
            "distributional_loss": f"{prefix}Distributional Loss",
            "spectral_loss": f"{prefix}Spectral Loss",
        }

    @override
    def plotted_series(
        self,
        training_results: TrainingResults,
        inference_results: InferenceResults,
        label: str,
    ) -> PlottedSeries:
        del inference_results, label
        telemetry = LossTelemetry(selected_node=self.selected_node)
        times, losses = self._loss_series(training_results.telemetries[telemetry])
        result: PlottedSeries = {
            "iteration": times.tolist(),
            "distributional_loss": losses.tolist(),
        }
        spectral_telemetry = SpectralLossTelemetry(selected_node=self.selected_node)
        if spectral_telemetry in training_results.telemetries:
            _, spectral_losses = self._loss_series(training_results.telemetries[spectral_telemetry])
            result["spectral_loss"] = spectral_losses.tolist()
        return result
