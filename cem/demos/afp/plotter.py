"""AFP IV plotter: telemetry and loss curves for the AFP demo."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import KW_ONLY
from typing import Any, override

import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from tjax.dataclasses import field

from cem.structure.plotter.with_smooth_graph import PlotterWithSmoothGraph, smooth_data
from cem.structure.solution import InferenceResults, Telemetries, TrainingResults
from cem.structure.solution.inference import Inference, InferenceResult, TrainingResult
from cem.structure.solution.telemetry import Telemetry
from cem.structure.solution.training_solution import TrainingSolution

from .solution import AFPConfiguration


class AFPTelemetry(Telemetry):
    """Telemetry for AFP-specific loss terms on a selected node."""

    selected_node: str = field(static=True)

    def _extract(self, configuration: object) -> AFPConfiguration:
        assert isinstance(configuration, AFPConfiguration)
        return configuration

    @override
    def training_snapshot(
        self,
        training_solution: TrainingSolution,
        training_result: TrainingResult,
        snapshots: Mapping[Telemetry, Any],
    ) -> AFPConfiguration | None:
        config = training_result.inference_result.model_configuration.get(self.selected_node)
        return None if config is None else self._extract(config)

    @override
    def inference_snapshot(
        self,
        inference: Inference,
        inference_result: InferenceResult,
        snapshots: Mapping[Telemetry, Any],
    ) -> AFPConfiguration | None:
        config = inference_result.model_configuration.get(self.selected_node)
        return None if config is None else self._extract(config)


class AFPLossPlotter(PlotterWithSmoothGraph):
    """Plots AFP diagnostics over training/inference."""

    _: KW_ONLY
    name: str = field(static=True, default="afp-losses")
    title: str = field(static=True, default="AFP Losses")
    selected_node: str = field(static=True, default="afp")

    def telemetries(self) -> Telemetries:
        return Telemetries((AFPTelemetry(selected_node=self.selected_node),))

    def _mean_over_non_time_axes(self, values: object) -> np.ndarray:
        np_values = np.asarray(values, dtype=np.float64)
        if np_values.ndim == 0:
            return np_values[None]
        if np_values.ndim == 1:
            return np_values
        axes = tuple(range(1, np_values.ndim))
        return np.mean(np_values, axis=axes)

    def _plot_axis(self, ax: Axes, losses: AFPConfiguration, *, split: str) -> None:
        series = (
            ("Reconstruction", self._mean_over_non_time_axes(losses.recon_loss)),
            ("Exogeneity", self._mean_over_non_time_axes(losses.exo_loss)),
            ("Endogenous Separation", self._mean_over_non_time_axes(losses.endo_loss)),
        )
        for label, values in series:
            times = np.arange(values.shape[0], dtype=np.float64)
            ax.plot(times, smooth_data(values, self.smoothing), label=label)
        ax.axhline(0.0, color="black", linewidth=0.8, alpha=0.3)
        ax.set_title(split)
        ax.set_xlabel("Episode")
        ax.set_ylabel("Loss")
        ax.legend()

    def plot(
        self,
        figure: Figure,
        training_results: TrainingResults,
        inference_results: InferenceResults,
        label: str,
    ) -> None:
        del label
        telemetry = AFPTelemetry(selected_node=self.selected_node)
        training_losses = training_results.telemetries[telemetry]
        inference_losses = inference_results.telemetries[telemetry]
        axes = figure.subplots(1, 2, squeeze=False)
        self._plot_axis(axes[0, 0], training_losses, split="Training")
        self._plot_axis(axes[0, 1], inference_losses, split="Inference")
