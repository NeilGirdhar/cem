"""AFP IV plotter: telemetry and loss curves for the AFP demo."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import KW_ONLY
from typing import Any, override

import jax.random as jr
import numpy as np
from jax import enable_x64
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from tjax.dataclasses import field

from cem.structure.plotter.plotter import Plotter
from cem.structure.plotter.with_smooth_graph import PlotterWithSmoothGraph, smooth_data
from cem.structure.solution import InferenceResults, Telemetries, TrainingResults
from cem.structure.solution.inference import Inference, InferenceResult, TrainingResult
from cem.structure.solution.telemetry import Telemetry
from cem.structure.solution.training_solution import TrainingSolution

from .readout import gamma_readout, gamma_recovery_error
from .solution import AFPConfiguration, AFPModel, AFPSolver


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
        # symlog so we can see small recon values clearly while exo/endo can still
        # cross zero (negative gain = critic worse than uniform).
        ax.set_yscale("symlog", linthresh=1e-3)
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
        del inference_results, label
        telemetry = AFPTelemetry(selected_node=self.selected_node)
        training_losses = training_results.telemetries[telemetry]
        ax = figure.subplots(1, 1)
        self._plot_axis(ax, training_losses, split="Training")


class AFPGammaRecoveryPlotter(Plotter):
    """Probes the trained model for ``γ̂`` and plots it against the true ``γ``.

    For each ``(label, training_results)`` pair, this plotter rebuilds the trained
    ``AFPModel`` from ``training_results.final_state`` together with the solver's
    fixed parameters, then runs :func:`gamma_readout` and renders a scatter of
    ``γ̂`` vs ``γ`` with the ``y = x`` reference line and the relative Frobenius
    error in the legend.
    """

    solver: AFPSolver
    _: KW_ONLY
    name: str = field(static=True, default="afp-gamma-recovery")
    title: str = field(static=True, default="AFP gamma Recovery")
    n_probes: int = field(static=True, default=256)
    readout_seed: int = field(static=True, default=0)

    def _trained_model(self, training_results: TrainingResults) -> AFPModel:
        solution = self.solver.solution()
        learnable = training_results.final_state.dis_learnable_parameters.assembled()
        model = solution.inference.assemble_model(learnable)
        assert isinstance(model, AFPModel)
        return model

    def plot(
        self,
        figure: Figure,
        training_results: TrainingResults,
        inference_results: InferenceResults,
        label: str,
    ) -> None:
        del inference_results
        # The model was trained inside an `enable_x64()` context, so its parameters are
        # complex128/float64.  Re-enter that context for the readout to keep dtypes aligned.
        with enable_x64():
            problem = self.solver.problem()
            model = self._trained_model(training_results)
            gamma_hat = gamma_readout(
                model, problem, key=jr.key(self.readout_seed), n_probes=self.n_probes
            )
            gamma_true = problem.gamma
            error = float(gamma_recovery_error(gamma_hat, gamma_true))

        ax = figure.gca() if figure.axes else figure.subplots(1, 1)
        true_flat = np.asarray(gamma_true).reshape(-1)
        hat_flat = np.asarray(gamma_hat).reshape(-1)
        scatter_label = f"{label}: rel. err = {error:.3f}" if label else f"rel. err = {error:.3f}"
        ax.scatter(true_flat, hat_flat, label=scatter_label, alpha=0.8)
        lo = float(min(true_flat.min(), hat_flat.min()))
        hi = float(max(true_flat.max(), hat_flat.max()))
        pad = 0.05 * (hi - lo + 1e-12)
        ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], color="black", linewidth=0.8, alpha=0.3)
        ax.set_xlabel("gamma (true)")
        ax.set_ylabel("gamma_hat (recovered)")
        ax.set_title("gamma_hat vs gamma")
        ax.legend()
