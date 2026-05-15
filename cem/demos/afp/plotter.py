"""AFP IV plotter: telemetry, loss curves, and causal-pathway diagnostics."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import KW_ONLY
from typing import Any, override

import equinox as eqx
import jax
import jax.random as jr
import numpy as np
from jax import enable_x64
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from tjax import create_streams
from tjax.dataclasses import field

from cem.structure.plotter.plotter import Plotter
from cem.structure.plotter.with_smooth_graph import PlotterWithSmoothGraph, smooth_data
from cem.structure.solution import InferenceResults, Telemetries, TrainingResults
from cem.structure.solution.inference import Inference, InferenceResult, TrainingResult
from cem.structure.solution.telemetry import Telemetry
from cem.structure.solution.training_solution import TrainingSolution

from .problem import IVProblem
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


def _populated_solver(template: AFPSolver, training_results: TrainingResults) -> AFPSolver:
    """Recover the actual solver used during training from the trained model's shapes.

    ``template`` has problem-shape fields (n_instruments, n_candidate_confounders, …) but
    its tunable architecture fields (``endo_latent``, ``exo_latent``, ``n_frequencies``)
    are at their defaults — not whatever value Optuna picked for this trial.  The
    *partial* model in ``training_results.final_state.dis_learnable_parameters`` still
    carries the trained static fields, so we copy them across into the template.
    """
    partial = training_results.final_state.dis_learnable_parameters.assembled()
    assert isinstance(partial, AFPModel)
    derived_n_frequencies = int(partial.obs_features) // int(template.n_outcomes)
    return eqx.tree_at(
        lambda s: (s.endo_latent, s.exo_latent, s.n_frequencies),
        template,
        (int(partial.endo_latent), int(partial.exo_latent), derived_n_frequencies),
    )


def _trained_model(solver: AFPSolver, training_results: TrainingResults) -> AFPModel:
    populated = _populated_solver(solver, training_results)
    solution = populated.solution()
    learnable = training_results.final_state.dis_learnable_parameters.assembled()
    model = solution.inference.assemble_model(learnable)
    assert isinstance(model, AFPModel)
    return model


def _sample_eval_states(
    problem: IVProblem, *, n_eval: int, eval_seed: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sample test states; return (z, y, e) as numpy arrays for downstream use."""
    ds = problem.create_data_source()
    keys = jr.split(jr.key(eval_seed), n_eval)

    def gather(k: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array]:
        state = ds.initial_problem_state(k)
        return state.z, state.y, state.e

    zs, ys, es = jax.vmap(gather)(keys)
    return np.asarray(zs), np.asarray(ys), np.asarray(es)


def _decode_via_channel(
    problem: IVProblem,
    model: AFPModel,
    channel: str,
    *,
    n_eval: int,
    eval_seed: int,
    inference_seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (ŷ_channel, z, e) for ``n_eval`` test samples drawn from the data source.

    ``channel`` is ``"exo"`` or ``"endo"``.
    """
    ds = problem.create_data_source()
    streams = create_streams({"inference": jr.key(inference_seed)})
    keys = jr.split(jr.key(eval_seed), n_eval)

    if channel == "exo":
        predict_y = lambda obs: model.predict_y_exo(obs, streams=streams, inference=True)  # noqa: E731
    elif channel == "endo":
        predict_y = lambda obs: model.predict_y_endo(obs, streams=streams, inference=True)  # noqa: E731
    else:
        msg = f"Unknown channel {channel!r}"
        raise ValueError(msg)

    def per_sample(k: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array]:
        state = ds.initial_problem_state(k)
        obs = problem.extract_observation(state)
        return predict_y(obs), state.z, state.e

    y_hats, zs, es = jax.vmap(per_sample)(keys)
    return np.asarray(y_hats), np.asarray(zs), np.asarray(es)


def _scatter_with_yx_line(
    ax: Axes,
    xs: np.ndarray,
    ys: np.ndarray,
    *,
    xlabel: str,
    ylabel: str,
) -> tuple[float, float]:
    """Scatter (xs, ys), overlay y=x, return (slope_through_origin, R²)."""
    xs_flat = xs.reshape(-1)
    ys_flat = ys.reshape(-1)
    denom = float(np.dot(xs_flat, xs_flat))
    slope = float(np.dot(xs_flat, ys_flat) / denom) if denom > 0 else float("nan")
    r2 = (
        float(np.corrcoef(xs_flat, ys_flat)[0, 1] ** 2)
        if xs_flat.size > 1 and xs_flat.std() > 0 and ys_flat.std() > 0
        else float("nan")
    )
    ax.scatter(xs_flat, ys_flat, alpha=0.4, s=8)
    lo = float(min(xs_flat.min(), ys_flat.min()))
    hi = float(max(xs_flat.max(), ys_flat.max()))
    pad = 0.05 * (hi - lo + 1e-12)
    ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], color="black", linewidth=0.8, alpha=0.4)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    return slope, r2


class AFPCausalExoPlotter(Plotter):
    """Compare AFP's exogenous prediction to the structural causal contribution.

    On held-out samples, scatter ``ŷ_exo`` (decoded from the exogenous channel)
    against ``γα·Z`` (the part of Y caused by the instrument pathway).  Slope ≈ 1
    on the y=x line means AFP has correctly attributed the causal contribution to
    its exogenous channel.
    """

    solver: AFPSolver
    _: KW_ONLY
    name: str = field(static=True, default="afp-causal-exo")
    title: str = field(static=True, default="AFP Causal Pathway Recovery")
    n_eval: int = field(static=True, default=512)
    eval_seed: int = field(static=True, default=0)
    inference_seed: int = field(static=True, default=0)

    @override
    def plot(
        self,
        figure: Figure,
        training_results: TrainingResults,
        inference_results: InferenceResults,
        label: str,
    ) -> None:
        del inference_results, label
        with enable_x64():
            problem = self.solver.problem()
            model = _trained_model(self.solver, training_results)
            y_hats, zs, _ = _decode_via_channel(
                problem,
                model,
                "exo",
                n_eval=self.n_eval,
                eval_seed=self.eval_seed,
                inference_seed=self.inference_seed,
            )
            gamma_alpha = np.asarray(problem.exo_causal_weight)  # (n_outcomes, n_instruments)
            structural = zs @ gamma_alpha.T  # (n_eval, n_outcomes)

        ax = figure.gca() if figure.axes else figure.subplots(1, 1)
        slope, r2 = _scatter_with_yx_line(
            ax,
            structural,
            y_hats,
            xlabel="gamma @ alpha @ Z (structural causal contribution)",
            ylabel="y_hat_exo (AFP exogenous prediction)",
        )
        ax.set_title(f"y_hat_exo vs gamma*alpha*Z   slope={slope:.3f}, R2={r2:.3f}")


class AFPConfoundedEndoPlotter(Plotter):
    """Compare AFP's endogenous prediction to the confounded residual.

    On held-out samples, scatter ``ŷ_endo`` (decoded from the endogenous channel)
    against ``Y − γα·Z`` (the confounded U-driven residual that the endo channel
    should capture).  Slope ≈ 1 means AFP has correctly routed the confounded
    variation into its endogenous channel.
    """

    solver: AFPSolver
    _: KW_ONLY
    name: str = field(static=True, default="afp-confounded-endo")
    title: str = field(static=True, default="AFP Confounded Residual Recovery")
    n_eval: int = field(static=True, default=512)
    eval_seed: int = field(static=True, default=0)
    inference_seed: int = field(static=True, default=0)

    @override
    def plot(
        self,
        figure: Figure,
        training_results: TrainingResults,
        inference_results: InferenceResults,
        label: str,
    ) -> None:
        del inference_results, label
        with enable_x64():
            problem = self.solver.problem()
            model = _trained_model(self.solver, training_results)
            y_hats, zs, _ = _decode_via_channel(
                problem,
                model,
                "endo",
                n_eval=self.n_eval,
                eval_seed=self.eval_seed,
                inference_seed=self.inference_seed,
            )
            _, ys, _ = _sample_eval_states(problem, n_eval=self.n_eval, eval_seed=self.eval_seed)
            gamma_alpha = np.asarray(problem.exo_causal_weight)
            residual = ys - zs @ gamma_alpha.T  # (n_eval, n_outcomes)

        ax = figure.gca() if figure.axes else figure.subplots(1, 1)
        slope, r2 = _scatter_with_yx_line(
            ax,
            residual,
            y_hats,
            xlabel="Y - gamma*alpha*Z (confounded residual)",
            ylabel="y_hat_endo (AFP endogenous prediction)",
        )
        ax.set_title(f"y_hat_endo vs Y-gamma*alpha*Z   slope={slope:.3f}, R2={r2:.3f}")


class AFPCrossEnvPlotter(Plotter):
    """Visualize cross-environment invariance of the exogenous channel.

    On held-out samples, scatter ``ŷ_exo`` against ``γα·Z``, coloured by
    environment.  Overlapping point clouds across environments indicate that the
    AFP exogenous channel has learned the *invariant* causal relationship — its
    response to Z does not depend on the env-specific Z distribution shift.
    Cluster separation by colour indicates an env-specific shortcut.
    """

    solver: AFPSolver
    _: KW_ONLY
    name: str = field(static=True, default="afp-cross-env")
    title: str = field(static=True, default="AFP Cross-Environment Invariance")
    n_eval: int = field(static=True, default=1024)
    eval_seed: int = field(static=True, default=0)
    inference_seed: int = field(static=True, default=0)

    @override
    def plot(
        self,
        figure: Figure,
        training_results: TrainingResults,
        inference_results: InferenceResults,
        label: str,
    ) -> None:
        del inference_results, label
        with enable_x64():
            problem = self.solver.problem()
            model = _trained_model(self.solver, training_results)
            y_hats, zs, es = _decode_via_channel(
                problem,
                model,
                "exo",
                n_eval=self.n_eval,
                eval_seed=self.eval_seed,
                inference_seed=self.inference_seed,
            )
            gamma_alpha = np.asarray(problem.exo_causal_weight)
            structural = zs @ gamma_alpha.T  # (n_eval, n_outcomes)

        n_envs = int(problem.n_environments)
        ax = figure.gca() if figure.axes else figure.subplots(1, 1)
        env_indices = np.asarray(es).reshape(-1)
        # Flatten across outcomes by repeating env index per outcome.
        struct_flat = structural.reshape(-1)
        y_hat_flat = y_hats.reshape(-1)
        env_repeated = np.repeat(env_indices, structural.shape[1])
        # Use matplotlib's default colour cycle.
        slopes: list[tuple[int, float, float]] = []
        for env in range(n_envs):
            mask = env_repeated == env
            if not mask.any():
                continue
            x_env = struct_flat[mask]
            y_env = y_hat_flat[mask]
            ax.scatter(x_env, y_env, alpha=0.4, s=8, label=f"env={env}")
            denom = float(np.dot(x_env, x_env))
            slope = float(np.dot(x_env, y_env) / denom) if denom > 0 else float("nan")
            r2 = (
                float(np.corrcoef(x_env, y_env)[0, 1] ** 2)
                if x_env.size > 1 and x_env.std() > 0 and y_env.std() > 0
                else float("nan")
            )
            slopes.append((env, slope, r2))
        lo = float(min(struct_flat.min(), y_hat_flat.min()))
        hi = float(max(struct_flat.max(), y_hat_flat.max()))
        pad = 0.05 * (hi - lo + 1e-12)
        ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], color="black", linewidth=0.8, alpha=0.4)
        ax.set_xlabel("gamma @ alpha @ Z")
        ax.set_ylabel("y_hat_exo")
        slopes_str = ", ".join(f"env{env}: slope={s:.2f}" for env, s, _r2 in slopes)
        ax.set_title(f"y_hat_exo vs gamma*alpha*Z by environment    {slopes_str}")
        ax.legend()
