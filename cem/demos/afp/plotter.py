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
from tjax import create_streams
from tjax.dataclasses import field

from cem.structure.plotter.plotter import LinePlotTitles, PlottedSeries, Plotter
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

    @override
    def line_plot_titles(self, label: str) -> LinePlotTitles:
        del label
        return {
            "reconstruction_loss": "Reconstruction Loss",
            "exogeneity_loss": "Exogeneity Loss",
            "endogenous_separation_loss": "Endogenous Separation Loss",
        }

    def _mean_over_non_time_axes(self, values: object) -> np.ndarray:
        np_values = np.asarray(values, dtype=np.float64)
        if np_values.ndim == 0:
            return np_values[None]
        if np_values.ndim == 1:
            return np_values
        axes = tuple(range(1, np_values.ndim))
        return np.mean(np_values, axis=axes)

    def _series(self, losses: AFPConfiguration) -> dict[str, np.ndarray]:
        recon = self._mean_over_non_time_axes(losses.recon_loss)
        exo = self._mean_over_non_time_axes(losses.exo_loss)
        endo = self._mean_over_non_time_axes(losses.endo_loss)
        return {
            "iteration": np.arange(recon.shape[0], dtype=np.float64),
            "reconstruction_loss": smooth_data(recon, self.smoothing),
            "exogeneity_loss": smooth_data(exo, self.smoothing),
            "endogenous_separation_loss": smooth_data(endo, self.smoothing),
        }

    @override
    def plotted_series(
        self,
        training_results: TrainingResults,
        inference_results: InferenceResults,
        label: str,
    ) -> PlottedSeries:
        del inference_results, label
        telemetry = AFPTelemetry(selected_node=self.selected_node)
        training_losses = training_results.telemetries[telemetry]
        return {key: values.tolist() for key, values in self._series(training_losses).items()}


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
    def plotted_series(
        self,
        training_results: TrainingResults,
        inference_results: InferenceResults,
        label: str,
    ) -> PlottedSeries:
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
            gamma_alpha = np.asarray(problem.exo_causal_weight)
            structural = zs @ gamma_alpha.T
        y_hat_flat = y_hats.reshape(-1)
        structural_flat = structural.reshape(-1)
        return {
            "iteration": np.arange(structural_flat.shape[0], dtype=np.float64).tolist(),
            "structural_causal_contribution": structural_flat.tolist(),
            "y_hat_exo": y_hat_flat.tolist(),
        }


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
    def plotted_series(
        self,
        training_results: TrainingResults,
        inference_results: InferenceResults,
        label: str,
    ) -> PlottedSeries:
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
            residual = ys - zs @ gamma_alpha.T
        y_hat_flat = y_hats.reshape(-1)
        residual_flat = residual.reshape(-1)
        return {
            "iteration": np.arange(residual_flat.shape[0], dtype=np.float64).tolist(),
            "confounded_residual": residual_flat.tolist(),
            "y_hat_endo": y_hat_flat.tolist(),
        }


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
    def plotted_series(
        self,
        training_results: TrainingResults,
        inference_results: InferenceResults,
        label: str,
    ) -> PlottedSeries:
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
            structural = zs @ gamma_alpha.T
        structural_flat = structural.reshape(-1)
        y_hat_flat = y_hats.reshape(-1)
        env_repeated = np.repeat(np.asarray(es).reshape(-1), structural.shape[1])
        return {
            "iteration": np.arange(structural_flat.shape[0], dtype=np.float64).tolist(),
            "structural_causal_contribution": structural_flat.tolist(),
            "y_hat_exo": y_hat_flat.tolist(),
            "environment": env_repeated.astype(np.float64).tolist(),
        }
