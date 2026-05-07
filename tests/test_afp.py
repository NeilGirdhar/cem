from __future__ import annotations

import jax.numpy as jnp
import jax.random as jr

from cem.demos.afp.plotter import AFPTelemetry
from cem.demos.afp.problem import NonlinearityKind
from cem.demos.afp.readout import gamma_readout, gamma_recovery_error
from cem.demos.afp.solution import AFPModel, AFPSolver
from cem.structure.solution import ExecutionPacket, Telemetries


def _trained_solver_and_results(solver: AFPSolver) -> tuple[AFPSolver, object]:
    telemetry = AFPTelemetry(selected_node="afp")
    packet = ExecutionPacket(telemetries=Telemetries((telemetry,)))
    training_results, _ = solver.training_and_inference_result(packet=packet)
    return solver, training_results


def test_gamma_readout_shape_matches_true_gamma() -> None:
    solver = AFPSolver(n_instruments=4, n_confounders=3, n_treatments=3, n_outcomes=1)
    solution = solver.solution()
    learnable = solution.solution_state.dis_learnable_parameters.assembled()
    model = solution.inference.assemble_model(learnable)
    assert isinstance(model, AFPModel)
    gamma_hat = gamma_readout(model, solver.problem(), key=jr.key(0), n_probes=8)
    assert gamma_hat.shape == solver.problem().gamma.shape


def test_gamma_readout_finite_on_untrained_model() -> None:
    solver = AFPSolver(n_instruments=2, n_confounders=2, n_treatments=2, n_outcomes=2)
    solution = solver.solution()
    learnable = solution.solution_state.dis_learnable_parameters.assembled()
    model = solution.inference.assemble_model(learnable)
    assert isinstance(model, AFPModel)
    gamma_hat = gamma_readout(model, solver.problem(), key=jr.key(1), n_probes=4)
    assert jnp.all(jnp.isfinite(gamma_hat))
    error = gamma_recovery_error(gamma_hat, solver.problem().gamma)
    assert jnp.isfinite(error)


_DEFAULT_FREQUENCIES = 10


def test_iv_problem_default_shapes() -> None:
    solver = AFPSolver()
    problem = solver.problem()
    assert problem.obs_x_features == solver.n_instruments + solver.n_treatments
    assert problem.obs_y_features == solver.n_outcomes


def test_iv_problem_multienv_shape_includes_env() -> None:
    solver = AFPSolver(n_environments=3)
    problem = solver.problem()
    assert problem.obs_x_features == solver.n_instruments + solver.n_treatments + 3


def test_iv_data_source_produces_correct_shapes() -> None:
    solver = AFPSolver(n_instruments=3, n_treatments=2, n_outcomes=2, n_confounders=2)
    state = solver.problem().create_data_source().initial_problem_state(jr.key(0))
    assert state.z.shape == (3,)
    assert state.t.shape == (2,)
    assert state.y.shape == (2,)
    assert state.u.shape == (2,)


def test_iv_extract_observation_concatenates_z_t_env() -> None:
    solver = AFPSolver(n_instruments=2, n_treatments=1, n_environments=3)
    problem = solver.problem()
    state = problem.create_data_source().initial_problem_state(jr.key(1))
    obs = problem.extract_observation(state)
    assert obs.x.shape == (problem.obs_x_features,)
    assert obs.y.shape == (problem.obs_y_features,)


def test_iv_nonlinearity_changes_outcome() -> None:
    linear = AFPSolver(coefficient_seed=7, nonlinearity=NonlinearityKind.none).problem()
    nonlinear = AFPSolver(coefficient_seed=7, nonlinearity=NonlinearityKind.tanh).problem()
    state_lin = linear.create_data_source().initial_problem_state(jr.key(2))
    state_nl = nonlinear.create_data_source().initial_problem_state(jr.key(2))
    assert not jnp.allclose(state_lin.y, state_nl.y)


def test_afp_purifiers_size_to_problem() -> None:
    solver = AFPSolver(n_instruments=4, n_confounders=3, n_treatments=3, n_outcomes=1)
    solution = solver.solution()
    model = solution.assemble_model(fixed_parameters=True, learnable_parameters=True)
    assert isinstance(model, AFPModel)
    expected = (4 + 3) * solver.n_frequencies
    assert model.exo_purifier.f1.weight.value.shape[1] == expected
    assert model.exo_purifier.f2.weight.value.shape[1] == expected
    assert model.endo_purifier.f1.weight.value.shape[1] == expected
    assert model.endo_purifier.f2.weight.value.shape[1] == expected


def test_afp_short_training_records_finite_losses() -> None:
    solver = AFPSolver(
        training_examples=2,
        training_batch_size=4,
        inference_examples=2,
        inference_batch_size=4,
    )
    telemetry = AFPTelemetry(selected_node="afp")
    packet = ExecutionPacket(telemetries=Telemetries((telemetry,)))

    training_results, inference_results = solver.training_and_inference_result(packet=packet)
    config = inference_results.telemetries[telemetry]

    assert training_results.count == solver.training_examples
    assert inference_results.count == solver.inference_examples
    assert jnp.all(jnp.isfinite(config.recon_loss))
    assert jnp.all(jnp.isfinite(config.exo_loss))
    assert jnp.all(jnp.isfinite(config.endo_loss))
