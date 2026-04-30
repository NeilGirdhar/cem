from __future__ import annotations

import jax.numpy as jnp

from cem.demos.afp.plotter import AFPTelemetry
from cem.demos.afp.solution import AFPModel, AFPSolver
from cem.structure.solution import ExecutionPacket, Telemetries

_JOINT_OBSERVED_FEATURES = 2
_DEFAULT_FREQUENCIES = 10


def test_afp_purifiers_receive_joint_observed_inputs() -> None:
    solution = AFPSolver().solution()
    model = solution.assemble_model(fixed_parameters=True, learnable_parameters=True)

    assert isinstance(model, AFPModel)
    encoded_features = _JOINT_OBSERVED_FEATURES * _DEFAULT_FREQUENCIES
    assert model.exo_purifier.f1.weight.value.shape[1] == encoded_features
    assert model.exo_purifier.f2.weight.value.shape[1] == encoded_features
    assert model.endo_purifier.f1.weight.value.shape[1] == encoded_features
    assert model.endo_purifier.f2.weight.value.shape[1] == encoded_features


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
