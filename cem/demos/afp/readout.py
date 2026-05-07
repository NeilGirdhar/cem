"""γ-readout: extract a recovered causal-effect estimate from a trained AFPModel.

The AFP model never directly emits ``γ̂``.  This module probes the trained model's
mapping ``T → Ŷ`` (with ``Z`` and the env held fixed at sampled baselines) by
reverse-mode autodiff, decoding the predicted phasor ``z_hat`` back to ``Ŷ`` via
the characteristic function of ``UnitVarianceNormalNP``.  The recovered Jacobian
is averaged over ``n_probes`` baselines drawn from the data source, yielding
``γ̂`` of shape ``(n_outcomes, n_treatments)``.

By default the readout uses the *exogenous* channel only (``predict_exo_phasor``):
its Jacobian targets the structural causal effect γ, while the total prediction's
Jacobian carries the confounder bias.  ``channel="endo"`` and ``channel="total"``
are exposed for diagnostics.
"""

from __future__ import annotations

from typing import Literal

import jax
import jax.numpy as jnp
import jax.random as jr
from tjax import JaxRealArray, KeyArray, create_streams

from cem.demos.afp.problem import IVObservation, IVProblem
from cem.demos.afp.solution import AFPModel
from cem.transforms import encode_flat

Channel = Literal["exo", "endo", "total"]


def _decode_phasor_to_y(
    z_hat_raveled: JaxRealArray,
    n_outcomes: int,
    frequencies: JaxRealArray,
) -> JaxRealArray:
    """Recover the mean of ``UnitVarianceNormalNP(μ)`` from its phasor encoding.

    For each outcome ``k`` and frequency ``ω_j`` the phasor is
    ``z[k, j] = exp(i ω_j μ_k − ω_j² / 2)``, so ``arg(z[k, j]) = ω_j μ_k`` modulo
    ``2π``.  A multi-frequency least-squares fit gives
    ``μ_k = (Σ_j ω_j arg z[k, j]) / Σ_j ω_j²``.
    """
    z_hat = z_hat_raveled.reshape(n_outcomes, frequencies.shape[0])
    arg_z = jnp.angle(z_hat)
    return jnp.sum(arg_z * frequencies[None, :], axis=-1) / jnp.sum(frequencies**2)


def _build_observation(
    z: JaxRealArray,
    t: JaxRealArray,
    e: JaxRealArray,
    problem: IVProblem,
) -> IVObservation:
    if problem.env_features > 0:
        env_one_hot = jax.nn.one_hot(e, problem.n_environments) - 1.0 / problem.n_environments
        x_raw = jnp.concatenate((z, t, env_one_hot))
    else:
        x_raw = jnp.concatenate((z, t))
    return IVObservation(
        x=encode_flat(x_raw),
        y=encode_flat(jnp.zeros(problem.obs_y_features, dtype=t.dtype)),
    )


def gamma_readout(
    model: AFPModel,
    problem: IVProblem,
    *,
    key: KeyArray,
    n_probes: int = 256,
    channel: Channel = "exo",
) -> JaxRealArray:
    """Recover ``γ̂`` from a trained :class:`AFPModel`.

    For each of ``n_probes`` baselines ``(Z, T_base, e)`` drawn from the data source,
    holding ``Z`` and ``e`` fixed and treating ``T`` as the variable, computes the
    Jacobian ``∂Ŷ/∂T`` at ``T = T_base`` by reverse-mode autodiff through the
    selected channel and the phasor decoder.  Averages across baselines.

    Args:
        model: A trained ``AFPModel``.
        problem: The corresponding ``IVProblem`` (provides shapes and the data source).
        key: RNG key for sampling baselines.
        n_probes: Number of baseline samples to average over.
        channel: Which AFP channel to differentiate.  ``"exo"`` (default) targets the
            structural causal effect γ; ``"endo"`` returns the confounder-bias term;
            ``"total"`` returns the sum, i.e. the conditional response ``∂E[Y|T,Z,e]/∂T``.

    Returns:
        ``γ̂`` of shape ``(n_outcomes, n_treatments)``, in raw ``Y`` space.
    """
    data_source = problem.create_data_source()
    streams = create_streams({"inference": jr.key(0)})
    frequencies = model._frequencies.value  # noqa: SLF001

    if channel == "exo":
        phasor_fn = model.predict_exo_phasor
    elif channel == "endo":
        phasor_fn = model.predict_endo_phasor
    else:
        phasor_fn = model.predict_phasor

    def predict_y(t: JaxRealArray, z: JaxRealArray, e: JaxRealArray) -> JaxRealArray:
        observation = _build_observation(z, t, e, problem)
        z_hat = phasor_fn(observation, streams=streams, inference=True)
        return _decode_phasor_to_y(z_hat, problem.n_outcomes, frequencies)

    def per_sample_jacobian(probe_key: KeyArray) -> JaxRealArray:
        state = data_source.initial_problem_state(probe_key)
        return jax.jacrev(predict_y)(state.t, state.z, state.e)

    probe_keys = jr.split(key, n_probes)
    jacobians = jax.vmap(per_sample_jacobian)(probe_keys)
    return jnp.mean(jacobians, axis=0)


def gamma_recovery_error(
    gamma_hat: JaxRealArray,
    gamma_true: JaxRealArray,
) -> JaxRealArray:
    """Relative Frobenius error ``‖γ̂ − γ‖ / ‖γ‖``."""
    return jnp.linalg.norm(gamma_hat - gamma_true) / jnp.linalg.norm(gamma_true)
