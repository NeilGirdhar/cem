"""Alternating objectives for real-valued SIP score circuits."""

import jax.numpy as jnp
from jax.lax import stop_gradient
from tjax import JaxRealArray

from cem.sip.score import ScoreOutput


def purification_loss(
    output: ScoreOutput,
    *,
    confounding_weight: float = 1.0,
) -> JaxRealArray:
    """Train the predictor to reconstruct while removing instrument correlation."""
    if confounding_weight < 0.0:
        msg = "confounding_weight must be nonnegative"
        raise ValueError(msg)
    residual = output.observation_score
    confounding = jnp.square(jnp.sum(residual * stop_gradient(output.witness), axis=-1))
    return jnp.mean(output.reconstruction_loss + confounding_weight * confounding)


def witness_loss(output: ScoreOutput) -> JaxRealArray:
    """Train the witness to expose predictable residual structure."""
    residual = stop_gradient(output.observation_score)
    confounding = jnp.square(jnp.sum(residual * output.witness, axis=-1))
    return -jnp.mean(confounding)
