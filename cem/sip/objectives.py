"""Alternating objectives for real-valued SIP score circuits."""

from typing import Protocol

import jax.numpy as jnp
from jax.lax import stop_gradient
from tjax import JaxRealArray


class PurifiableOutput(Protocol):
    """The fields ScoreOutput and TDErrorOutput share, needed to purify either one."""

    observation_score: JaxRealArray
    reconstruction_loss: JaxRealArray
    witness: JaxRealArray


def _confounding_moment(
    residual: JaxRealArray,
    witness: JaxRealArray,
) -> JaxRealArray:
    if residual.ndim == 1:
        residual = residual[jnp.newaxis, :]
        witness = witness[jnp.newaxis, :]
    residual -= jnp.mean(residual, axis=0, keepdims=True)
    witness -= jnp.mean(witness, axis=0, keepdims=True)
    moment = jnp.mean(residual * witness, axis=0)
    return jnp.sum(jnp.square(moment))


def purification_loss(
    output: PurifiableOutput,
    *,
    confounding_weight: float = 1.0,
) -> JaxRealArray:
    """Train the predictor to reconstruct while removing instrument correlation."""
    if confounding_weight < 0.0:
        msg = "confounding_weight must be nonnegative"
        raise ValueError(msg)
    confounding = _confounding_moment(
        output.observation_score,
        stop_gradient(output.witness),
    )
    return jnp.mean(output.reconstruction_loss) + confounding_weight * confounding


def witness_loss(output: PurifiableOutput) -> JaxRealArray:
    """Train the witness to expose predictable residual structure."""
    confounding = _confounding_moment(
        stop_gradient(output.observation_score),
        output.witness,
    )
    return -confounding
