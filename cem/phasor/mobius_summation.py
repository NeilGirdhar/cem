from collections.abc import Mapping
from typing import Self

import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
from jax.nn import sigmoid
from tjax import JaxRealArray, RngStream

from cem.phasor.message import JaxComplexArray
from cem.structure.graph import LearnableParameter


def mobius_sum(
    x: JaxComplexArray,
    stretches: JaxRealArray,
    participations: JaxRealArray,
) -> JaxComplexArray:
    """Construct phasor features by softly summing Möbius-warped input phases.

    Args:
        x: Input phasors, shape (..., in_features).
        stretches: Möbius stretches in (-1, 1), shape (out_features, in_features).
        participations: Soft participations in [0, 1], shape (out_features, in_features).

    Returns:
        Constructed phasors, shape (..., out_features).
    """
    parameter_shape = (1,) * (x.ndim - 1) + stretches.shape
    stretches = jnp.reshape(stretches, parameter_shape)
    participations = jnp.reshape(participations, parameter_shape)

    presence = jnp.abs(x)[..., jnp.newaxis, :]
    unit = x[..., jnp.newaxis, :] / jnp.where(presence > 0, presence, 1)

    warped = (unit + stretches) / (1 + stretches * unit)
    contribution = (1 - participations) + participations * warped

    total_participation = 1 - jnp.prod(1 - participations, axis=-1)
    safe_presence = jnp.where(presence > 0, presence, 1)
    inverse_presence = participations / safe_presence
    inverse_presence = jnp.where(
        (participations > 0) & (presence == 0),
        jnp.inf,
        inverse_presence,
    )
    inverse_presence = jnp.where(participations > 0, inverse_presence, 0)
    denominator = jnp.sum(inverse_presence, axis=-1)
    safe_denominator = jnp.where(denominator > 0, denominator, 1)
    base_presence = jnp.where(
        total_participation > 0,
        total_participation**2 / safe_denominator,
        0,
    )
    return base_presence * jnp.prod(contribution, axis=-1)


class MobiusSummation(eqx.Module):
    """Bank of softly participating Möbius phase summations.

    Each output feature has one real stretch and one soft participation per input.
    Unconstrained learned stretches are mapped into (-1, 1); learned participation
    logits are mapped into (0, 1).

    Attributes:
        raw_stretches: Unconstrained real stretches, shape (out_features, in_features).
        participation_logits: Unconstrained participation logits, same shape.
    """

    raw_stretches: LearnableParameter[JaxRealArray]
    participation_logits: LearnableParameter[JaxRealArray]

    @classmethod
    def create(
        cls,
        in_features: int,
        out_features: int,
        *,
        streams: Mapping[str, RngStream],
    ) -> Self:
        shape = (out_features, in_features)
        stream = streams["parameters"]
        scale = 1 / jnp.sqrt(in_features)
        initial_logit = -jnp.log(in_features)
        return cls(
            raw_stretches=LearnableParameter(
                scale * jr.normal(stream.key(), shape, dtype=jnp.float64)
            ),
            participation_logits=LearnableParameter(
                initial_logit + scale * jr.normal(stream.key(), shape, dtype=jnp.float64)
            ),
        )

    def sum(self, x: JaxComplexArray) -> JaxComplexArray:
        """Apply the learned Möbius summation bank."""
        raw_stretches = self.raw_stretches.value
        stretches = raw_stretches / jnp.sqrt(1 + raw_stretches**2)
        participations = sigmoid(self.participation_logits.value)
        return mobius_sum(x, stretches, participations)
