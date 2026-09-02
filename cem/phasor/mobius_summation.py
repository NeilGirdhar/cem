from collections.abc import Mapping
from typing import Self

import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
from jax.nn import sigmoid
from tjax import JaxRealArray, RngStream

from cem.phasor.message import JaxComplexArray
from cem.structure.graph import LearnableParameter


def phase_warp(u: JaxComplexArray, weights: JaxRealArray) -> JaxComplexArray:
    """Multiply unit phasor values by real weights in the Cayley coordinate.

    The exceptional pair ``u == -1`` and ``weight == 0`` is assigned the
    collapsed value ``+1``. This makes zero weight a total, noninvertible map.
    """
    numerator = (1 + weights) * u + (1 - weights)
    denominator = (1 - weights) * u + (1 + weights)
    defined = denominator != 0
    safe_denominator = jnp.where(defined, denominator, 1)
    return jnp.where(defined, numerator / safe_denominator, jnp.ones_like(numerator))


def mobius_sum(
    x: JaxComplexArray,
    weights: JaxRealArray,
    participations: JaxRealArray,
) -> JaxComplexArray:
    """Construct phasor features with signed value weights and soft participation.

    Args:
        x: Input phasors, shape (..., in_features).
        weights: Signed Cayley-coordinate weights, shape
            (out_features, in_features).
        participations: Participation probabilities, same shape.

    Returns:
        Constructed phasors, shape (..., out_features).
    """
    parameter_shape = (1,) * (x.ndim - 1) + weights.shape
    weights = jnp.reshape(weights, parameter_shape)
    participations = jnp.reshape(participations, parameter_shape)

    presence = jnp.abs(x)[..., jnp.newaxis, :]
    unit = x[..., jnp.newaxis, :] / jnp.where(presence > 0, presence, 1)
    warped = phase_warp(unit, weights)
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


def _signed_unit_initialization(
    shape: tuple[int, ...],
    *,
    stream: RngStream,
) -> JaxRealArray:
    signs = jnp.where(jr.bernoulli(stream.key(), shape=shape), 1.0, -1.0)
    log_magnitudes = 0.05 * jr.normal(stream.key(), shape, dtype=jnp.float64)
    return signs * jnp.exp(log_magnitudes)


class MobiusSummation(eqx.Module):
    """Bank of signed phase warps with softly participating inputs.

    Attributes:
        weights: Signed Cayley-coordinate weights, shape
            (out_features, in_features).
        participation_logits: Unconstrained participation logits, same shape.
    """

    weights: LearnableParameter[JaxRealArray]
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
            weights=LearnableParameter(_signed_unit_initialization(shape, stream=stream)),
            participation_logits=LearnableParameter(
                initial_logit + scale * jr.normal(stream.key(), shape, dtype=jnp.float64)
            ),
        )

    def sum(self, x: JaxComplexArray) -> JaxComplexArray:
        """Apply the learned Möbius summation bank."""
        return mobius_sum(
            x,
            self.weights.value,
            sigmoid(self.participation_logits.value),
        )


class LowRankMobiusSummation(eqx.Module):
    """Möbius summation bank with low-rank weight and participation matrices."""

    weight_output: LearnableParameter[JaxRealArray]
    weight_input: LearnableParameter[JaxRealArray]
    participation_output: LearnableParameter[JaxRealArray]
    participation_input: LearnableParameter[JaxRealArray]

    @classmethod
    def create(
        cls,
        in_features: int,
        out_features: int,
        rank: int,
        *,
        streams: Mapping[str, RngStream],
    ) -> Self:
        if rank < 1 or rank > min(in_features, out_features):
            msg = f"rank must lie in [1, {min(in_features, out_features)}], got {rank}"
            raise ValueError(msg)
        stream = streams["parameters"]
        factor_noise = 0.05
        weight_output = factor_noise * jr.normal(
            stream.key(), (out_features, rank), dtype=jnp.float64
        )
        weight_input = factor_noise * jr.normal(
            stream.key(), (rank, in_features), dtype=jnp.float64
        )
        weight_output = weight_output.at[:, 0].add(
            _signed_unit_initialization((out_features,), stream=stream)
        )
        weight_input = weight_input.at[0, :].add(
            _signed_unit_initialization((in_features,), stream=stream)
        )

        participation_output = factor_noise * jr.normal(
            stream.key(), (out_features, rank), dtype=jnp.float64
        )
        participation_input = factor_noise * jr.normal(
            stream.key(), (rank, in_features), dtype=jnp.float64
        )
        participation_output = participation_output.at[:, 0].add(1)
        participation_input = participation_input.at[0, :].add(-jnp.log(in_features))
        return cls(
            weight_output=LearnableParameter(weight_output),
            weight_input=LearnableParameter(weight_input),
            participation_output=LearnableParameter(participation_output),
            participation_input=LearnableParameter(participation_input),
        )

    def sum(self, x: JaxComplexArray) -> JaxComplexArray:
        """Apply the generated dense Möbius summation bank."""
        return mobius_sum(
            x,
            self.weight_output.value @ self.weight_input.value,
            sigmoid(self.participation_output.value @ self.participation_input.value),
        )
