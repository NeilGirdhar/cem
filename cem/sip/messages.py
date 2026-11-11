"""SIP-specific Gaussian message helpers."""

from typing import Final

import equinox as eqx
import jax.numpy as jnp
from efax import NormalNP, NormalPP
from tjax import JaxRealArray, divide_where

from cem.structure.graph.surrogate import (
    promoting_link_loss,
)

__all__ = [
    "MAX_VARIANCE",
    "InstrumentedMessage",
    "concat_instrumented_messages",
    "concat_messages",
    "evidence_scale",
    "promoting_link_loss",
    "variance_from_precision",
]

MAX_VARIANCE: Final[float] = 1e12
"""Fallback variance for absent (κ = 0) external messages. Large but finite so
downstream cross-entropy terms stay well-defined. Module-patch to override for
experiments that need a different ceiling."""


class InstrumentedMessage(eqx.Module):
    """A Gaussian observation paired with its instrument, exchanged between SIP features."""

    observation: NormalNP
    instrument: NormalNP

    @classmethod
    def standard(cls, size: int) -> InstrumentedMessage:
        """Standard normal on both halves: mean 0, precision 1."""
        return InstrumentedMessage(
            observation=NormalPP(jnp.zeros(size), jnp.ones(size)).to_nat(),
            instrument=NormalPP(jnp.zeros(size), jnp.ones(size)).to_nat(),
        )

    @classmethod
    def initial(cls, size: int) -> InstrumentedMessage:
        """First-step prior: standard normal observation, null instrument.

        The observation side is a proper standard normal (precision 1) so
        downstream reads that invert precision are well-posed. The instrument
        side is the null probe (precision 0): no experimental noise has been
        injected yet, and the next-level witness / instrument maps correctly
        see zero on both natural-parameter channels.
        """
        zero = jnp.zeros(size)
        return InstrumentedMessage(
            observation=NormalPP(zero, jnp.ones(size)).to_nat(),
            instrument=NormalPP(zero, zero).to_nat(),
        )


def variance_from_precision(
    precision: JaxRealArray, *, otherwise: float = MAX_VARIANCE
) -> JaxRealArray:
    """Return `1 / precision`, mapping absent messages (κ = 0) to `otherwise`.

    Use for precisions that aren't strictly positive by construction — external
    inputs such as an observed reward that may arrive in absent-message encoding.
    For `GaussianNPN` and `MLP` outputs whose softplus readout keeps precision
    strictly positive, use `1 / precision` directly.
    """
    return divide_where(
        jnp.ones_like(precision),
        precision,
        where=precision > 0,
        otherwise=jnp.full_like(precision, otherwise),
    )


def concat_messages(*messages: NormalNP) -> NormalNP:
    """Concatenate 1-D Gaussian messages along the component axis.

    Treats each input as an independent-component diagonal Gaussian and
    returns the joint with the components laid out in the given order.
    """
    return NormalNP(
        mean_times_precision=jnp.concat([m.mean_times_precision for m in messages]),
        negative_half_precision=jnp.concat([m.negative_half_precision for m in messages]),
    )


def concat_instrumented_messages(*messages: InstrumentedMessage) -> InstrumentedMessage:
    """Concatenate instrumented messages along the component axis on each channel."""
    return InstrumentedMessage(
        observation=concat_messages(*[m.observation for m in messages]),
        instrument=concat_messages(*[m.instrument for m in messages]),
    )


def evidence_scale(message: NormalNP, gain: JaxRealArray) -> NormalNP:
    """Scale a Gaussian message's evidence by `gain`.

    Multiplies both natural parameters by `gain`. The mean is preserved and the
    precision is scaled by `gain` (@evidence-scaling-link in `thesis.typ`).
    """
    return NormalNP(
        mean_times_precision=gain * message.mean_times_precision,
        negative_half_precision=gain * message.negative_half_precision,
    )
