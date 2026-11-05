"""SIP-specific Gaussian message helpers."""

from typing import Final

import equinox as eqx
import jax
import jax.numpy as jnp
from efax import NormalNP
from tjax import JaxRealArray, divide_or_zero, divide_where

from cem.npn.gaussian import normal_from_mean_and_presence

MAX_VARIANCE: Final[float] = 1e12
"""Fallback variance for absent (κ = 0) external messages. Large but finite so
downstream cross-entropy terms stay well-defined. Module-patch to override for
experiments that need a different ceiling."""


class InstrumentedMessage(eqx.Module):
    """A Gaussian message paired with its instrument half.

    SIP signals travel on two co-indexed channels: an observation side carrying
    the content and an instrument side carrying the causal-identification
    counterpart. This pair is the message exchanged between features and
    between a feature and its recruiter.
    """

    observation: NormalNP
    instrument: NormalNP

    @classmethod
    def standard(cls, size: int) -> InstrumentedMessage:
        """Standard normal on both halves: mean 0, precision 1."""
        return InstrumentedMessage(
            observation=normal_from_mean_and_presence(jnp.zeros(size), jnp.ones(size)),
            instrument=normal_from_mean_and_presence(jnp.zeros(size), jnp.ones(size)),
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
            observation=normal_from_mean_and_presence(zero, jnp.ones(size)),
            instrument=normal_from_mean_and_presence(zero, zero),
        )


def mean_and_precision(distribution: NormalNP) -> tuple[JaxRealArray, JaxRealArray]:
    """Extract the mean and precision from a Gaussian natural-parameter message."""
    precision = -2 * distribution.negative_half_precision
    mean = divide_or_zero(distribution.mean_times_precision, precision)
    return mean, precision


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


def promoting_link_loss(quantity: JaxRealArray, cotangent: JaxRealArray) -> JaxRealArray:
    """Return a surrogate scalar whose minimization ascends `quantity` by `cotangent`.

    Follows the thesis convention `cotangent(X) := -dL/dX`. The returned scalar's
    autodiff gradient with respect to `quantity` is `-cotangent`, so JAX descent
    moves `quantity` in direction `+cotangent`. See `promoting-link` in
    `~/src/typst/thesis.typ`.
    """
    return -jnp.sum(jax.lax.stop_gradient(cotangent) * quantity)


def suppressing_link_loss(quantity: JaxRealArray, cotangent: JaxRealArray) -> JaxRealArray:
    """Return a surrogate scalar whose minimization descends `quantity` by `cotangent`.

    The returned scalar's autodiff gradient with respect to `quantity` is
    `+cotangent`, so JAX descent moves `quantity` in direction `-cotangent`. See
    `suppressing-link` in `~/src/typst/thesis.typ`.
    """
    return jnp.sum(jax.lax.stop_gradient(cotangent) * quantity)
