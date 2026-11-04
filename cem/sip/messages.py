"""SIP-specific Gaussian message helpers."""

from typing import Final

import jax
import jax.numpy as jnp
from efax import NormalNP
from tjax import JaxRealArray, divide_or_zero, divide_where

MAX_VARIANCE: Final[float] = 1e12
"""Fallback variance for absent (κ = 0) external messages. Large but finite so
downstream cross-entropy terms stay well-defined. Module-patch to override for
experiments that need a different ceiling."""


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
    strictly positive, use `jnp.reciprocal` directly.
    """
    return divide_where(
        jnp.ones_like(precision),
        precision,
        where=precision > 0,
        otherwise=jnp.full_like(precision, otherwise),
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
