"""Scalar surrogate losses that supply explicit cotangents to autodiff."""

import jax
import jax.numpy as jnp
from tjax import JaxArray, JaxRealArray


def promoting_link_loss(quantity: JaxArray, cotangent: JaxArray) -> JaxRealArray:
    """Return a surrogate scalar whose minimization ascends `quantity` by `cotangent`.

    Follows the thesis convention `cotangent(X) := -dL/dX`. The returned scalar's
    autodiff gradient with respect to `quantity` is `-cotangent`, so JAX descent
    moves `quantity` in direction `+cotangent`. See `promoting-link` in the thesis.
    """
    return -jnp.sum(jnp.real(jax.lax.stop_gradient(cotangent) * quantity))


def suppressing_link_loss(quantity: JaxRealArray, cotangent: JaxRealArray) -> JaxRealArray:
    """Return a surrogate scalar whose minimization descends `quantity` by `cotangent`.

    The returned scalar's autodiff gradient with respect to `quantity` is
    `+cotangent`, so JAX descent moves `quantity` in direction `-cotangent`. See
    `suppressing-link` in the thesis.
    """
    return jnp.sum(jax.lax.stop_gradient(cotangent) * quantity)
