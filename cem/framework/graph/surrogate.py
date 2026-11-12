"""Scalar surrogate loss that applies a supplied cotangent through autodiff."""

import jax
import jax.numpy as jnp
from tjax import JaxArray, JaxRealArray


def cotangent_surrogate_loss(quantity: JaxArray, cotangent: JaxArray) -> JaxRealArray:
    """Return a scalar whose descent moves `quantity` along the supplied ascent cotangent.

    A link that receives a cotangent, rather than a scalar objective to differentiate, trains its
    destination through this surrogate. The thesis writes cotangent(X) := -dL/dX, so the
    surrogate's gradient with respect to `quantity` is `-cotangent`, and the cotangent itself is
    held fixed. See `eligibility-training-link` in the thesis.
    """
    return -jnp.sum(jnp.real(jax.lax.stop_gradient(cotangent) * quantity))
