"""Differentiate named objectives and the remaining loss from the same forward pass."""

from collections.abc import Callable

import equinox as eqx
import jax
import jax.numpy as jnp
from jax import tree
from tjax import JaxArray, PyTree


class LossGradients[Parameters: PyTree](eqx.Module):
    """Gradients of the selected losses and of the rest of the objective."""

    balanced_gradients: dict[str, Parameters]
    fixed_weight_gradient: Parameters


def differentiate_losses[Parameters: PyTree, Aux](
    forward: Callable[[Parameters], tuple[tuple[dict[str, JaxArray], JaxArray], Aux]],
    parameters: Parameters,
) -> tuple[LossGradients[Parameters], Aux]:
    """Pull back each scalar loss separately while retaining one set of forward outputs."""
    (named, remaining), pullback, aux = jax.vjp(forward, parameters, has_aux=True)
    zeros = tree.map(jnp.zeros_like, named)
    gradients = {
        name: pullback(({**zeros, name: jnp.ones_like(loss)}, jnp.zeros_like(remaining)))[0]
        for name, loss in named.items()
    }
    (remaining_gradient,) = pullback((zeros, jnp.ones_like(remaining)))
    return LossGradients(gradients, remaining_gradient), aux
