"""Check the surrogate loss that applies a supplied cotangent."""

import jax
import jax.numpy as jnp

from cem.framework.graph import cotangent_surrogate_loss


def test_cotangent_surrogate_gradient_equals_negative_cotangent() -> None:
    """The surrogate has gradient `-cotangent`, so JAX descent ascends by `+cotangent`."""
    quantity = jnp.asarray([1.0, 2.0, 3.0])
    cotangent = jnp.asarray([0.5, -1.0, 2.0])
    grad = jax.grad(cotangent_surrogate_loss)(quantity, cotangent)
    assert jnp.array_equal(grad, -cotangent)


def test_cotangent_surrogate_ignores_cotangent_gradient() -> None:
    """No gradient flows into the cotangent argument."""
    quantity = jnp.asarray([1.0, 2.0])
    cotangent = jnp.asarray([0.5, -0.5])

    def surrogate_of_cotangent(c: jnp.ndarray) -> jnp.ndarray:
        return cotangent_surrogate_loss(quantity, c)

    assert jnp.array_equal(jax.grad(surrogate_of_cotangent)(cotangent), jnp.zeros_like(cotangent))
