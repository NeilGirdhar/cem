import jax
import jax.numpy as jnp

from cem.sip.messages import promoting_link_loss


def test_promoting_link_gradient_equals_negative_cotangent() -> None:
    """Promoting delivers a gradient of `-cotangent` so JAX descent ascends by `+cotangent`."""
    quantity = jnp.asarray([1.0, 2.0, 3.0])
    cotangent = jnp.asarray([0.5, -1.0, 2.0])
    grad = jax.grad(promoting_link_loss)(quantity, cotangent)
    assert jnp.array_equal(grad, -cotangent)


def test_promoting_link_ignores_cotangent_gradient() -> None:
    """No gradient flows into the cotangent argument."""
    quantity = jnp.asarray([1.0, 2.0])
    cotangent = jnp.asarray([0.5, -0.5])

    def promoting_of_cot(c: jnp.ndarray) -> jnp.ndarray:
        return promoting_link_loss(quantity, c)

    assert jnp.array_equal(jax.grad(promoting_of_cot)(cotangent), jnp.zeros_like(cotangent))
