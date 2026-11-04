import jax
import jax.numpy as jnp

from cem.sip.messages import promoting_link_loss, suppressing_link_loss


def test_promoting_link_gradient_equals_negative_cotangent() -> None:
    """Promoting delivers a gradient of `-cotangent` so JAX descent ascends by `+cotangent`."""
    quantity = jnp.array([1.0, 2.0, 3.0])
    cotangent = jnp.array([0.5, -1.0, 2.0])
    grad = jax.grad(promoting_link_loss)(quantity, cotangent)
    assert jnp.array_equal(grad, -cotangent)


def test_suppressing_link_gradient_equals_cotangent() -> None:
    """Suppressing delivers a gradient of `+cotangent` so JAX descent descends by `-cotangent`."""
    quantity = jnp.array([1.0, 2.0, 3.0])
    cotangent = jnp.array([0.5, -1.0, 2.0])
    grad = jax.grad(suppressing_link_loss)(quantity, cotangent)
    assert jnp.array_equal(grad, cotangent)


def test_link_losses_ignore_cotangent_gradient() -> None:
    """Neither link lets gradient flow into the cotangent argument."""
    quantity = jnp.array([1.0, 2.0])
    cotangent = jnp.array([0.5, -0.5])

    def promoting_of_cot(c: jnp.ndarray) -> jnp.ndarray:
        return promoting_link_loss(quantity, c)

    def suppressing_of_cot(c: jnp.ndarray) -> jnp.ndarray:
        return suppressing_link_loss(quantity, c)

    assert jnp.array_equal(jax.grad(promoting_of_cot)(cotangent), jnp.zeros_like(cotangent))
    assert jnp.array_equal(jax.grad(suppressing_of_cot)(cotangent), jnp.zeros_like(cotangent))
