import jax.numpy as jnp
from tjax import JaxArray, JaxRealArray

type JaxComplexArray = JaxArray


def phasor_concordance(left: JaxComplexArray, right: JaxComplexArray) -> JaxRealArray:
    """Elementwise agreement Re(left * conj(right))."""
    return jnp.real(left * jnp.conj(right))


def phasor_to_real(z: JaxComplexArray) -> JaxArray:
    """Return (Re, Im) parts concatenated as a real array for neural network input."""
    return jnp.concat([jnp.real(z), jnp.imag(z)], axis=-1)
