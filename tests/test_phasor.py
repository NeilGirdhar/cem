import jax.numpy as jnp

from cem.phasor.message import phasor_concordance, phasor_to_real


def test_concordance_with_self_is_squared_presence() -> None:
    phasor = jnp.array([3 + 4j, 1 + 0j, 0 + 2j])
    assert jnp.allclose(phasor_concordance(phasor, phasor), jnp.abs(phasor) ** 2)


def test_concordance_orthogonal_is_zero_antiphase_is_negative() -> None:
    left = jnp.array([1 + 0j, 1 + 0j])
    right = jnp.array([0 + 1j, -1 + 0j])
    concordance = phasor_concordance(left, right)
    assert jnp.allclose(concordance[0], 0.0, atol=1e-7)
    assert concordance[1] < 0


def test_to_real_doubles_last_dimension() -> None:
    phasor = jnp.ones((3, 4), dtype=jnp.complex128)
    assert phasor_to_real(phasor).shape == (3, 8)
