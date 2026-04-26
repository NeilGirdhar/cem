from __future__ import annotations

import jax.numpy as jnp
import jax.scipy.special as jss

from cem.phasor.loss import (
    centering_loss,
    decorrelation_loss,
    spectral_reconstruction_loss,
    strength_loss,
)

# ── spectral_reconstruction_loss ──────────────────────────────────────────────


def test_reconstruction_loss_is_real() -> None:
    z = jnp.array([1 + 1j, 2 + 0j])
    z_hat = jnp.array([0.5 + 0.5j, 1 + 1j])
    loss = spectral_reconstruction_loss(z, z_hat)
    assert jnp.isrealobj(loss) or jnp.allclose(jnp.imag(loss), 0.0)


def test_reconstruction_loss_output_shape() -> None:
    z = jnp.ones((3, 4), dtype=jnp.complex128)
    assert spectral_reconstruction_loss(z, z).shape == (3, 4)


def test_reconstruction_loss_zero_input_equals_log_2pi() -> None:
    # z = z_hat = 0: log(2π I₀(0)) - Re(g(0) conj(0)) = log(2π * 1) - 0 = log(2π)
    z = jnp.zeros(2, dtype=jnp.complex128)
    assert jnp.allclose(spectral_reconstruction_loss(z, z), jnp.log(2.0 * jnp.pi) * jnp.ones(2))


def test_reconstruction_loss_self_is_minimum() -> None:
    # L(z, z) ≤ L(z, z_hat) for small perturbation.
    z = jnp.array([1.0 + 0j])
    z_hat = jnp.array([1.1 + 0j])
    assert spectral_reconstruction_loss(z, z)[0] <= spectral_reconstruction_loss(z, z_hat)[0]


# ── centering_loss ────────────────────────────────────────────────────────────


def test_centering_loss_zero_mean_is_zero() -> None:
    # Batch symmetric around zero.
    z = jnp.array([[1 + 0j, 0 + 1j], [-1 + 0j, 0 - 1j]])
    assert jnp.allclose(centering_loss(z), 0.0)


def test_centering_loss_is_nonnegative() -> None:
    z = jnp.array([[1 + 1j, 2 + 0j], [0.5 + 0.5j, -1 + 0j], [0 + 1j, 1 - 1j]])
    assert centering_loss(z) >= 0.0


def test_centering_loss_is_scalar() -> None:
    z = jnp.ones((4, 5), dtype=jnp.complex128)
    assert centering_loss(z).shape == ()


def test_centering_loss_nonzero_mean_gives_positive() -> None:
    z = jnp.array([[2 + 0j, 0 + 2j], [2 + 0j, 0 + 2j]])  # non-zero mean
    assert centering_loss(z) > 0.0


def test_centering_loss_scales_with_features() -> None:
    # Doubling the number of features doubles the loss (since it sums over features).
    z1 = jnp.array([[1 + 0j], [2 + 0j]])  # 1 feature, non-zero mean
    z2 = jnp.array([[1 + 0j, 1 + 0j], [2 + 0j, 2 + 0j]])  # 2 identical features
    assert jnp.allclose(centering_loss(z2), 2.0 * centering_loss(z1))


# ── strength_loss ─────────────────────────────────────────────────────────────


def test_strength_loss_zero_input_is_zero() -> None:
    # log I₀(0) = 0, so strength_loss = 0.
    z = jnp.zeros((4, 3), dtype=jnp.complex128)
    assert jnp.allclose(strength_loss(z), 0.0)


def test_strength_loss_is_nonpositive() -> None:
    z = jnp.array([[1 + 0j, 2 + 0j, 0.5 + 0j], [0.1 + 0j, 3 + 0j, 1 + 1j]])
    assert strength_loss(z) <= 0.0


def test_strength_loss_is_scalar() -> None:
    z = jnp.ones((4, 5), dtype=jnp.complex128)
    assert strength_loss(z).shape == ()


def test_strength_loss_larger_presence_more_negative() -> None:
    z_small = jnp.array([[0.5 + 0j, 0.5 + 0j]])
    z_large = jnp.array([[5.0 + 0j, 5.0 + 0j]])
    assert strength_loss(z_large) < strength_loss(z_small)


def test_strength_loss_formula() -> None:
    # Verify against direct computation: -sum(mean(log I₀(|z|))).
    z = jnp.array([[1.0 + 0j, 2.0 + 0j], [0.5 + 0.5j, 1.0 - 1.0j]])
    r = jnp.abs(z)
    expected = -jnp.sum(jnp.mean(r + jnp.log(jss.i0e(r)), axis=0))
    assert jnp.allclose(strength_loss(z), expected)


# ── decorrelation_loss ────────────────────────────────────────────────────────


def test_decorrelation_loss_output_shape_1d() -> None:
    pred = jnp.ones(4, dtype=jnp.complex128)
    target = jnp.ones(4, dtype=jnp.complex128)
    assert decorrelation_loss(pred, target).shape == ()


def test_decorrelation_loss_output_shape_batched() -> None:
    pred = jnp.ones((3, 4), dtype=jnp.complex128)
    target = jnp.ones((3, 4), dtype=jnp.complex128)
    assert decorrelation_loss(pred, target).shape == (3,)


def test_decorrelation_loss_zero_prediction_is_zero() -> None:
    pred = jnp.zeros(4, dtype=jnp.complex128)
    target = jnp.array([1 + 0j, 0 + 1j, -1 + 0j, 0 - 1j])
    assert jnp.allclose(decorrelation_loss(pred, target), 0.0)


def test_decorrelation_loss_aligned_is_positive() -> None:
    pred = jnp.array([1 + 0j, 0 + 1j])
    target = jnp.array([1 + 0j, 0 + 1j])
    assert decorrelation_loss(pred, target) > 0.0


def test_decorrelation_loss_antialigned_is_negative() -> None:
    pred = jnp.array([1 + 0j, 0 + 1j])
    target = jnp.array([-1 + 0j, 0 - 1j])
    assert decorrelation_loss(pred, target) < 0.0


def test_decorrelation_loss_is_real() -> None:
    pred = jnp.array([1 + 1j, 2 - 1j])
    target = jnp.array([0.5 + 0.5j, -1 + 2j])
    loss = decorrelation_loss(pred, target)
    assert jnp.isrealobj(loss) or jnp.allclose(jnp.imag(loss), 0.0)


def test_decorrelation_loss_formula() -> None:
    pred = jnp.array([1 + 2j, -1 + 1j])
    target = jnp.array([0 + 1j, 2 - 1j])
    expected = jnp.real(jnp.sum(jnp.conj(pred) * target))
    assert jnp.allclose(decorrelation_loss(pred, target), expected)
