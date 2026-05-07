from __future__ import annotations

import jax
import jax.numpy as jnp
from efax import Flattener, NormalNP, UnitVarianceNormalNP

from cem.phasor.frequency import geometric_frequencies, make_frequency_grid
from cem.phasor.message import (
    encode_scalar_phasors,
    phasor_concordance,
    phasor_from_distribution,
    phasor_to_conjugate_prior,
    phasor_to_distribution,
    phasor_to_real,
)

# ── from_distribution / to_distribution / to_conjugate_prior ─────────────────


def _normal_t(m: int, base: float = 2.0 * float(jnp.pi)) -> NormalNP:
    """Frequency grid t for NormalNP with m frequencies, d=2 sufficient statistics."""
    flattener, _ = Flattener.flatten(
        NormalNP(jnp.array(0.0), jnp.array(0.0)), mapped_to_plane=False
    )
    return make_frequency_grid(flattener, geometric_frequencies(m, base=base))


def _uvnormal_t(m: int, base: float = 1.0) -> UnitVarianceNormalNP:
    """Frequency grid t for UnitVarianceNormalNP with m frequencies, d=1."""
    flattener, _ = Flattener.flatten(UnitVarianceNormalNP(jnp.array(0.0)), mapped_to_plane=False)
    return make_frequency_grid(flattener, geometric_frequencies(m, base=base))


def test_from_distribution_shape() -> None:
    # NormalNP has d=2 sufficient statistics; m=4 frequencies → m*d=8 phasors
    dist = NormalNP(jnp.array(1.0), jnp.array(-0.5))
    freqs = geometric_frequencies(4)
    p = phasor_from_distribution(dist, freqs)
    assert p.shape == (8,)


def test_from_distribution_batched_shape() -> None:
    # dist shape (*s,) = (3,); m=4, d=2 → output (*s, m*d) = (3, 8)
    dist = NormalNP(jnp.array([0.0, 1.0, -1.0]), jnp.array([-0.5, -0.5, -0.5]))
    freqs = geometric_frequencies(4)
    p = phasor_from_distribution(dist, freqs)
    assert p.shape == (3, 8)


def test_from_distribution_unit_magnitude_for_point_mass() -> None:
    # A Normal with tiny variance (~point mass) should have phasors near unit magnitude
    dist = NormalNP(jnp.array(1e6), jnp.array(-1e6))  # huge precision → near point mass
    freqs = geometric_frequencies(4)
    p = phasor_from_distribution(dist, freqs)
    assert jnp.all(jnp.abs(p) <= 1.0 + 1e-6)


def test_from_distribution_presences_scales_magnitude() -> None:
    dist = UnitVarianceNormalNP(jnp.array(0.5))
    freqs = geometric_frequencies(8, base=1.0)
    presence = 2.0
    z_scaled = phasor_from_distribution(dist, freqs, presences=jnp.array(presence))
    z_base = phasor_from_distribution(dist, freqs)
    assert jnp.allclose(jnp.abs(z_scaled), jnp.abs(z_base) * presence)
    assert jnp.allclose(jnp.angle(z_scaled), jnp.angle(z_base), atol=1e-7)


def test_to_distribution_recovers_normal_mean() -> None:
    # NormalNP is d=2; to_distribution works for any d, no presence recovery.
    mu, var = 0.5, 1.0
    dist = NormalNP(jnp.array(mu), jnp.array(-0.5 / var))
    m = 8
    t = _normal_t(m, base=1.0)
    z = phasor_from_distribution(dist, geometric_frequencies(m, base=1.0))
    ep = phasor_to_distribution(z, t)
    assert jnp.allclose(ep.mean, jnp.array(mu), atol=1e-4)  # ty: ignore


def test_to_conjugate_prior_recovers_mean_and_presence() -> None:
    # UnitVarianceNormal is d=1 with HasConjugatePrior; OLS is exact for Normal.
    mu, presence = 0.5, 2.0
    dist = UnitVarianceNormalNP(jnp.array(mu))
    m = 8
    freqs = geometric_frequencies(m, base=1.0)
    z = phasor_from_distribution(dist, freqs, presences=jnp.array(presence))
    t = _uvnormal_t(m, base=1.0)
    cp = phasor_to_conjugate_prior(z, t)
    expected_cp = dist.to_exp().conjugate_prior_distribution(jnp.array(presence))
    for a, b in zip(
        jax.tree_util.tree_leaves(cp), jax.tree_util.tree_leaves(expected_cp), strict=True
    ):
        assert jnp.allclose(a, b, atol=1e-4)


# ── concordance ───────────────────────────────────────────────────────────────


def test_concordance_with_self_is_squared_presence() -> None:
    p = jnp.array([3 + 4j, 1 + 0j, 0 + 2j])
    assert jnp.allclose(phasor_concordance(p, p), jnp.abs(p) ** 2)


def test_concordance_orthogonal_is_zero_antiphase_is_negative() -> None:
    a = jnp.array([1 + 0j, 1 + 0j])
    b = jnp.array([0 + 1j, -1 + 0j])
    c = phasor_concordance(a, b)
    assert jnp.allclose(c[0], 0.0, atol=1e-7)
    assert c[1] < 0


# ── to_real ───────────────────────────────────────────────────────────────────


def test_to_real_doubles_last_dim() -> None:
    p = jnp.ones((3, 4), dtype=jnp.complex128)
    assert phasor_to_real(p).shape == (3, 8)


# ── encode_scalar ─────────────────────────────────────────────────────────────


def test_encode_scalar_batched_shape() -> None:
    freqs = geometric_frequencies(4)
    p = encode_scalar_phasors(jnp.array([1.0, 2.0, 3.0]), jnp.ones(3), freqs)
    assert p.shape == (3, 4)


def test_encode_scalar_presence_and_phase() -> None:
    freqs = geometric_frequencies(4)
    x, weight = 0.3, 2.5
    p = encode_scalar_phasors(jnp.array(x), jnp.array(weight), freqs)
    expected_phases = (x * freqs + jnp.pi) % (2 * jnp.pi) - jnp.pi
    assert jnp.allclose(jnp.abs(p), jnp.full(4, weight))
    assert jnp.allclose(jnp.angle(p), expected_phases, atol=1e-6)


# ── geometric_frequencies ─────────────────────────────────────────────────────


def test_geometric_frequencies() -> None:
    freqs = geometric_frequencies(5)
    assert freqs.shape == (5,)
    assert jnp.allclose(freqs[0], 1.0)
    assert jnp.allclose(freqs[1:] / freqs[:-1], jnp.full(4, 0.5))
