from __future__ import annotations

from collections.abc import Mapping

import jax.numpy as jnp
from efax import Flattener, NormalNP
from tjax import RngStream

from cem.phasor.frequency import geometric_frequencies, make_frequency_grid
from cem.phasor.message import PhasorMessage

# ── from_distribution / to_distribution ──────────────────────────────────────


def _normal_t(m: int, base: float = 2.0 * float(jnp.pi)) -> NormalNP:
    """Frequency grid t for NormalNP with m frequencies, d=2 sufficient statistics."""
    flattener, _ = Flattener.flatten(
        NormalNP(jnp.array(0.0), jnp.array(0.0)), mapped_to_plane=False
    )
    return make_frequency_grid(flattener, geometric_frequencies(m, base=base))


def test_from_distribution_shape() -> None:
    # NormalNP has d=2 sufficient statistics; m=4 frequencies → m*d=8 phasors
    dist = NormalNP(jnp.array(1.0), jnp.array(-0.5))
    freqs = geometric_frequencies(4)
    p = PhasorMessage.from_distribution(dist, freqs)
    assert p.data.shape == (8,)


def test_from_distribution_batched_shape() -> None:
    # dist shape (*s,) = (3,); m=4, d=2 → output (*s, m*d) = (3, 8)
    dist = NormalNP(jnp.array([0.0, 1.0, -1.0]), jnp.array([-0.5, -0.5, -0.5]))
    freqs = geometric_frequencies(4)
    p = PhasorMessage.from_distribution(dist, freqs)
    assert p.data.shape == (3, 8)


def test_from_distribution_unit_magnitude_for_point_mass() -> None:
    # A Normal with tiny variance (~point mass) should have phasors near unit magnitude
    dist = NormalNP(jnp.array(1e6), jnp.array(-1e6))  # huge precision → near point mass
    freqs = geometric_frequencies(4)
    p = PhasorMessage.from_distribution(dist, freqs)
    assert jnp.all(p.presence <= 1.0 + 1e-6)


def test_to_distribution_recovers_normal_mean() -> None:
    # OLS recovery uses Im(log CF(t)) ≈ ⟨t, E[T(x)]⟩, valid when |f * σ²| << 1.
    # base=1e-4, m=8 → max f = 1e-4 * 2^7 = 0.0128; max f*σ² = 0.0128 << 1 ✓
    mu, sigma2 = 0.5, 1.0
    dist = NormalNP(jnp.array(mu / sigma2), jnp.array(-0.5 / sigma2))
    m = 8
    base = 1e-4
    freqs = geometric_frequencies(m, base=base)
    p = PhasorMessage.from_distribution(dist, freqs)
    t = _normal_t(m, base=base)
    ep = p.to_distribution(t)
    # NormalEP fields: mean=μ, second_moment=μ²+σ²
    assert jnp.allclose(ep.mean, jnp.array(mu), atol=1e-3)  # ty: ignore
    assert jnp.allclose(ep.second_moment, jnp.array(mu**2 + sigma2), atol=1e-3)  # ty: ignore


# ── construction ──────────────────────────────────────────────────────────────


def test_zeros_shape() -> None:
    assert PhasorMessage.zeros(5).data.shape == (5,)


def test_from_polar_roundtrip() -> None:
    presence = jnp.array([1.0, 2.0, 0.5])
    value = jnp.array([0.0, jnp.pi / 2, -jnp.pi / 3])
    p = PhasorMessage.from_polar(presence, value)
    assert jnp.allclose(p.presence, presence)
    assert jnp.allclose(p.value, value, atol=1e-7)


# ── scaled ────────────────────────────────────────────────────────────────────


def test_scaled_adjusts_presence_preserves_phase() -> None:
    p = PhasorMessage(jnp.array([1 + 1j, 0 + 2j, -1 + 0j]))
    factor = jnp.array([3.0, 0.5, 2.0])
    scaled = p.scaled(factor)
    assert jnp.allclose(scaled.presence, p.presence * factor)
    assert jnp.allclose(scaled.value, p.value, atol=1e-7)


# ── rotated ───────────────────────────────────────────────────────────────────


def test_rotated_unit_preserves_presence_shifts_phase() -> None:
    p = PhasorMessage(jnp.array([2 + 0j]))  # phase = 0, presence = 2
    rotation = jnp.array([jnp.exp(0.5j)])
    out = p.rotated(rotation)
    assert jnp.allclose(out.presence, p.presence, atol=1e-7)
    assert jnp.allclose(out.value, jnp.array([0.5]), atol=1e-7)


# ── concordance ───────────────────────────────────────────────────────────────


def test_concordance_with_self_is_squared_presence() -> None:
    p = PhasorMessage(jnp.array([3 + 4j, 1 + 0j, 0 + 2j]))
    assert jnp.allclose(p.concordance(p), p.presence**2)


def test_concordance_orthogonal_is_zero_antiphase_is_negative() -> None:
    a = PhasorMessage(jnp.array([1 + 0j, 1 + 0j]))
    b = PhasorMessage(jnp.array([0 + 1j, -1 + 0j]))
    c = a.concordance(b)
    assert jnp.allclose(c[0], 0.0, atol=1e-7)
    assert c[1] < 0


# ── dropout ───────────────────────────────────────────────────────────────────


def test_dropout_zero_rate_is_identity(streams: Mapping[str, RngStream]) -> None:
    p = PhasorMessage(jnp.array([1 + 1j, 2 - 1j, 0.5 + 0.5j]))
    assert jnp.allclose(p.dropout(streams["inference"].key(), 0.0).data, p.data)


def test_dropout_preserves_expected_value(streams: Mapping[str, RngStream]) -> None:
    p = PhasorMessage(jnp.array([1 + 0j, 0 + 2j, -1 + 1j]))
    stream = streams["inference"]
    samples = jnp.stack([p.dropout(stream.key(), 0.3).data for _ in range(2000)])
    assert jnp.allclose(jnp.mean(samples, axis=0), p.data, atol=0.1)


# ── to_real ───────────────────────────────────────────────────────────────────


def test_to_real_doubles_last_dim() -> None:
    p = PhasorMessage(jnp.ones((3, 4), dtype=jnp.complex128))
    assert p.to_real().shape == (3, 8)


# ── encode_scalar ─────────────────────────────────────────────────────────────


def test_encode_scalar_batched_shape() -> None:
    freqs = geometric_frequencies(4)
    p = PhasorMessage.encode_scalar(jnp.array([1.0, 2.0, 3.0]), jnp.ones(3), freqs)
    assert p.data.shape == (3, 4)


def test_encode_scalar_presence_and_phase() -> None:
    freqs = geometric_frequencies(4)
    x, weight = 0.3, 2.5
    p = PhasorMessage.encode_scalar(jnp.array(x), jnp.array(weight), freqs)
    expected_phases = (x * freqs + jnp.pi) % (2 * jnp.pi) - jnp.pi
    assert jnp.allclose(p.presence, jnp.full(4, weight))
    assert jnp.allclose(p.value, expected_phases, atol=1e-6)


# ── geometric_frequencies ─────────────────────────────────────────────────────


def test_geometric_frequencies() -> None:
    freqs = geometric_frequencies(5)
    assert freqs.shape == (5,)
    assert jnp.allclose(freqs[0], 1.0)
    assert jnp.allclose(freqs[1:] / freqs[:-1], jnp.full(4, 0.5))
