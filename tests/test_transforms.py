from __future__ import annotations

from collections.abc import Mapping

import jax.numpy as jnp
import jax.random as jr
import pytest
from tjax import RngStream

from cem.phasor import (
    Accumulator,
    Linear,
    LinearWithDropout,
    Nonlinear,
    PhasorMessage,
    RivalryGroups,
    RivalryNorm,
    interpolate,
    phasor_gate,
    rotate_by_location,
    select,
)

# ── phasor_gate ────────────────────────────────────────────────────────────────


def test_gate_zero_real_gives_half_scale() -> None:
    gate = jnp.zeros(3, dtype=jnp.complex128)
    z = jnp.array([1 + 2j, -1 + 0j, 0 + 1j])
    assert jnp.allclose(phasor_gate(gate, z), 0.5 * z)


def test_gate_large_positive_real_passes_through() -> None:
    gate = jnp.full(3, 100.0, dtype=jnp.complex128)
    z = jnp.array([1 + 1j, 2 + 0j, -1 - 1j])
    assert jnp.allclose(phasor_gate(gate, z), z, atol=1e-4)


def test_gate_large_negative_real_suppresses() -> None:
    gate = jnp.full(3, -100.0, dtype=jnp.complex128)
    z = jnp.array([1 + 1j, 2 + 0j, -1 - 1j])
    assert jnp.allclose(phasor_gate(gate, z), jnp.zeros(3, dtype=jnp.complex128), atol=1e-4)


def test_gate_imaginary_part_ignored() -> None:
    # Only Re(gate) matters; imaginary part should not affect output.
    z = jnp.array([1 + 1j, 2 - 1j, 0 + 0.5j])
    gate_zero = jnp.zeros(3, dtype=jnp.complex128)
    gate_imag = jnp.array([0 + 5j, 0 - 3j, 0 + 1j])
    assert jnp.allclose(phasor_gate(gate_zero, z), phasor_gate(gate_imag, z))


def test_gate_output_shape() -> None:
    gate = jnp.zeros((2, 4), dtype=jnp.complex128)
    z = jnp.ones((2, 4), dtype=jnp.complex128)
    assert phasor_gate(gate, z).shape == (2, 4)


# ── rotate_by_location ────────────────────────────────────────────────────────


def test_rotate_preserves_presence() -> None:
    z = jnp.array([1 + 1j, 2 - 1j, 0.5 + 0.5j])
    loc = jnp.array([0.3 + 0.4j, 1 + 0j, -1 + 1j])
    assert jnp.allclose(jnp.abs(rotate_by_location(z, loc)), jnp.abs(z))


def test_rotate_zero_location_is_identity() -> None:
    z = jnp.array([1 + 1j, 2 - 1j])
    assert jnp.allclose(rotate_by_location(z, jnp.zeros(2, dtype=jnp.complex128)), z)


def test_rotate_90_degrees() -> None:
    # Multiplying by i rotates 90 degrees.
    z = jnp.array([1 + 0j])
    loc = jnp.array([0 + 1j])
    assert jnp.allclose(rotate_by_location(z, loc), jnp.array([0 + 1j]), atol=1e-7)


def test_rotate_location_magnitude_ignored() -> None:
    # A location with magnitude 5 should rotate the same as magnitude 1.
    z = jnp.array([1 + 0j, 0 + 1j])
    loc_unit = jnp.array([0 + 1j, 1 + 0j])
    assert jnp.allclose(rotate_by_location(z, loc_unit), rotate_by_location(z, 5.0 * loc_unit))


# ── Accumulator ───────────────────────────────────────────────────────────────


def test_accumulator_init_shape_and_dtype() -> None:
    state = Accumulator(features=5).init()
    assert state.shape == (5,)
    assert state.dtype == jnp.complex128


def test_accumulator_init_is_zero() -> None:
    assert jnp.all(Accumulator(features=5).init() == 0)


def test_accumulator_zero_gate_forgets_state() -> None:
    acc = Accumulator(features=3)
    state = jnp.array([1 + 1j, 2 - 1j, -1 + 0.5j])
    decay = jnp.full(3, -100.0, dtype=jnp.complex128)  # sigmoid(-100) ≈ 0
    out = acc.infer(state, decay, jnp.zeros(3, dtype=jnp.complex128))
    assert jnp.allclose(out, jnp.zeros(3, dtype=jnp.complex128), atol=1e-4)


def test_accumulator_full_gate_retains_state_plus_increment() -> None:
    acc = Accumulator(features=3)
    state = jnp.ones(3, dtype=jnp.complex128)
    decay = jnp.full(3, 100.0, dtype=jnp.complex128)  # sigmoid(100) ≈ 1
    increment = jnp.array([0.5 + 0.5j, -1 + 0j, 0 + 2j])
    assert jnp.allclose(acc.infer(state, decay, increment), state + increment, atol=1e-4)


def test_accumulator_batched_shape() -> None:
    acc = Accumulator(features=4)
    state = jnp.zeros((3, 4), dtype=jnp.complex128)
    decay = jnp.zeros((3, 4), dtype=jnp.complex128)
    increment = jnp.ones((3, 4), dtype=jnp.complex128)
    assert acc.infer(state, decay, increment).shape == (3, 4)


# ── Linear ────────────────────────────────────────────────────────────────────


def test_linear_output_shape(streams: Mapping[str, RngStream]) -> None:
    f = Linear.create(3, 5, streams=streams)
    assert f.infer(jnp.ones(3, dtype=jnp.complex128)).shape == (5,)


def test_linear_output_dtype(streams: Mapping[str, RngStream]) -> None:
    assert (
        Linear.create(3, 5, streams=streams).infer(jnp.ones(3, dtype=jnp.complex128)).dtype
        == jnp.complex128
    )


def test_linear_batched_shape(streams: Mapping[str, RngStream]) -> None:
    f = Linear.create(3, 5, streams=streams)
    assert f.infer(jnp.ones((4, 3), dtype=jnp.complex128)).shape == (4, 5)


def test_linear_zero_input_gives_bias(streams: Mapping[str, RngStream]) -> None:
    f = Linear.create(3, 5, streams=streams)
    assert jnp.allclose(f.infer(jnp.zeros(3, dtype=jnp.complex128)), f.bias.value)


def test_linear_is_linear(streams: Mapping[str, RngStream]) -> None:
    f = Linear.create(4, 6, streams=streams)
    stream = streams["inference"]
    z1 = jr.normal(stream.key(), (4,)) + 1j * jr.normal(stream.key(), (4,))
    z2 = jr.normal(stream.key(), (4,)) + 1j * jr.normal(stream.key(), (4,))
    a, b = 2.0 + 1j, -1.0 + 0.5j
    assert jnp.allclose(
        f.infer(a * z1 + b * z2),
        a * f.infer(z1) + b * f.infer(z2),
    )


def test_linear_weight_shape(streams: Mapping[str, RngStream]) -> None:
    f = Linear.create(3, 5, streams=streams)
    assert f.weight.value.shape == (5, 3)
    assert f.bias.value.shape == (5,)


# ── LinearWithDropout ─────────────────────────────────────────────────────────


def test_linear_with_dropout_output_shape(streams: Mapping[str, RngStream]) -> None:
    f = LinearWithDropout.create(3, 5, streams=streams)
    assert f.infer(jnp.ones(3, dtype=jnp.complex128), streams=streams, inference=True).shape == (5,)


def test_linear_with_dropout_zero_rate_gives_bias(streams: Mapping[str, RngStream]) -> None:
    f = LinearWithDropout.create(3, 5, dropout_rate=0.0, streams=streams)
    assert jnp.allclose(
        f.infer(jnp.zeros(3, dtype=jnp.complex128), streams=streams, inference=True), f.bias.value
    )


def test_linear_with_dropout_skips_dropout_when_inference_true(
    streams: Mapping[str, RngStream],
) -> None:
    # inference=True: output matches the raw affine transform (no dropout applied).
    f = LinearWithDropout.create(3, 5, dropout_rate=0.9, streams=streams)
    z = jnp.ones(3, dtype=jnp.complex128)
    assert jnp.allclose(f.infer(z, streams=streams, inference=True), Linear.infer(f, z))


def test_linear_with_dropout_applies_dropout_when_inference_false(
    streams: Mapping[str, RngStream],
) -> None:
    # inference=False: at least one output differs from the no-dropout result (inference=True).
    f = LinearWithDropout.create(3, 20, dropout_rate=0.5, streams=streams)
    z = jnp.ones(3, dtype=jnp.complex128)
    out_train = f.infer(z, streams=streams, inference=False)
    out_eval = f.infer(z, streams=streams, inference=True)
    assert not jnp.allclose(out_train, out_eval)


# ── RivalryGroups ─────────────────────────────────────────────────────────────


def test_rivalry_groups_weights_shape(streams: Mapping[str, RngStream]) -> None:
    assert RivalryGroups.create(8, 4, streams=streams).weights.shape == (4, 8)


def test_rivalry_groups_weights_sum_to_one_columnwise(streams: Mapping[str, RngStream]) -> None:
    g = RivalryGroups.create(8, 4, streams=streams)
    assert jnp.allclose(g.weights.sum(axis=0), jnp.ones(8))


def test_rivalry_groups_weights_nonnegative(streams: Mapping[str, RngStream]) -> None:
    assert jnp.all(RivalryGroups.create(8, 4, streams=streams).weights >= 0)


# ── RivalryNorm ───────────────────────────────────────────────────────────────


@pytest.fixture
def rivalry_norm(streams: Mapping[str, RngStream]) -> RivalryNorm:
    return RivalryNorm.create(6, 3, streams=streams)


@pytest.fixture
def rivalry_z() -> jnp.ndarray:
    return jnp.array([1 + 1j, 2 - 1j, 0.5 + 0.5j, -1 + 0j, 0.3 - 0.7j, 1 + 0j])


def test_rivalry_norm_output_shape(rivalry_norm: RivalryNorm, rivalry_z: jnp.ndarray) -> None:
    assert rivalry_norm.infer(rivalry_z).shape == (6,)


def test_rivalry_norm_preserves_phase(rivalry_norm: RivalryNorm, rivalry_z: jnp.ndarray) -> None:
    out = rivalry_norm.infer(rivalry_z)
    assert jnp.allclose(jnp.angle(out), jnp.angle(rivalry_z), atol=1e-6)


def test_rivalry_norm_gauge_invariance(rivalry_norm: RivalryNorm, rivalry_z: jnp.ndarray) -> None:
    # Scaling all presences by a constant c leaves normalized presences unchanged.
    out1 = rivalry_norm.infer(rivalry_z)
    out2 = rivalry_norm.infer(10.0 * rivalry_z)
    assert jnp.allclose(jnp.abs(out1), jnp.abs(out2), rtol=1e-5)


def test_rivalry_norm_batched_shape(rivalry_norm: RivalryNorm) -> None:
    assert rivalry_norm.infer(jnp.ones((3, 6), dtype=jnp.complex128)).shape == (3, 6)


# ── Nonlinear ─────────────────────────────────────────────────────────────────


def test_nonlinear_output_shape(streams: Mapping[str, RngStream]) -> None:
    f = Nonlinear.create(4, 6, 3, streams=streams)
    assert f.infer(
        PhasorMessage(jnp.ones(4, dtype=jnp.complex128)), streams=streams, inference=True
    ).shape == (6,)


def test_nonlinear_output_dtype(streams: Mapping[str, RngStream]) -> None:
    f = Nonlinear.create(4, 6, 3, streams=streams)
    assert (
        f.infer(
            PhasorMessage(jnp.ones(4, dtype=jnp.complex128)), streams=streams, inference=True
        ).data.dtype
        == jnp.complex128
    )


def test_nonlinear_batched_shape(streams: Mapping[str, RngStream]) -> None:
    f = Nonlinear.create(4, 6, 3, streams=streams)
    assert f.infer(
        PhasorMessage(jnp.ones((5, 4), dtype=jnp.complex128)), streams=streams, inference=True
    ).shape == (5, 6)


def test_nonlinear_custom_mid_features(streams: Mapping[str, RngStream]) -> None:
    f = Nonlinear.create(4, 6, 3, mid_features=8, streams=streams)
    assert f.infer(
        PhasorMessage(jnp.ones(4, dtype=jnp.complex128)), streams=streams, inference=True
    ).shape == (6,)


# ── select ────────────────────────────────────────────────────────────────────


def test_select_output_shape() -> None:
    # 2 heads, 5 candidates, 4 alignment features → (2, 5)
    assert select(
        jnp.ones((2, 5, 4), dtype=jnp.complex128), jnp.ones((2, 4), dtype=jnp.complex128)
    ).shape == (2, 5)


def test_select_sums_to_one_per_head() -> None:
    # Each head's weights must form a probability distribution over candidates.
    keys = jnp.array(
        [
            [[1 + 0j, 0 + 1j], [0 + 1j, 1 + 0j], [-1 + 0j, 0 - 1j]],
            [[1 + 0j, 0 + 1j], [0 + 1j, 1 + 0j], [-1 + 0j, 0 - 1j]],
        ]
    )  # (2, 3, 2)
    query = jnp.array([[1 + 0j, 0 + 1j], [1 + 0j, 0 + 1j]])  # (2, 2)
    weights = select(keys, query)  # (2, 3)
    assert jnp.allclose(weights.sum(axis=-1), jnp.ones(2))


def test_select_nonnegative() -> None:
    keys = jnp.ones((2, 3, 4), dtype=jnp.complex128)
    query = jnp.ones((2, 4), dtype=jnp.complex128)
    assert jnp.all(select(keys, query) >= 0)


def test_select_aligned_key_wins() -> None:
    # Within each head, the key that exactly matches the query gets the highest weight.
    query_vec = jnp.array([1 + 0j, 0 + 1j])
    candidate_keys = jnp.stack(
        [
            query_vec,  # concordance = 2 (aligned)
            jnp.array([0 + 1j, -1 + 0j]),  # concordance = 0 (orthogonal)
            jnp.array([-1 + 0j, 0 - 1j]),  # concordance = -2 (anti-aligned)
        ]
    )  # (3, 2)
    keys = candidate_keys[jnp.newaxis]  # (1, 3, 2)
    query = query_vec[jnp.newaxis]  # (1, 2)
    weights = select(keys, query)[0]  # (3,)
    assert weights[0] == weights.max()


def test_select_heads_are_independent() -> None:
    # Different queries should yield different weights even with the same keys.
    keys = jnp.ones((2, 3, 2), dtype=jnp.complex128)
    query = jnp.array([[1 + 0j, 0 + 0j], [0 + 0j, 1 + 0j]])  # (2, 2)
    w = select(keys, query)
    # Uniform keys → weights should be uniform regardless of query, but shapes should be (2, 3)
    assert w.shape == (2, 3)


def test_select_batched_shape() -> None:
    # Batch dim prepended: (*batch, h, m, n) and (*batch, h, n) → (*batch, h, m)
    keys = jnp.ones((3, 2, 5, 4), dtype=jnp.complex128)
    query = jnp.ones((3, 2, 4), dtype=jnp.complex128)
    assert select(keys, query).shape == (3, 2, 5)


# ── interpolate ───────────────────────────────────────────────────────────────


def test_interpolate_output_shape() -> None:
    # 2 heads, 3 candidates, 4 content features → (2*4,) = (8,)
    weights = jnp.ones((2, 3)) / 3
    content = jnp.ones((2, 3, 4), dtype=jnp.complex128)
    assert interpolate(weights, content).shape == (8,)


def test_interpolate_uniform_weights_give_per_head_mean() -> None:
    # With uniform weights each head returns the mean of its content rows.
    weights = jnp.ones((1, 3)) / 3
    content = jnp.array([[[1 + 0j, 2 + 0j], [3 + 0j, 4 + 0j], [5 + 0j, 6 + 0j]]])  # (1, 3, 2)
    expected = jnp.mean(content[0], axis=0)  # (2,)
    assert jnp.allclose(interpolate(weights, content), expected)


def test_interpolate_one_hot_selects_row() -> None:
    weights = jnp.array([[0.0, 1.0, 0.0]])  # (1, 3)
    content = jnp.array([[[1 + 0j, 2 + 0j], [3 + 4j, 5 + 6j], [7 + 0j, 8 + 0j]]])  # (1, 3, 2)
    assert jnp.allclose(interpolate(weights, content), content[0, 1])


def test_interpolate_batched_shape() -> None:
    # Batch dim prepended: (*batch, h, m) and (*batch, h, m, d) → (*batch, h*d)
    weights = jnp.ones((3, 2, 5)) / 5
    content = jnp.ones((3, 2, 5, 4), dtype=jnp.complex128)
    assert interpolate(weights, content).shape == (3, 8)


def test_interpolate_concatenates_heads() -> None:
    # Two heads with one-hot selection: output should be the concatenation of selected rows.
    weights = jnp.array([[1.0, 0.0], [0.0, 1.0]])  # (2, 2) — head 0 picks row 0, head 1 picks row 1
    content = jnp.array(
        [
            [[1 + 0j, 2 + 0j], [3 + 0j, 4 + 0j]],  # head 0 content
            [[5 + 0j, 6 + 0j], [7 + 0j, 8 + 0j]],  # head 1 content
        ]
    )  # (2, 2, 2)
    result = interpolate(weights, content)  # should be (4,)
    expected = jnp.array([1 + 0j, 2 + 0j, 7 + 0j, 8 + 0j])
    assert jnp.allclose(result, expected)
