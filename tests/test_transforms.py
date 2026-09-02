from collections.abc import Mapping

import jax.numpy as jnp
import pytest
from tjax import RngStream

from cem.phasor import (
    Accumulator,
    ElementwiseRotation,
    FrequencyAdaptedEvidencePooling,
    FrequencyElementwiseRotation,
    FrequencyGatedProjection,
    FrequencyMobiusProjection,
    FrequencyPhaseActivation,
    GatedProjection,
    LogSpaceProjection,
    LogSpaceProjectionWithDropout,
    LowRankFrequencyAdaptedEvidencePooling,
    LowRankMobiusSummation,
    MobiusParameterization,
    MobiusSummation,
    RecurrentPhaseFocusing,
    ValueProjection,
    interpolate,
    mobius_sum,
    phase_warp,
    phasor_gate,
    rotate_by_location,
    select,
)
from cem.structure.graph import FixedParameter, LearnableParameter
from cem.transforms import dropout

# ── dropout ───────────────────────────────────────────────────────────────────


def test_dropout_zero_rate_is_identity(streams: Mapping[str, RngStream]) -> None:
    p = jnp.array([1 + 1j, 2 - 1j, 0.5 + 0.5j])
    assert jnp.allclose(dropout(p, streams["inference"].key(), 0.0), p)


def test_dropout_preserves_expected_value(streams: Mapping[str, RngStream]) -> None:
    p = jnp.array([1 + 0j, 0 + 2j, -1 + 1j])
    stream = streams["inference"]
    samples = jnp.stack([dropout(p, stream.key(), 0.3) for _ in range(2000)])
    assert jnp.allclose(jnp.mean(samples, axis=0), p, atol=0.1)


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


# ── LogSpaceProjection ────────────────────────────────────────────────────────────────────


def test_linear_output_shape(streams: Mapping[str, RngStream]) -> None:
    f = LogSpaceProjection.create(3, 5, streams=streams)
    assert f.project(jnp.ones(3, dtype=jnp.complex128)).shape == (5,)


def test_linear_output_dtype(streams: Mapping[str, RngStream]) -> None:
    assert (
        LogSpaceProjection.create(3, 5, streams=streams)
        .project(jnp.ones(3, dtype=jnp.complex128))
        .dtype
        == jnp.complex128
    )


def test_linear_batched_shape(streams: Mapping[str, RngStream]) -> None:
    f = LogSpaceProjection.create(3, 5, streams=streams)
    assert f.project(jnp.ones((4, 3), dtype=jnp.complex128)).shape == (4, 5)


def test_linear_zero_input_is_finite_and_bounded(streams: Mapping[str, RngStream]) -> None:
    f = LogSpaceProjection.create(3, 5, streams=streams)
    out = f.project(jnp.zeros(3, dtype=jnp.complex128))
    assert jnp.all(jnp.isfinite(out))
    assert jnp.all(jnp.abs(out) <= 1.0)


def test_linear_phase_scales_log_domain_phase() -> None:
    f = LogSpaceProjection(
        weight=LearnableParameter(jnp.eye(2, dtype=jnp.complex128)),
        phase_scales=LearnableParameter(jnp.array([2.0, 0.5], dtype=jnp.float64)),
    )
    theta = jnp.array([0.2, -0.4], dtype=jnp.float64)
    z = jnp.exp(1j * theta)
    assert jnp.allclose(f.project(z), jnp.exp(1j * jnp.array([0.4, -0.2])), atol=1e-7)


def test_linear_weight_shape(streams: Mapping[str, RngStream]) -> None:
    f = LogSpaceProjection.create(3, 5, streams=streams)
    assert f.weight.value.shape == (5, 3)
    assert f.phase_scales.value.shape == (5,)


# ── LogSpaceProjectionWithDropout ─────────────────────────────────────────────────────────


def test_linear_with_dropout_output_shape(streams: Mapping[str, RngStream]) -> None:
    f = LogSpaceProjectionWithDropout.create(3, 5, streams=streams)
    assert f.infer(jnp.ones(3, dtype=jnp.complex128), streams=streams, inference=True).shape == (5,)


def test_linear_with_dropout_zero_rate_matches_linear_inference(
    streams: Mapping[str, RngStream],
) -> None:
    f = LogSpaceProjectionWithDropout.create(3, 5, dropout_rate=0.0, streams=streams)
    assert jnp.allclose(
        f.infer(jnp.zeros(3, dtype=jnp.complex128), streams=streams, inference=False),
        LogSpaceProjection.project(f, jnp.zeros(3, dtype=jnp.complex128)),
    )


def test_linear_with_dropout_skips_dropout_when_inference_true(
    streams: Mapping[str, RngStream],
) -> None:
    # inference=True: output matches the raw affine transform (no dropout applied).
    f = LogSpaceProjectionWithDropout.create(3, 5, dropout_rate=0.9, streams=streams)
    z = jnp.ones(3, dtype=jnp.complex128)
    assert jnp.allclose(
        f.infer(z, streams=streams, inference=True), LogSpaceProjection.project(f, z)
    )


def test_linear_with_dropout_applies_dropout_when_inference_false(
    streams: Mapping[str, RngStream],
) -> None:
    # inference=False: at least one output differs from the no-dropout result (inference=True).
    f = LogSpaceProjectionWithDropout.create(3, 20, dropout_rate=0.5, streams=streams)
    z = jnp.ones(3, dtype=jnp.complex128)
    out_train = f.infer(z, streams=streams, inference=False)
    out_eval = f.infer(z, streams=streams, inference=True)
    assert not jnp.allclose(out_train, out_eval)


# ── MobiusSummation ───────────────────────────────────────────────────────────


def test_mobius_summation_output_shape(streams: Mapping[str, RngStream]) -> None:
    f = MobiusSummation.create(4, 6, streams=streams)
    assert f.sum(jnp.ones((5, 4), dtype=jnp.complex128)).shape == (5, 6)


def test_mobius_summation_full_participation_reduces_to_parallel_sum() -> None:
    presence = jnp.array([0.5, 2.0])
    phase = jnp.array([0.2, -0.4])
    x = presence * jnp.exp(1j * phase)
    result = mobius_sum(
        x,
        jnp.ones((1, 2)),
        jnp.ones((1, 2)),
    )
    expected_presence = 1 / jnp.sum(1 / presence)
    expected = expected_presence * jnp.exp(1j * jnp.sum(phase))
    assert jnp.allclose(result, expected)


def test_mobius_summation_soft_single_input() -> None:
    x = jnp.array([0.7j])
    participation = 0.25
    result = mobius_sum(
        x,
        jnp.ones((1, 1)),
        jnp.array([[participation]]),
    )
    expected = (
        participation * jnp.abs(x[0]) * ((1 - participation) + participation * x[0] / jnp.abs(x[0]))
    )
    assert jnp.allclose(result, expected)


def test_mobius_summation_zero_input_is_zero_and_finite() -> None:
    result = mobius_sum(
        jnp.zeros(3, dtype=jnp.complex128),
        jnp.ones((2, 3)),
        jnp.full((2, 3), 0.5),
    )
    assert jnp.all(result == 0)
    assert jnp.all(jnp.isfinite(result))


def test_mobius_summation_is_continuous_across_phase_branch_cut() -> None:
    epsilon = 1e-8
    x = jnp.exp(1j * jnp.array([jnp.pi - epsilon, -jnp.pi + epsilon]))[:, jnp.newaxis]
    result = mobius_sum(
        x,
        jnp.array([[0.5]]),
        jnp.ones((1, 1)),
    )
    assert jnp.allclose(result[0], result[1], atol=1e-7)


def test_mobius_summation_multiplies_weights_under_composition() -> None:
    x = jnp.exp(1j * jnp.array([0.7]))
    weight = jnp.array([[0.3]])
    other_weight = jnp.array([[-0.5]])
    full_participation = jnp.ones((1, 1))
    sequential = mobius_sum(
        mobius_sum(x, other_weight, full_participation),
        weight,
        full_participation,
    )
    combined = mobius_sum(x, weight * other_weight, full_participation)
    assert jnp.allclose(sequential, combined)


def test_low_rank_mobius_summation_output_shape(streams: Mapping[str, RngStream]) -> None:
    f = LowRankMobiusSummation.create(4, 6, 2, streams=streams)
    assert f.sum(jnp.ones((5, 4), dtype=jnp.complex128)).shape == (5, 6)


def test_mobius_summation_reversal_endpoints() -> None:
    x = jnp.exp(1j * jnp.array([0.3, -0.8]))
    weights = jnp.array([[0.2, 0.7]])
    full_presence = jnp.ones((1, 2))
    positive = mobius_sum(x, weights, full_presence)
    negative = mobius_sum(x, -weights, full_presence)

    assert jnp.allclose(negative, jnp.conj(positive))


def test_phase_warp_zero_weight_is_finite_at_negative_one() -> None:
    assert jnp.allclose(
        phase_warp(jnp.array([-1.0 + 0.0j]), jnp.array([0.0])),
        jnp.array([1.0 + 0.0j]),
    )


# ── FrequencyAdaptedEvidencePooling ──────────────────────────────────────────


def test_frequency_adapted_pooling_output_shape(streams: Mapping[str, RngStream]) -> None:
    frequencies = jnp.array([1.0, 0.5, 0.25])
    f = FrequencyAdaptedEvidencePooling.create(4, 6, frequencies, streams=streams)
    assert f.project(jnp.ones((5, 12), dtype=jnp.complex128)).shape == (5, 18)


def test_frequency_adapted_pooling_scales_rotation_by_frequency() -> None:
    frequencies = jnp.array([1.0, 0.5])
    displacement = 0.8
    f = FrequencyAdaptedEvidencePooling(
        log_gains=LearnableParameter(jnp.zeros((1, 1))),
        displacements=LearnableParameter(jnp.array([[displacement]])),
        frequencies=FixedParameter(frequencies),
    )
    result = f.project(jnp.ones(2, dtype=jnp.complex128))
    assert jnp.allclose(result, jnp.exp(1j * frequencies * displacement))


def test_low_rank_frequency_adapted_pooling_output_shape(
    streams: Mapping[str, RngStream],
) -> None:
    frequencies = jnp.array([1.0, 0.5, 0.25])
    f = LowRankFrequencyAdaptedEvidencePooling.create(4, 6, 2, frequencies, streams=streams)
    assert f.project(jnp.ones((5, 12), dtype=jnp.complex128)).shape == (5, 18)


# ── FrequencyElementwiseRotation ─────────────────────────────────────────────


def test_elementwise_rotation_starts_as_identity(streams: Mapping[str, RngStream]) -> None:
    f = ElementwiseRotation.create(4, streams=streams)
    x = jnp.arange(4, dtype=jnp.float64).astype(jnp.complex128)
    assert jnp.allclose(f.rotate(x), x)


def test_elementwise_rotation_preserves_presence() -> None:
    f = ElementwiseRotation(displacements=LearnableParameter(jnp.array([0.8, -0.4, 0.2, 1.1])))
    x = jnp.arange(1, 5, dtype=jnp.float64).astype(jnp.complex128)
    assert jnp.allclose(jnp.abs(f.rotate(x)), jnp.abs(x))


def test_frequency_elementwise_rotation_starts_as_identity(
    streams: Mapping[str, RngStream],
) -> None:
    frequencies = jnp.array([1.0, 0.5, 0.25])
    f = FrequencyElementwiseRotation.create(4, frequencies, streams=streams)
    x = jnp.arange(12, dtype=jnp.float64).reshape(3, 4).T.reshape(-1).astype(jnp.complex128)
    assert jnp.allclose(f.rotate(x), x)


def test_frequency_elementwise_rotation_scales_phase_by_frequency() -> None:
    frequencies = jnp.array([1.0, 0.5])
    displacements = jnp.array([0.8, -0.4])
    f = FrequencyElementwiseRotation(
        displacements=LearnableParameter(displacements),
        frequencies=FixedParameter(frequencies),
    )
    result = f.rotate(jnp.ones(4, dtype=jnp.complex128)).reshape(2, 2)
    expected = jnp.exp(1j * displacements[:, jnp.newaxis] * frequencies[jnp.newaxis, :])
    assert jnp.allclose(result, expected)


def test_frequency_elementwise_rotation_preserves_presence(
    streams: Mapping[str, RngStream],
) -> None:
    frequencies = jnp.array([1.0, 0.5, 0.25])
    f = FrequencyElementwiseRotation.create(4, frequencies, streams=streams)
    f = FrequencyElementwiseRotation(
        displacements=LearnableParameter(jnp.array([0.8, -0.4, 0.2, 1.1])),
        frequencies=f.frequencies,
    )
    x = jnp.arange(1, 13, dtype=jnp.float64).astype(jnp.complex128)
    assert jnp.allclose(jnp.abs(f.rotate(x)), jnp.abs(x))


def test_frequency_phase_activation_depends_only_on_phase() -> None:
    frequencies = jnp.array([1.0])
    activation = FrequencyPhaseActivation(
        rotation=FrequencyElementwiseRotation(
            displacements=LearnableParameter(jnp.zeros(1)),
            frequencies=FixedParameter(frequencies),
        ),
        log_sharpnesses=LearnableParameter(jnp.log(jnp.array([2.0]))),
    )
    phase = 0.6
    x = jnp.array([[jnp.exp(1j * phase)], [3 * jnp.exp(1j * phase)]])

    result = activation.activate(x)
    attenuation = jnp.abs(result) / jnp.abs(x)

    assert jnp.allclose(attenuation[0], attenuation[1])
    assert jnp.all(attenuation > 0)


def test_frequency_phase_activation_uses_coherent_value_rotation() -> None:
    frequencies = jnp.array([1.0, 0.5, 0.25])
    value = 0.8
    activation = FrequencyPhaseActivation(
        rotation=FrequencyElementwiseRotation(
            displacements=LearnableParameter(jnp.array([-value])),
            frequencies=FixedParameter(frequencies),
        ),
        log_sharpnesses=LearnableParameter(jnp.zeros(1)),
    )
    x = jnp.exp(1j * frequencies * value)

    result = activation.activate(x)

    assert jnp.allclose(result, jnp.ones_like(result))


# ── GatedProjection ───────────────────────────────────────────────────────────


def test_value_projection_output_shape(streams: Mapping[str, RngStream]) -> None:
    f = ValueProjection.create(4, streams=streams)
    assert f.infer(
        jnp.ones((5, 4), dtype=jnp.complex128), streams=streams, inference=True
    ).shape == (5, 4)


def test_value_projection_preserves_presence(streams: Mapping[str, RngStream]) -> None:
    f = ValueProjection.create(4, streams=streams)
    z = jnp.array([0.0 + 0.0j, 0.2 + 0.1j, -1.0 + 2.0j, 3.0 - 4.0j])
    projected = f.infer(z, streams=streams, inference=True)
    assert jnp.allclose(jnp.abs(projected), jnp.abs(z))
    assert jnp.all(jnp.isfinite(projected))


def test_gated_projection_output_shape(streams: Mapping[str, RngStream]) -> None:
    f = GatedProjection.create(4, 6, streams=streams)
    assert isinstance(f.value, MobiusSummation)
    assert f.infer(jnp.ones(4, dtype=jnp.complex128), streams=streams, inference=True).shape == (6,)


def test_gated_projection_output_dtype(streams: Mapping[str, RngStream]) -> None:
    f = GatedProjection.create(4, 6, streams=streams)
    assert (
        f.infer(jnp.ones(4, dtype=jnp.complex128), streams=streams, inference=True).dtype
        == jnp.complex128
    )


def test_gated_projection_batched_shape(streams: Mapping[str, RngStream]) -> None:
    f = GatedProjection.create(4, 6, streams=streams)
    assert f.infer(
        jnp.ones((5, 4), dtype=jnp.complex128), streams=streams, inference=True
    ).shape == (5, 6)


def test_gated_projection_custom_mid_features(streams: Mapping[str, RngStream]) -> None:
    f = GatedProjection.create(4, 6, mid_features=8, streams=streams)
    assert f.infer(jnp.ones(4, dtype=jnp.complex128), streams=streams, inference=True).shape == (6,)


def test_frequency_gated_projection_output_shape(streams: Mapping[str, RngStream]) -> None:
    frequencies = jnp.array([1.0, 0.5, 0.25])
    f = FrequencyGatedProjection.create(
        4,
        2,
        frequencies,
        hidden_features=6,
        mobius_rank=2,
        streams=streams,
    )
    assert isinstance(f.value, LowRankMobiusSummation)
    result = f.infer(jnp.ones((5, 12), dtype=jnp.complex128), streams=streams, inference=True)
    assert result.shape == (5, 6)


# ── RecurrentPhaseFocusing ────────────────────────────────────────────────────


def test_recurrent_phase_focusing_output_shape(streams: Mapping[str, RngStream]) -> None:
    f = RecurrentPhaseFocusing.create(
        4,
        2,
        hidden_features=6,
        iterations=3,
        streams=streams,
    )
    z = jnp.exp(1j * jnp.linspace(-1.0, 1.0, 20).reshape(5, 4))

    assert f.infer(z, streams=streams, inference=True).shape == (5, 2)


def test_recurrent_phase_focus_preserves_input_presence(
    streams: Mapping[str, RngStream],
) -> None:
    f = RecurrentPhaseFocusing.create(3, 1, hidden_features=4, streams=streams)
    z = jnp.array([0.2 + 0.1j, -1.0 + 2.0j, 3.0 - 4.0j])
    focused = f.focus(
        z,
        rotations=jnp.array([0.1, -0.2, 0.3]),
        weights=jnp.array([0.4, -0.5, 0.0]),
    )

    assert jnp.allclose(jnp.abs(focused), jnp.abs(z))


def test_recurrent_phase_focusing_preserves_absence(
    streams: Mapping[str, RngStream],
) -> None:
    f = RecurrentPhaseFocusing.create(3, 2, hidden_features=4, streams=streams)
    z = jnp.zeros(3, dtype=jnp.complex128)

    assert jnp.allclose(f.infer(z, streams=streams, inference=True), 0)


@pytest.mark.parametrize("parameterization", list(MobiusParameterization))
def test_frequency_mobius_projection_output_shape(
    streams: Mapping[str, RngStream],
    *,
    parameterization: MobiusParameterization,
) -> None:
    frequencies = jnp.array([1.0, 0.5, 0.25])
    f = FrequencyMobiusProjection.create(
        4,
        2,
        frequencies,
        hidden_features=6,
        mobius_rank=2,
        parameterization=parameterization,
        streams=streams,
    )
    result = f.infer(jnp.ones((5, 12), dtype=jnp.complex128), streams=streams, inference=True)
    assert result.shape == (5, 6)


def test_frequency_mobius_projection_supports_phase_activation(
    streams: Mapping[str, RngStream],
) -> None:
    frequencies = jnp.array([1.0, 0.5, 0.25])
    f = FrequencyMobiusProjection.create(
        4,
        2,
        frequencies,
        hidden_features=6,
        parameterization=MobiusParameterization.phase_activated_dense,
        streams=streams,
    )

    assert isinstance(f.activation, FrequencyPhaseActivation)
    result = f.infer(jnp.ones((5, 12), dtype=jnp.complex128), streams=streams, inference=True)
    assert result.shape == (5, 6)


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
