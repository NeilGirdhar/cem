from collections.abc import Mapping
from typing import TypedDict

import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
from efax import NormalNP
from tjax import JaxRealArray, KeyArray, RngStream

from cem.sip.features import (
    RecruiterFeature,
    SensationInputFeature,
    SIPFeature,
    SIPFeatureSizes,
    TDFeature,
)
from cem.sip.messages import InstrumentedMessage

from .tools import _message


class _SIPStepInputs(TypedDict):
    key: KeyArray
    observation_content: NormalNP
    predictor: InstrumentedMessage
    gain: InstrumentedMessage
    goal: JaxRealArray


class _TDStepInputs(TypedDict):
    reward_observation: NormalNP
    predictor: InstrumentedMessage


def _sip_feature(
    n: int, streams: Mapping[str, RngStream], *, controllable: bool = True
) -> SIPFeature:
    sizes = SIPFeatureSizes(
        observation_content=1,
        predictor=2,
        gain=1,
    )
    return SIPFeature.create(n, sizes, controllable=controllable, streams=streams)


def _sip_inputs(n: int) -> _SIPStepInputs:
    return _SIPStepInputs(
        key=jr.key(0),
        observation_content=_message(jnp.asarray([0.3]), jnp.ones(1)),
        predictor=InstrumentedMessage(
            observation=_message(jnp.asarray([0.3, -0.2]), jnp.ones(2)),
            instrument=_message(jnp.asarray([0.5, 0.1]), jnp.ones(2)),
        ),
        gain=InstrumentedMessage(
            observation=_message(jnp.ones(n), jnp.ones(n)),
            instrument=_message(jnp.asarray([0.4]), jnp.ones(1)),
        ),
        goal=jnp.asarray(0.5),
    )


def _td_feature(n: int, streams: Mapping[str, RngStream], *, discount: float = 1.0) -> TDFeature:
    return TDFeature.create(
        n,
        predictor_features=2,
        discount=discount,
        streams=streams,
    )


def _td_inputs(n: int) -> _TDStepInputs:
    return _TDStepInputs(
        reward_observation=_message(jnp.ones(n), jnp.ones(n)),
        predictor=InstrumentedMessage(
            observation=_message(jnp.asarray([0.3, -0.2]), jnp.ones(2)),
            instrument=_message(jnp.asarray([0.5, 0.1]), jnp.ones(2)),
        ),
    )


def test_sip_feature_outputs_have_expected_shapes(
    streams: Mapping[str, RngStream],
) -> None:
    n = 2
    feature = _sip_feature(n, streams)
    state = eqx.nn.State(feature)
    _, outputs = feature.step(state, **_sip_inputs(n), streams=streams, inference=True)
    assert outputs.message.observation.mean_times_precision.shape == (n,)
    assert outputs.message.instrument.mean_times_precision.shape == (n,)
    assert outputs.observation_mismatch.mean_times_precision.shape == (2 * n,)
    assert outputs.reconstruction_cost.shape == ()
    assert outputs.confounding_cost.shape == ()
    assert outputs.loss.shape == ()


def test_sip_feature_loss_sums_emitter_and_mismatch(
    streams: Mapping[str, RngStream],
) -> None:
    """Combined loss equals emitter.loss + mismatch.loss computed from the sub-parts."""
    n = 2
    feature = _sip_feature(n, streams)
    state = eqx.nn.State(feature)
    inputs = _sip_inputs(n)
    _, outputs = feature.step(state, **inputs, streams=streams, inference=True)

    # Rerun sub-parts the way the feature does.
    fresh_state = eqx.nn.State(feature)
    _, emitter_outputs = feature.emitter.step(
        fresh_state,
        inputs["key"],
        observation_content=inputs["observation_content"],
        predictor_instrument=inputs["predictor"].instrument,
        gain_observation=inputs["gain"].observation,
        goal=inputs["goal"],
    )
    mismatch_outputs = feature.mismatch.step(
        raw_observation=emitter_outputs.raw_observation,
        local_variance=emitter_outputs.local_variance,
        predictor=inputs["predictor"],
        gain=inputs["gain"],
        streams=streams,
        inference=True,
    )
    assert jnp.allclose(outputs.loss, emitter_outputs.loss + mismatch_outputs.loss)


def test_sip_feature_state_carries_credit_traces(
    streams: Mapping[str, RngStream],
) -> None:
    """After a step, the new state's credit traces are nonzero (updated from zero init)."""
    n = 2
    feature = _sip_feature(n, streams)
    state = eqx.nn.State(feature)
    new_state, _ = feature.step(state, **_sip_inputs(n), streams=streams, inference=True)
    traces = new_state.get(feature.emitter.credit.trace_index)
    assert jnp.any(jnp.abs(traces.linear) > 0) or jnp.any(jnp.abs(traces.quadratic) > 0)


def test_td_feature_observation_is_reward_passthrough(
    streams: Mapping[str, RngStream],
) -> None:
    n = 2
    feature = _td_feature(n, streams)
    state = eqx.nn.State(feature)
    inputs = _td_inputs(n)
    _, outputs = feature.step(state, **inputs, streams=streams, inference=True)
    assert jnp.allclose(
        outputs.observation.mean_times_precision,
        inputs["reward_observation"].mean_times_precision,
    )
    assert jnp.allclose(
        outputs.observation.negative_half_precision,
        inputs["reward_observation"].negative_half_precision,
    )


def test_td_feature_loss_equals_td_error_loss(
    streams: Mapping[str, RngStream],
) -> None:
    n = 2
    feature = _td_feature(n, streams)
    state = eqx.nn.State(feature)
    inputs = _td_inputs(n)

    _, feature_outputs = feature.step(state, **inputs, streams=streams, inference=True)
    fresh_state = eqx.nn.State(feature)
    _, td_outputs = feature.td_error.step(
        fresh_state,
        reward_observation=inputs["reward_observation"],
        predictor=inputs["predictor"],
        streams=streams,
        inference=True,
    )
    assert jnp.allclose(feature_outputs.loss, td_outputs.loss)
    assert jnp.allclose(feature_outputs.advantage, td_outputs.advantage)


def test_td_feature_state_delays_predictor_inputs(
    streams: Mapping[str, RngStream],
) -> None:
    n = 2
    feature = _td_feature(n, streams)
    state = eqx.nn.State(feature)
    inputs = _td_inputs(n)
    new_state, _ = feature.step(state, **inputs, streams=streams, inference=True)
    stored = new_state.get(feature.td_error.delay_index)
    assert jnp.allclose(
        stored.predictor.observation.mean_times_precision,
        inputs["predictor"].observation.mean_times_precision,
    )


def _recruiter_feature(n: int, streams: Mapping[str, RngStream]) -> RecruiterFeature:
    return RecruiterFeature.create(
        n,
        observation_content_features=1,
        predictor_features=2,
        streams=streams,
    )


def test_recruiter_feature_observation_equals_pre_gain(
    streams: Mapping[str, RngStream],
) -> None:
    """With no gain parent the outgoing observation equals the noised raw observation.

    The emitter skips evidence scaling and the result should match running the
    emitter directly with `gain_observation=None`.
    """
    n = 2
    feature = _recruiter_feature(n, streams)
    state = eqx.nn.State(feature)
    _, outputs = feature.step(
        state,
        jr.key(0),
        observation_content=_message(jnp.asarray([0.3]), jnp.ones(1)),
        predictor=InstrumentedMessage(
            observation=_message(jnp.asarray([0.3, -0.2]), jnp.ones(2)),
            instrument=_message(jnp.asarray([0.5, 0.1]), jnp.ones(2)),
        ),
        goal=jnp.asarray(0.5),
        streams=streams,
        inference=True,
    )
    fresh_state = eqx.nn.State(feature)
    _, emitter_outputs = feature.feature.emitter.step(
        fresh_state,
        jr.key(0),
        observation_content=_message(jnp.asarray([0.3]), jnp.ones(1)),
        predictor_instrument=_message(jnp.asarray([0.5, 0.1]), jnp.ones(2)),
        gain_observation=None,
        goal=jnp.asarray(0.5),
    )
    assert jnp.allclose(
        outputs.message.observation.mean_times_precision,
        emitter_outputs.message.observation.mean_times_precision,
    )


def test_recruiter_feature_witness_input_omits_gain(
    streams: Mapping[str, RngStream],
) -> None:
    """The recruiter's witness map input dim is 2 * predictor_features only."""
    n = 2
    feature = _recruiter_feature(n, streams)
    first_layer = feature.feature.mismatch.witness_map.layers[0]
    # predictor_features=2 → witness input dim = 2 * 2 (no gain concat).
    expected_input_dim = 2 * 2
    assert first_layer.weight.value.shape[-1] == expected_input_dim


def test_recruiter_feature_step_runs_without_gain(
    streams: Mapping[str, RngStream],
) -> None:
    """Smoke test: a full recruiter step produces finite outputs with no gain."""
    n = 2
    feature = _recruiter_feature(n, streams)
    state = eqx.nn.State(feature)
    _, outputs = feature.step(
        state,
        jr.key(0),
        observation_content=_message(jnp.asarray([0.3]), jnp.ones(1)),
        predictor=InstrumentedMessage(
            observation=_message(jnp.asarray([0.3, -0.2]), jnp.ones(2)),
            instrument=_message(jnp.asarray([0.5, 0.1]), jnp.ones(2)),
        ),
        goal=jnp.asarray(0.5),
        streams=streams,
        inference=True,
    )
    assert jnp.isfinite(outputs.loss)
    assert outputs.message.observation.mean_times_precision.shape == (n,)


def _sensation_input(n: int, streams: Mapping[str, RngStream]) -> SensationInputFeature:
    return SensationInputFeature.create(n, predictor_features=2, streams=streams)


def test_sensation_input_observation_pass_through(
    streams: Mapping[str, RngStream],
) -> None:
    """With no gain parent the outgoing observation equals the environment observation."""
    n = 3
    feature = _sensation_input(n, streams)
    state = eqx.nn.State(feature)
    observation = _message(jnp.asarray([0.1, 0.2, 0.3]), jnp.ones(n))
    _, outputs = feature.step(
        state,
        observation=observation,
        predictor=InstrumentedMessage(
            observation=_message(jnp.asarray([0.3, -0.2]), jnp.ones(2)),
            instrument=_message(jnp.asarray([0.5, 0.1]), jnp.ones(2)),
        ),
        gain=None,
        streams=streams,
        inference=True,
    )
    assert jnp.allclose(
        outputs.message.observation.mean_times_precision, observation.mean_times_precision
    )
    assert jnp.allclose(
        outputs.message.observation.negative_half_precision, observation.negative_half_precision
    )


def test_sip_feature_stores_message_into_state(
    streams: Mapping[str, RngStream],
) -> None:
    """After a step `last_message_index` holds the full InstrumentedMessage."""
    n = 2
    feature = _sip_feature(n, streams)
    state = eqx.nn.State(feature)
    new_state, outputs = feature.step(state, **_sip_inputs(n), streams=streams, inference=True)
    stored = new_state.get(feature.last_message_index)
    assert jnp.allclose(
        stored.observation.mean_times_precision, outputs.message.observation.mean_times_precision
    )
    assert jnp.allclose(
        stored.instrument.mean_times_precision, outputs.message.instrument.mean_times_precision
    )


def test_sensation_input_stores_message_into_state(
    streams: Mapping[str, RngStream],
) -> None:
    n = 3
    feature = _sensation_input(n, streams)
    state = eqx.nn.State(feature)
    observation = _message(jnp.asarray([0.1, 0.2, 0.3]), jnp.ones(n))
    new_state, outputs = feature.step(
        state,
        observation=observation,
        predictor=InstrumentedMessage(
            observation=_message(jnp.asarray([0.3, -0.2]), jnp.ones(2)),
            instrument=_message(jnp.asarray([0.5, 0.1]), jnp.ones(2)),
        ),
        gain=None,
        streams=streams,
        inference=True,
    )
    stored = new_state.get(feature.last_message_index)
    assert jnp.allclose(
        stored.observation.mean_times_precision, outputs.message.observation.mean_times_precision
    )
    assert jnp.allclose(
        stored.instrument.mean_times_precision, outputs.message.instrument.mean_times_precision
    )


def test_sensation_input_outputs_dissonance_from_mismatch(
    streams: Mapping[str, RngStream],
) -> None:
    """The feature's `dissonance` equals its mismatch's dissonance."""
    n = 3
    feature = _sensation_input(n, streams)
    state = eqx.nn.State(feature)
    predictor = InstrumentedMessage(
        observation=_message(jnp.asarray([0.3, -0.2]), jnp.ones(2)),
        instrument=_message(jnp.asarray([0.5, 0.1]), jnp.ones(2)),
    )
    observation = _message(jnp.asarray([0.1, 0.2, 0.3]), jnp.ones(n))
    _, outputs = feature.step(
        state,
        observation=observation,
        predictor=predictor,
        gain=None,
        streams=streams,
        inference=True,
    )
    mismatch_outputs = feature.mismatch.step(
        raw_observation=observation,
        local_variance=jnp.zeros(n),
        predictor=predictor,
        gain=None,
        streams=streams,
        inference=True,
    )
    assert jnp.allclose(outputs.dissonance, mismatch_outputs.dissonance)
    assert jnp.allclose(outputs.loss, mismatch_outputs.loss)
