from collections.abc import Mapping
from typing import TypedDict

import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
from efax import NormalNP
from tjax import JaxRealArray, KeyArray, RngStream

from cem.framework.graph import LearnableParameter
from cem.sip.features import (
    RecruiterFeature,
    SensationFeature,
    SensationInputFeature,
    SIPFeatureSizes,
    TDFeature,
)
from cem.sip.instrument import inherit_instrument, instrument_loss
from cem.sip.messages import InstrumentedMessage
from cem.sip.settings import TDSettings

from .tools import MISMATCH_SETTINGS, _clone, _message


class _SIPStepInputs(TypedDict):
    key: KeyArray
    observation_content: NormalNP
    predictor: InstrumentedMessage
    gain: InstrumentedMessage
    reconstruction: JaxRealArray
    confounding: JaxRealArray


class _TDStepInputs(TypedDict):
    reward_observation: NormalNP
    predictor: InstrumentedMessage


def _sip_feature(
    n: int, streams: Mapping[str, RngStream], *, controllable: bool = True
) -> SensationFeature:
    sizes = SIPFeatureSizes(
        observation_content=1,
        predictor=2,
        gain=1,
    )
    return SensationFeature.create(
        n,
        sizes,
        controllable=controllable,
        mismatch_settings=MISMATCH_SETTINGS,
        streams=streams,
    )


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
        reconstruction=jnp.asarray(0.5),
        confounding=jnp.asarray(0.2),
    )


def _td_feature(n: int, streams: Mapping[str, RngStream], *, discount: float = 1.0) -> TDFeature:
    return TDFeature.create(
        n,
        predictor_features=2,
        td_settings=TDSettings(discount=discount),
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
    assert outputs.reconstruction_loss.shape == ()
    assert outputs.confounding_objective.shape == ()
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
        goal=-(inputs["reconstruction"] + inputs["confounding"]),
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


def test_sensation_credit_lowers_dissonance(streams: Mapping[str, RngStream]) -> None:
    """Higher dissonance pushes the observation map against the recorded experiments.

    Raising the reconstruction loss by one lowers the emitter's goal by one, so a descent step
    moves the parameters opposite the mean and precision traces. The mismatch circuit holds the
    observation fixed, so only credit depends on the reconstruction loss.
    """
    n = 2
    feature = _sip_feature(n, streams, controllable=False)
    # The first step records the experiments that the second step's dissonance credits.
    state, _ = feature.step(
        eqx.nn.State(feature), **_sip_inputs(n), streams=streams, inference=True
    )
    lag_filter = feature.emitter.credit.lag_filter.value
    traces = state.get(feature.emitter.trace_index)
    credit_trace = (
        lag_filter.value(traces.mean).layers[0].weight.value
        + lag_filter.value(traces.precision).layers[0].weight.value
    )

    def weight_gradient(reconstruction: float) -> JaxRealArray:
        def loss_fn(model: SensationFeature) -> JaxRealArray:
            inputs = _sip_inputs(n)
            inputs["reconstruction"] = jnp.asarray(reconstruction)
            _, outputs = model.step(_clone(state), **inputs, streams=streams, inference=True)
            return outputs.loss

        grads = eqx.filter_grad(loss_fn)(feature)
        return grads.emitter.observation_map.layers[0].weight.value

    assert jnp.any(jnp.abs(credit_trace) > 0)
    assert jnp.allclose(weight_gradient(1.0) - weight_gradient(0.0), credit_trace)


def test_sip_feature_state_retains_its_emission(
    streams: Mapping[str, RngStream],
) -> None:
    """After a step, the credit circuit retains that step's emission for the next goal."""
    n = 2
    feature = _sip_feature(n, streams)
    state = eqx.nn.State(feature)
    new_state, _ = feature.step(state, **_sip_inputs(n), streams=streams, inference=True)
    credit = feature.emitter.credit
    retained = new_state.get(credit.state_index)
    assert jnp.allclose(credit.lag_filter.value.value(retained.emission_weight), 1.0)


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
        mismatch_settings=MISMATCH_SETTINGS,
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
    observation_weight = feature.feature.emitter.observation_map.layers[0].weight
    assert isinstance(observation_weight, LearnableParameter)
    state = eqx.nn.State(feature)
    _, outputs = feature.step(
        state,
        jr.key(0),
        observation_content=_message(jnp.asarray([0.3]), jnp.ones(1)),
        predictor=InstrumentedMessage(
            observation=_message(jnp.asarray([0.3, -0.2]), jnp.ones(2)),
            instrument=_message(jnp.asarray([0.5, 0.1]), jnp.ones(2)),
        ),
        reconstruction=jnp.asarray(0.5),
        confounding=jnp.asarray(0.2),
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
        goal=jnp.asarray(0.2) - jnp.asarray(0.5),
    )
    assert jnp.allclose(
        outputs.message.observation.mean_times_precision,
        emitter_outputs.message.observation.mean_times_precision,
    )


def test_recruiter_feature_witness_input_omits_gain(
    streams: Mapping[str, RngStream],
) -> None:
    """The recruiter's witness maps take 2 * predictor_features inputs only."""
    n = 2
    feature = _recruiter_feature(n, streams)
    mismatch = feature.feature.mismatch
    # predictor_features=2 → witness input dim = 2 * 2 (no gain concat).
    expected_input_dim = 2 * 2
    for witness_map in (mismatch.integral_witness_map, mismatch.proportional_witness_map):
        first_layer = witness_map.perceptron.layers[0]
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
        reconstruction=jnp.asarray(0.5),
        confounding=jnp.asarray(0.2),
        streams=streams,
        inference=True,
    )
    assert jnp.isfinite(outputs.loss)
    assert outputs.message.observation.mean_times_precision.shape == (n,)


def _sensation_input(n: int, streams: Mapping[str, RngStream]) -> SensationInputFeature:
    return SensationInputFeature.create(
        n, predictor_features=2, mismatch_settings=MISMATCH_SETTINGS, streams=streams
    )


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


def test_sensation_input_reports_prediction_costs(
    streams: Mapping[str, RngStream],
) -> None:
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
    assert jnp.allclose(outputs.reconstruction_loss, mismatch_outputs.reconstruction_loss)
    assert jnp.allclose(outputs.confounding_objective, mismatch_outputs.confounding_objective)
    inherited = inherit_instrument(feature.instrument_map, predictor.instrument)
    expected_instrument_loss = instrument_loss(inherited, observation, None)
    assert jnp.allclose(outputs.loss, mismatch_outputs.loss + expected_instrument_loss)
