from collections.abc import Mapping
from typing import TypedDict

import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
from efax import NormalNP
from tjax import JaxRealArray, KeyArray, RngStream

from cem.npn.gaussian import normal_from_mean_and_presence
from cem.sip.features import (
    FeatureGain,
    FeaturePredictors,
    SIPFeature,
    SIPFeatureSizes,
    TDFeature,
)


class _SIPStepInputs(TypedDict):
    key: KeyArray
    observation_content: NormalNP
    predictors: FeaturePredictors
    gain: FeatureGain
    goal: JaxRealArray


class _TDStepInputs(TypedDict):
    reward_observation: NormalNP
    predictors: FeaturePredictors


def _message(mean: jnp.ndarray, precision: jnp.ndarray) -> NormalNP:
    return normal_from_mean_and_presence(mean, precision)


def _sip_feature(
    n: int, streams: Mapping[str, RngStream], *, controllable: bool = True
) -> SIPFeature:
    sizes = SIPFeatureSizes(
        observation_content=1,
        predictor_observations=2,
        predictor_instruments=2,
        gain_instrument=1,
    )
    return SIPFeature.create(n, sizes, controllable=controllable, streams=streams)


def _sip_inputs(n: int) -> _SIPStepInputs:
    return _SIPStepInputs(
        key=jr.key(0),
        observation_content=_message(jnp.array([0.3]), jnp.ones(1)),
        predictors=FeaturePredictors(
            observations=_message(jnp.array([0.3, -0.2]), jnp.ones(2)),
            instruments=_message(jnp.array([0.5, 0.1]), jnp.ones(2)),
        ),
        gain=FeatureGain(
            observation=_message(jnp.ones(n), jnp.ones(n)),
            instrument=_message(jnp.array([0.4]), jnp.ones(1)),
        ),
        goal=jnp.asarray(0.5),
    )


def _td_feature(n: int, streams: Mapping[str, RngStream], *, discount: float = 1.0) -> TDFeature:
    return TDFeature.create(
        n,
        predictor_observation_features=2,
        predictor_instrument_features=2,
        discount=discount,
        streams=streams,
    )


def _td_inputs(n: int) -> _TDStepInputs:
    return _TDStepInputs(
        reward_observation=_message(jnp.ones(n), jnp.ones(n)),
        predictors=FeaturePredictors(
            observations=_message(jnp.array([0.3, -0.2]), jnp.ones(2)),
            instruments=_message(jnp.array([0.5, 0.1]), jnp.ones(2)),
        ),
    )


def test_sip_feature_outputs_have_expected_shapes(
    streams: Mapping[str, RngStream],
) -> None:
    n = 2
    feature = _sip_feature(n, streams)
    state = eqx.nn.State(feature)
    _, outputs = feature.step(state, **_sip_inputs(n), streams=streams, inference=True)
    assert outputs.observation.mean_times_precision.shape == (n,)
    assert outputs.instrument.mean_times_precision.shape == (n,)
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
        predictor_instruments=inputs["predictors"].instruments,
        gain_observation=inputs["gain"].observation,
        goal=inputs["goal"],
    )
    mismatch_outputs = feature.mismatch.step(
        raw_observation=emitter_outputs.raw_observation,
        local_variance=emitter_outputs.local_variance,
        predictor_observations=inputs["predictors"].observations,
        predictor_instruments=inputs["predictors"].instruments,
        gain_observation=inputs["gain"].observation,
        gain_instrument=inputs["gain"].instrument,
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
        predictor_observations=inputs["predictors"].observations,
        predictor_instruments=inputs["predictors"].instruments,
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
        stored.predictor_observations.mean_times_precision,
        inputs["predictors"].observations.mean_times_precision,
    )
