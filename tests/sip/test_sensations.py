from collections.abc import Mapping

import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
from efax import NormalNP
from tjax import RngStream

from cem.npn.gaussian import normal_from_mean_and_presence
from cem.sip.messages import InstrumentedMessage
from cem.sip.sensations import Sensations


def _message(mean: jnp.ndarray, precision: jnp.ndarray) -> NormalNP:
    return normal_from_mean_and_presence(mean, precision)


def _predictors(size: int) -> InstrumentedMessage:
    return InstrumentedMessage(
        observation=_message(jnp.zeros(size), jnp.ones(size)),
        instrument=_message(jnp.zeros(size), jnp.ones(size)),
    )


def test_sensations_create_builds_stack_with_mismatch_cascade_sizes(
    streams: Mapping[str, RngStream],
) -> None:
    """Content dim at layer i > 0 equals 2 * sizes[i - 1] (mismatch cascade)."""
    sizes = [3, 2, 4]
    predictor_sizes = [2, 2, 2]
    observation_features = 5
    sensations = Sensations.create(
        sizes,
        observation_features=observation_features,
        predictor_features=predictor_sizes,
        streams=streams,
    )
    assert len(sensations.features) == len(sizes)
    expected_content_sizes = [observation_features, 2 * sizes[0], 2 * sizes[1]]
    for i, expected in enumerate(expected_content_sizes):
        weight = sensations.features[i].emitter.observation_map.layers[0].weight.value
        assert weight.shape[-1] == expected


def test_sensations_infer_returns_one_output_per_layer(
    streams: Mapping[str, RngStream],
) -> None:
    sizes = [2, 3]
    sensations = Sensations.create(
        sizes,
        observation_features=4,
        predictor_features=[1, 1],
        streams=streams,
    )
    state = eqx.nn.State(sensations)
    _, outputs = sensations.infer(
        state,
        jr.key(0),
        observation=_message(jnp.ones(4), jnp.ones(4)),
        predictors=[_predictors(1), _predictors(1)],
        goals=[jnp.asarray(0.0), jnp.asarray(0.0)],
        streams=streams,
        inference=True,
    )
    assert len(outputs) == len(sizes)
    # Each layer's observation has the layer's channel count.
    assert outputs[0].observation.mean_times_precision.shape == (sizes[0],)
    assert outputs[1].observation.mean_times_precision.shape == (sizes[1],)
    # Layer 1's observation_content is layer 0's mismatch; check that layer 0's
    # mismatch has the expected 2 * sizes[0] dim.
    assert outputs[0].observation_mismatch.mean_times_precision.shape == (2 * sizes[0],)


def test_sensations_infer_cascades_mismatch_as_next_content(
    streams: Mapping[str, RngStream],
) -> None:
    """Running layer 1 directly with layer 0's mismatch as content reproduces layer 1's output."""
    sizes = [2, 3]
    sensations = Sensations.create(
        sizes,
        observation_features=4,
        predictor_features=[1, 1],
        streams=streams,
    )
    key = jr.key(0)
    predictors = [_predictors(1), _predictors(1)]
    goals = [jnp.asarray(0.0), jnp.asarray(0.0)]
    observation = _message(jnp.ones(4), jnp.ones(4))

    state = eqx.nn.State(sensations)
    _, outputs = sensations.infer(
        state,
        key,
        observation=observation,
        predictors=predictors,
        goals=goals,
        streams=streams,
        inference=True,
    )

    # Re-run the same split manually.
    keys = jr.split(key, len(sizes))
    fresh_state = eqx.nn.State(sensations)
    fresh_state, layer0 = sensations.features[0].step(
        fresh_state,
        keys[0],
        observation_content=observation,
        predictors=predictors[0],
        gain=None,
        goal=goals[0],
        streams=streams,
        inference=True,
    )
    _, layer1 = sensations.features[1].step(
        fresh_state,
        keys[1],
        observation_content=layer0.observation_mismatch,
        predictors=predictors[1],
        gain=None,
        goal=goals[1],
        streams=streams,
        inference=True,
    )

    assert jnp.allclose(
        outputs[1].observation.mean_times_precision,
        layer1.observation.mean_times_precision,
    )
