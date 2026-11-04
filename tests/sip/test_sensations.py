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


def test_sensations_create_input_feature_has_no_observation_map(
    streams: Mapping[str, RngStream],
) -> None:
    """Layer 0 is a `SensationInputFeature` with no `observation_map` attribute."""
    sensations = Sensations.create(
        [3, 2, 4],
        predictor_features=[2, 2, 2],
        streams=streams,
    )
    assert not hasattr(sensations.input_feature, "observation_map")
    assert not hasattr(sensations.input_feature, "credit")


def test_sensations_create_interior_content_sizes_follow_mismatch_cascade(
    streams: Mapping[str, RngStream],
) -> None:
    """Interior layer i's observation_map input dim equals 2 * sizes[i - 1]."""
    sizes = [3, 2, 4]
    sensations = Sensations.create(
        sizes,
        predictor_features=[2, 2, 2],
        streams=streams,
    )
    assert len(sensations.interior_features) == len(sizes) - 1
    for i, feature in enumerate(sensations.interior_features, start=1):
        weight = feature.emitter.observation_map.layers[0].weight.value
        assert weight.shape[-1] == 2 * sizes[i - 1]


def test_sensations_infer_returns_one_output_per_layer(
    streams: Mapping[str, RngStream],
) -> None:
    sizes = [4, 3]
    sensations = Sensations.create(
        sizes,
        predictor_features=[1, 1],
        streams=streams,
    )
    state = eqx.nn.State(sensations)
    _, outputs = sensations.infer(
        state,
        jr.key(0),
        observation=_message(jnp.ones(sizes[0]), jnp.ones(sizes[0])),
        predictors=[_predictors(1), _predictors(1)],
        streams=streams,
        inference=True,
    )
    assert len(outputs) == len(sizes)
    assert outputs[0].observation.mean_times_precision.shape == (sizes[0],)
    assert outputs[1].observation.mean_times_precision.shape == (sizes[1],)
    assert outputs[0].observation_mismatch.mean_times_precision.shape == (2 * sizes[0],)


def test_sensations_input_feature_passes_observation_through_unchanged(
    streams: Mapping[str, RngStream],
) -> None:
    """With no gain parent, layer 0's output observation is the environment observation."""
    sizes = [4]
    sensations = Sensations.create(
        sizes,
        predictor_features=[1],
        streams=streams,
    )
    env_observation = _message(jnp.asarray([0.1, 0.2, 0.3, 0.4]), jnp.ones(sizes[0]))
    state = eqx.nn.State(sensations)
    _, outputs = sensations.infer(
        state,
        jr.key(0),
        observation=env_observation,
        predictors=[_predictors(1)],
        streams=streams,
        inference=True,
    )
    assert jnp.allclose(
        outputs[0].observation.mean_times_precision,
        env_observation.mean_times_precision,
    )
    assert jnp.allclose(
        outputs[0].observation.negative_half_precision,
        env_observation.negative_half_precision,
    )


def test_sensations_infer_feeds_dissonance_from_below_as_goal(
    streams: Mapping[str, RngStream],
) -> None:
    """Interior layer i's Credit goal equals the output dissonance of layer (i - 1)."""
    sizes = [2, 3]
    sensations = Sensations.create(
        sizes,
        predictor_features=[1, 1],
        streams=streams,
    )
    key = jr.key(0)
    predictors = [_predictors(1), _predictors(1)]
    observation = _message(jnp.ones(sizes[0]), jnp.ones(sizes[0]))

    state = eqx.nn.State(sensations)
    _, outputs = sensations.infer(
        state,
        key,
        observation=observation,
        predictors=predictors,
        streams=streams,
        inference=True,
    )

    # Re-run layer 0, then layer 1 with the explicit goal = outputs[0].dissonance.
    fresh = eqx.nn.State(sensations)
    fresh, layer0 = sensations.input_feature.step(
        fresh,
        observation=observation,
        predictors=predictors[0],
        gain=None,
        streams=streams,
        inference=True,
    )
    keys = jr.split(key, len(sensations.interior_features))
    _, layer1 = sensations.interior_features[0].step(
        fresh,
        keys[0],
        observation_content=layer0.observation_mismatch,
        predictors=predictors[1],
        gain=None,
        goal=jnp.asarray(layer0.dissonance),
        streams=streams,
        inference=True,
    )

    assert jnp.allclose(
        outputs[1].observation.mean_times_precision,
        layer1.observation.mean_times_precision,
    )
    assert jnp.allclose(
        outputs[1].dissonance,
        layer1.dissonance,
    )
