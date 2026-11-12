from collections.abc import Mapping

import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
from tjax import RngStream

from cem.sip.messages import InstrumentedMessage, concat_instrumented_messages
from cem.sip.sensations import Sensations

from .tools import CREDIT_SETTINGS, MISMATCH_SETTINGS, _message


def _predictors(size: int) -> InstrumentedMessage:
    return InstrumentedMessage.standard(size)


def test_sensations_create_input_feature_has_no_observation_map(
    streams: Mapping[str, RngStream],
) -> None:
    """Layer 0 is a `SensationInputFeature` with no `observation_map` attribute."""
    sensations = Sensations.create(
        [3, 2, 4],
        predictor_features=[2, 2, 2],
        credit_settings=CREDIT_SETTINGS,
        mismatch_settings=MISMATCH_SETTINGS,
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
        credit_settings=CREDIT_SETTINGS,
        mismatch_settings=MISMATCH_SETTINGS,
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
        credit_settings=CREDIT_SETTINGS,
        mismatch_settings=MISMATCH_SETTINGS,
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
    assert outputs[0].message.observation.mean_times_precision.shape == (sizes[0],)
    assert outputs[1].message.observation.mean_times_precision.shape == (sizes[1],)
    assert outputs[0].observation_mismatch.mean_times_precision.shape == (2 * sizes[0],)


def test_sensations_input_feature_passes_observation_through_unchanged(
    streams: Mapping[str, RngStream],
) -> None:
    """With no gain parent, layer 0's output observation is the environment observation."""
    sizes = [4]
    sensations = Sensations.create(
        sizes,
        predictor_features=[1],
        credit_settings=CREDIT_SETTINGS,
        mismatch_settings=MISMATCH_SETTINGS,
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
        outputs[0].message.observation.mean_times_precision,
        env_observation.mean_times_precision,
    )
    assert jnp.allclose(
        outputs[0].message.observation.negative_half_precision,
        env_observation.negative_half_precision,
    )


def test_sensations_infer_feeds_costs_from_below(
    streams: Mapping[str, RngStream],
) -> None:
    """Each interior layer receives both costs from the layer below."""
    sizes = [2, 3]
    sensations = Sensations.create(
        sizes,
        predictor_features=[1, 1],
        credit_settings=CREDIT_SETTINGS,
        mismatch_settings=MISMATCH_SETTINGS,
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

    # Layer 0's effective predictor concats its external predictor with the
    # (absent) last-message of the single interior layer above it.
    fresh = eqx.nn.State(sensations)
    top_prev = fresh.get(sensations.interior_features[0].last_message_index)
    layer0_predictor = concat_instrumented_messages(predictors[0], top_prev)
    fresh, layer0 = sensations.input_feature.step(
        fresh,
        observation=observation,
        predictor=layer0_predictor,
        gain=None,
        streams=streams,
        inference=True,
    )
    keys = jr.split(key, len(sensations.interior_features))
    _, layer1 = sensations.interior_features[0].step(
        fresh,
        keys[0],
        observation_content=layer0.observation_mismatch,
        predictor=predictors[1],  # top layer has no higher layers to concat
        gain=None,
        reconstruction=layer0.reconstruction_cost,
        confounding=layer0.confounding_cost,
        streams=streams,
        inference=True,
    )

    assert jnp.allclose(
        outputs[1].message.observation.mean_times_precision,
        layer1.message.observation.mean_times_precision,
    )
    assert jnp.allclose(outputs[1].loss, layer1.loss)
