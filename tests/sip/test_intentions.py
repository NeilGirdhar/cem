from collections.abc import Mapping

import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
from tjax import RngStream

from cem.sip.intentions import Intentions
from cem.sip.messages import (
    InstrumentedMessage,
    concat_instrumented_messages,
    concat_messages,
)

from .tools import _message


def _predictors(size: int) -> InstrumentedMessage:
    return InstrumentedMessage.standard(size)


def test_intentions_create_content_sizes_follow_readout_plus_demand_from_above(
    streams: Mapping[str, RngStream],
) -> None:
    """Top layer content = readout; lower layers = readout + 2 * sizes[i + 1]."""
    sizes = [2, 3, 4]
    readout_features = [5, 6, 7]
    intentions = Intentions.create(
        sizes,
        predictor_features=[1, 1, 1],
        readout_features=readout_features,
        streams=streams,
    )
    assert len(intentions.features) == len(sizes)
    top = len(sizes) - 1
    for i, feature in enumerate(intentions.features):
        weight = feature.emitter.observation_map.layers[0].weight.value
        expected = readout_features[i] + (0 if i == top else 2 * sizes[i + 1])
        assert weight.shape[-1] == expected


def test_intentions_infer_returns_one_output_per_layer(
    streams: Mapping[str, RngStream],
) -> None:
    sizes = [2, 3]
    intentions = Intentions.create(
        sizes,
        predictor_features=[1, 1],
        readout_features=[2, 2],
        streams=streams,
    )
    state = eqx.nn.State(intentions)
    _, outputs = intentions.infer(
        state,
        jr.key(0),
        readouts=[_message(jnp.ones(2), jnp.ones(2)) for _ in sizes],
        predictors=[_predictors(1), _predictors(1)],
        advantages=[jnp.asarray(0.0), jnp.asarray(0.0)],
        streams=streams,
        inference=True,
    )
    assert len(outputs) == len(sizes)
    assert outputs[0].message.observation.mean_times_precision.shape == (sizes[0],)
    assert outputs[1].message.observation.mean_times_precision.shape == (sizes[1],)


def test_intentions_infer_cascades_top_mismatch_as_lower_content(
    streams: Mapping[str, RngStream],
) -> None:
    """Bottom layer run with concat(readout, top's mismatch) reproduces infer's output."""
    sizes = [2, 3]
    intentions = Intentions.create(
        sizes,
        predictor_features=[1, 1],
        readout_features=[2, 2],
        streams=streams,
    )
    key = jr.key(0)
    readouts = [_message(jnp.ones(2), jnp.ones(2)) for _ in sizes]
    predictors = [_predictors(1), _predictors(1)]
    advantages = [jnp.asarray(0.1), jnp.asarray(0.2)]

    state = eqx.nn.State(intentions)
    _, outputs = intentions.infer(
        state,
        key,
        readouts=readouts,
        predictors=predictors,
        advantages=advantages,
        streams=streams,
        inference=True,
    )

    # Re-run manually: top first, then bottom with concat.
    keys = jr.split(key, len(sizes))
    fresh = eqx.nn.State(intentions)
    fresh, top_out = intentions.features[1].step(
        fresh,
        keys[1],
        observation_content=readouts[1],
        predictor=predictors[1],  # top has no higher layers to concat
        gain=None,
        goal=advantages[1],
        streams=streams,
        inference=True,
    )
    # Bottom's effective predictor concats its external predictor with the
    # top layer's just-written last message.
    top_message = fresh.get(intentions.features[1].last_message_index)
    bottom_predictor = concat_instrumented_messages(predictors[0], top_message)
    _, bottom_out = intentions.features[0].step(
        fresh,
        keys[0],
        observation_content=concat_messages(readouts[0], top_out.observation_mismatch),
        predictor=bottom_predictor,
        gain=None,
        goal=advantages[0],
        streams=streams,
        inference=True,
    )

    assert jnp.allclose(
        outputs[0].message.observation.mean_times_precision,
        bottom_out.message.observation.mean_times_precision,
    )
    assert jnp.allclose(
        outputs[1].message.observation.mean_times_precision,
        top_out.message.observation.mean_times_precision,
    )


def test_intentions_action_is_bottom_layer_observation(
    streams: Mapping[str, RngStream],
) -> None:
    """The action equals `outputs[0].observation` (bottom layer, processed last)."""
    sizes = [2, 3]
    intentions = Intentions.create(
        sizes,
        predictor_features=[1, 1],
        readout_features=[2, 2],
        streams=streams,
    )
    state = eqx.nn.State(intentions)
    _, outputs = intentions.infer(
        state,
        jr.key(0),
        readouts=[_message(jnp.ones(2), jnp.ones(2)) for _ in sizes],
        predictors=[_predictors(1), _predictors(1)],
        advantages=[jnp.asarray(0.0), jnp.asarray(0.0)],
        streams=streams,
        inference=True,
    )
    action = outputs[0].message.observation
    assert action.mean_times_precision.shape == (sizes[0],)
