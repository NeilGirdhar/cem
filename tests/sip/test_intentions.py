from collections.abc import Mapping

import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
from efax import NormalPP
from tjax import RngStream

from cem.sip.intentions import Intentions
from cem.sip.messages import InstrumentedMessage, concat_messages
from cem.structure.graph import LearnableParameter

from .tools import CREDIT_SETTINGS, MISMATCH_SETTINGS, _i_message, _message


def test_intentions_create_sizes_readouts_demands_and_predictors(
    streams: Mapping[str, RngStream],
) -> None:
    sizes = [2, 3, 4]
    readout_features = [5, 6, 7]
    intentions = Intentions.create(
        sizes,
        readout_features=readout_features,
        credit_settings=CREDIT_SETTINGS,
        mismatch_settings=MISMATCH_SETTINGS,
        streams=streams,
    )
    assert len(intentions.features) == len(sizes)
    top = len(sizes) - 1
    for i, feature in enumerate(intentions.features):
        observation_weight = feature.emitter.observation_map.layers[0].weight
        expected = readout_features[i] + (0 if i == top else 2 * sizes[i + 1])
        assert observation_weight.value.shape[-1] == expected
        assert isinstance(observation_weight, LearnableParameter)
        assert isinstance(feature.emitter.instrument_map.layers[0].weight, LearnableParameter)
        predictor_width = feature.mismatch.prediction_map.input_pool.dummy_mean.value.size
        assert predictor_width == (sizes[i - 1] if i else 0)


def test_intentions_infer_returns_one_output_per_layer(
    streams: Mapping[str, RngStream],
) -> None:
    sizes = [2, 3]
    intentions = Intentions.create(
        sizes,
        readout_features=[2, 2],
        credit_settings=CREDIT_SETTINGS,
        mismatch_settings=MISMATCH_SETTINGS,
        streams=streams,
    )
    state = eqx.nn.State(intentions)
    _, outputs = intentions.infer(
        state,
        jr.key(0),
        readouts=[_message(jnp.ones(2), jnp.ones(2)) for _ in sizes],
        reward=jnp.asarray(0.0),
        curiosity=jnp.asarray(0.0),
        tracking=jnp.asarray(0.0),
        streams=streams,
        inference=True,
    )
    assert len(outputs) == len(sizes)
    assert outputs[0].message.observation.mean_times_precision.shape == (sizes[0],)
    assert outputs[1].message.observation.mean_times_precision.shape == (sizes[1],)


def test_intentions_infer_uses_previous_lower_message_and_current_upper_mismatch(
    streams: Mapping[str, RngStream],
) -> None:
    sizes = [2, 3]
    intentions = Intentions.create(
        sizes,
        readout_features=[2, 2],
        credit_settings=CREDIT_SETTINGS,
        mismatch_settings=MISMATCH_SETTINGS,
        streams=streams,
    )
    key = jr.key(0)
    readouts = [_message(jnp.ones(2), jnp.ones(2)) for _ in sizes]
    advantages = {
        "reward": jnp.asarray(0.1),
        "curiosity": jnp.asarray(0.2),
        "tracking": jnp.asarray(0.3),
    }

    bottom_index = intentions.features[0].last_message_index
    bottom_message = _i_message(
        NormalPP(jnp.array([0.5, -0.3]), jnp.ones(2)), NormalPP(jnp.array([0.2, -0.1]), jnp.ones(2))
    )
    state = eqx.nn.State(intentions).set(bottom_index, bottom_message)
    _, outputs = intentions.infer(
        state,
        key,
        readouts=readouts,
        **advantages,
        streams=streams,
        inference=True,
    )

    keys = jr.split(key, len(sizes))
    fresh = eqx.nn.State(intentions).set(bottom_index, bottom_message)
    fresh, top_out = intentions.features[1].step(
        fresh,
        keys[1],
        observation_content=readouts[1],
        predictor=bottom_message,
        gain=None,
        **advantages,
        streams=streams,
        inference=True,
    )
    _, bottom_out = intentions.features[0].step(
        fresh,
        keys[0],
        observation_content=concat_messages(readouts[0], top_out.observation_mismatch),
        predictor=InstrumentedMessage.initial(0),
        gain=None,
        **advantages,
        streams=streams,
        inference=True,
    )

    assert jnp.allclose(
        outputs[0].message.observation.mean_times_precision,
        bottom_out.message.observation.mean_times_precision,
    )
    assert jnp.allclose(outputs[0].loss, bottom_out.loss)
    assert jnp.allclose(outputs[1].loss, top_out.loss)
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
        readout_features=[2, 2],
        credit_settings=CREDIT_SETTINGS,
        mismatch_settings=MISMATCH_SETTINGS,
        streams=streams,
    )
    state = eqx.nn.State(intentions)
    _, outputs = intentions.infer(
        state,
        jr.key(0),
        readouts=[_message(jnp.ones(2), jnp.ones(2)) for _ in sizes],
        reward=jnp.asarray(0.0),
        curiosity=jnp.asarray(0.0),
        tracking=jnp.asarray(0.0),
        streams=streams,
        inference=True,
    )
    action = outputs[0].message.observation
    assert action.mean_times_precision.shape == (sizes[0],)
