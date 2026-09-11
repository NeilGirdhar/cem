import jax.numpy as jnp
import jax.random as jr
from tjax import create_streams

from cem.sip import SIPEmitter


def _emitter() -> SIPEmitter:
    return SIPEmitter.create(
        innovation_features=3,
        goal_features=2,
        predictor_instrument_features=4,
        observation_features=5,
        hidden_features=7,
        initial_noise=0.2,
        streams=create_streams({"parameters": jr.key(1), "inference": jr.key(2)}),
    )


def _inputs() -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    return (
        jnp.array([0.2, -0.3, 0.5]),
        jnp.array([0.1, 0.4]),
        jnp.array([0.8]),
        jnp.array([0.4, -0.2, 0.3, 0.1]),
    )


def test_inference_disables_local_noise() -> None:
    emitter = _emitter()
    inputs = _inputs()
    result = emitter.infer(
        *inputs,
        streams=create_streams({"inference": jr.key(3)}),
        inference=True,
    )
    assert jnp.allclose(result.injected_noise, 0.0)
    assert jnp.allclose(result.observation, inputs[2] * result.raw_observation)
    assert jnp.allclose(result.instrument, inputs[2] * result.inherited_instrument)


def test_training_adds_the_same_noise_to_both_channels() -> None:
    emitter = _emitter()
    inputs = _inputs()
    result = emitter.infer(
        *inputs,
        streams=create_streams({"inference": jr.key(3)}),
        inference=False,
    )
    gain = inputs[2]
    assert jnp.allclose(
        result.observation - gain * result.raw_observation,
        result.instrument - gain * result.inherited_instrument,
    )


def test_gain_scales_observation_and_instrument() -> None:
    emitter = _emitter()
    innovation, goal, _gain, instruments = _inputs()
    streams = create_streams({"inference": jr.key(4)})
    one = emitter.infer(
        innovation,
        goal,
        jnp.array([1.0]),
        instruments,
        streams=streams,
        inference=True,
    )
    half = emitter.infer(
        innovation,
        goal,
        jnp.array([0.5]),
        instruments,
        streams=streams,
        inference=True,
    )
    assert jnp.allclose(half.observation, 0.5 * one.observation)
    assert jnp.allclose(half.instrument, 0.5 * one.instrument)


def test_noise_magnitudes_are_nonnegative() -> None:
    assert jnp.all(_emitter().noise_magnitudes >= 0.0)
