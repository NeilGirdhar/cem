import jax
import jax.numpy as jnp
import jax.random as jr
from tjax import create_streams

from cem.sip import SIPChain


def _chain() -> SIPChain:
    return SIPChain.create(
        innovation_features=2,
        goal_features=1,
        parent_instrument_features=2,
        source_features=3,
        target_features=2,
        hidden_features=5,
        initial_noise=0.1,
        streams=create_streams({"parameters": jr.key(20), "inference": jr.key(21)}),
    )


def _inputs() -> tuple[jnp.ndarray, ...]:
    return (
        jnp.array([0.2, -0.4]),
        jnp.array([0.3]),
        jnp.array([0.8]),
        jnp.array([0.1, -0.2]),
        jnp.array([0.5, -0.1]),
        jnp.array([0.7]),
    )


def test_chain_connects_source_observation_and_instrument() -> None:
    chain = _chain()
    result = chain.infer(
        *_inputs(),
        streams=create_streams({"inference": jr.key(22)}),
        inference=True,
    )
    assert result.source.observation.shape == (3,)
    assert result.source.instrument.shape == (3,)
    assert result.target.prediction.shape == (2,)
    assert result.target.witness.shape == (2,)
    assert jnp.all(jnp.isfinite(result.target.observation_score))


def test_chain_gradients_reach_emitter_and_score() -> None:
    chain = _chain()
    inputs = _inputs()
    streams = create_streams({"inference": jr.key(23)})

    def loss(model: SIPChain) -> jnp.ndarray:
        output = model.infer(*inputs, streams=streams, inference=True)
        return output.target.reconstruction_loss + output.target.confounding_error**2

    gradients = jax.grad(loss)(chain)
    assert all(jnp.all(jnp.isfinite(leaf)) for leaf in jax.tree.leaves(gradients))


def test_chain_passes_intervention_noise_to_downstream_prediction() -> None:
    chain = _chain()
    inputs = _inputs()
    first = chain.infer(
        *inputs,
        streams=create_streams({"inference": jr.key(24)}),
        inference=False,
    )
    second = chain.infer(
        *inputs,
        streams=create_streams({"inference": jr.key(25)}),
        inference=False,
    )

    assert not jnp.allclose(first.source.injected_noise, second.source.injected_noise)
    assert not jnp.allclose(first.source.observation, second.source.observation)
    assert not jnp.allclose(first.target.prediction, second.target.prediction)


def test_chain_is_deterministic_without_intervention_noise() -> None:
    chain = _chain()
    inputs = _inputs()
    first = chain.infer(
        *inputs,
        streams=create_streams({"inference": jr.key(26)}),
        inference=True,
    )
    second = chain.infer(
        *inputs,
        streams=create_streams({"inference": jr.key(27)}),
        inference=True,
    )

    assert jnp.allclose(first.source.observation, second.source.observation)
    assert jnp.allclose(first.source.instrument, second.source.instrument)
    assert jnp.allclose(first.target.prediction, second.target.prediction)
