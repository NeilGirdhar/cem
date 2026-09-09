import equinox as eqx
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


def test_chain_learns_a_synthetic_target_relation() -> None:
    """Both emitter and score weights can learn through a noisy observation."""
    key = jr.key(28)
    innovation_key, goal_key, parent_key, target_key = jr.split(key, 4)
    innovation = jr.normal(innovation_key, (32, 2))
    goal = jr.normal(goal_key, (32, 1))
    source_gain = jnp.ones((32, 1))
    parent_instruments = jr.normal(parent_key, (32, 1))
    target_observation = innovation[:, :1] + 0.2 * jr.normal(target_key, (32, 1))
    target_gain = jnp.ones((32, 1))
    noise_stream_key = jr.key(29)
    model = SIPChain.create(
        innovation_features=2,
        goal_features=1,
        parent_instrument_features=1,
        source_features=2,
        target_features=1,
        hidden_features=4,
        initial_noise=0.1,
        streams=create_streams({"parameters": jr.key(30), "inference": jr.key(31)}),
    )

    def loss(current: SIPChain, step: int) -> jnp.ndarray:
        result = current.infer(
            innovation,
            goal,
            source_gain,
            parent_instruments,
            target_observation,
            target_gain,
            streams=create_streams({"inference": jr.fold_in(noise_stream_key, step)}),
            inference=False,
        )
        return jnp.mean(result.target.reconstruction_loss)

    initial_loss = loss(model, 0)
    for step in range(1, 61):
        _, gradients = eqx.filter_value_and_grad(loss)(model, step)
        model = eqx.apply_updates(model, jax.tree.map(lambda value: -0.03 * value, gradients))
    final_loss = loss(model, 61)

    assert final_loss < 0.75 * initial_loss


def test_fixed_intervention_noise_improves_full_chain_shifted_error() -> None:
    """A fixed SIP intervention can improve robustness through the whole chain."""
    key = jr.key(37)
    signal_key, stable_key, nuisance_key, shift_key = jr.split(key, 4)
    count = 32
    signal = jr.normal(signal_key, (count,))
    stable = signal + 0.5 * jr.normal(stable_key, (count,))
    nuisance = signal + 0.05 * jr.normal(nuisance_key, (count,))
    training_innovation = jnp.stack((stable, nuisance), axis=-1)
    shifted_innovation = jnp.stack((stable, jr.normal(shift_key, (count,))), axis=-1)
    goal = jnp.zeros((count, 1))
    parent_instruments = jnp.zeros((count, 1))
    gain = jnp.ones((count, 1))
    target = signal[:, jnp.newaxis]
    noise_key = jr.key(38)

    def train(initial_noise: float) -> SIPChain:
        model = SIPChain.create(
            innovation_features=2,
            goal_features=1,
            parent_instrument_features=1,
            source_features=2,
            target_features=1,
            hidden_features=(),
            initial_noise=initial_noise,
            learn_noise=False,
            streams=create_streams({"parameters": jr.key(39), "inference": jr.key(40)}),
        )
        for step in range(80):

            def loss(current: SIPChain, update_step: int) -> jnp.ndarray:
                result = current.infer(
                    training_innovation,
                    goal,
                    gain,
                    parent_instruments,
                    target,
                    gain,
                    streams=create_streams({"inference": jr.fold_in(noise_key, update_step)}),
                    inference=False,
                )
                return jnp.mean(result.target.reconstruction_loss)

            _, gradients = eqx.filter_value_and_grad(loss)(model, step)
            model = eqx.apply_updates(model, jax.tree.map(lambda value: -0.02 * value, gradients))
        return model

    def shifted_loss(model: SIPChain) -> jnp.ndarray:
        result = model.infer(
            shifted_innovation,
            goal,
            gain,
            parent_instruments,
            target,
            gain,
            streams=create_streams({"inference": jr.key(41)}),
            inference=True,
        )
        return jnp.mean(result.target.reconstruction_loss)

    clean_error = shifted_loss(train(1e-6))
    noisy_error = shifted_loss(train(0.6))
    assert noisy_error < clean_error
