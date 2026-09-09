import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
from tjax import create_streams

from cem.sip import SIPChain, SIPScore, train_chain_adversarial, train_score_adversarial


def test_alternating_score_training_records_finite_objectives() -> None:
    count = 32
    predictor_observations = jr.normal(jr.key(50), (count, 2))
    observation = predictor_observations[:, :1]
    predictor_instruments = predictor_observations[:, 1:]
    gain = jnp.ones((count, 1))
    score = SIPScore.create(
        predictor_observation_features=2,
        predictor_instrument_features=1,
        observation_features=1,
        hidden_features=4,
        streams=create_streams({"parameters": jr.key(51), "inference": jr.key(52)}),
    )

    trained, history = train_score_adversarial(
        score,
        observation,
        predictor_observations,
        predictor_instruments,
        gain,
        steps=8,
        predictor_learning_rate=0.01,
        witness_learning_rate=0.01,
        streams=create_streams({"inference": jr.key(53)}),
    )

    assert history.purification_losses.shape == (8,)
    assert history.witness_losses.shape == (8,)
    assert jnp.all(jnp.isfinite(history.purification_losses))
    assert jnp.all(jnp.isfinite(history.witness_losses))
    output = trained.infer(
        observation,
        predictor_observations,
        predictor_instruments,
        gain,
        streams=create_streams({"inference": jr.key(54)}),
        inference=True,
    )
    assert jnp.all(jnp.isfinite(output.observation_score))
    assert all(jnp.all(jnp.isfinite(leaf)) for leaf in jax.tree.leaves(trained))


def test_alternating_chain_training_records_finite_objectives() -> None:
    count = 16
    innovation = jr.normal(jr.key(55), (count, 2))
    goal = jr.normal(jr.key(56), (count, 1))
    source_gain = jnp.ones((count, 1))
    parent_instruments = jr.normal(jr.key(57), (count, 1))
    target_observation = innovation[:, :1]
    target_gain = jnp.ones((count, 1))
    chain = SIPChain.create(
        innovation_features=2,
        goal_features=1,
        parent_instrument_features=1,
        source_features=2,
        target_features=1,
        hidden_features=3,
        streams=create_streams({"parameters": jr.key(58), "inference": jr.key(59)}),
    )

    trained, history = train_chain_adversarial(
        chain,
        innovation,
        goal,
        source_gain,
        parent_instruments,
        target_observation,
        target_gain,
        steps=4,
        predictor_learning_rate=0.01,
        witness_learning_rate=0.01,
        streams=create_streams({"inference": jr.key(60)}),
    )

    assert history.purification_losses.shape == (4,)
    assert history.witness_losses.shape == (4,)
    assert jnp.all(jnp.isfinite(history.purification_losses))
    assert jnp.all(jnp.isfinite(history.witness_losses))
    assert all(jnp.all(jnp.isfinite(leaf)) for leaf in jax.tree.leaves(trained))


def test_adversarial_training_reduces_instrument_conditioned_residual() -> None:
    count = 64
    signal = jr.normal(jr.key(61), (count,))
    clean_noise = jr.normal(jr.key(62), (count,))
    instrument = jr.normal(jr.key(63), (count,))
    observation = jnp.stack(
        (signal + 0.8 * clean_noise, signal + 0.1 * instrument),
        axis=-1,
    )
    target = signal[:, jnp.newaxis]
    instruments = instrument[:, jnp.newaxis]
    gain = jnp.ones((count, 1))

    def train(*, adversarial: bool) -> SIPScore:
        score = SIPScore.create(
            predictor_observation_features=2,
            predictor_instrument_features=1,
            observation_features=1,
            hidden_features=(),
            streams=create_streams({"parameters": jr.key(64), "inference": jr.key(65)}),
        )
        trained, _ = train_score_adversarial(
            score,
            target,
            observation,
            instruments,
            gain,
            steps=300,
            predictor_learning_rate=0.005,
            witness_learning_rate=0.005 if adversarial else 0.0,
            confounding_weight=2.0 if adversarial else 0.0,
            streams=create_streams({"inference": jr.key(66)}),
        )
        return trained

    def residual_correlation(score: SIPScore) -> jnp.ndarray:
        output = score.infer(
            target,
            observation,
            instruments,
            gain,
            streams=create_streams({"inference": jr.key(67)}),
            inference=True,
        )
        residual = output.observation_score[:, 0]
        return jnp.square(jnp.mean(residual * instruments[:, 0]))

    ordinary_correlation = residual_correlation(train(adversarial=False))
    adversarial_correlation = residual_correlation(train(adversarial=True))
    assert adversarial_correlation < ordinary_correlation


def test_full_chain_purification_survives_shifted_confounding() -> None:
    count = 32
    signal = jr.normal(jr.key(1), (count,))
    clean_noise = jr.normal(jr.key(2), (count,))
    instrument = jr.normal(jr.key(3), (count,))
    training_innovation = jnp.stack(
        (signal + 0.5 * clean_noise, signal + 0.1 * instrument),
        axis=-1,
    )
    shifted_innovation = jnp.stack(
        (signal + 0.5 * clean_noise, jr.normal(jr.key(8), (count,))),
        axis=-1,
    )
    goal = jnp.zeros((count, 1))
    parent_instruments = instrument[:, jnp.newaxis]
    gain = jnp.ones((count, 1))
    target = signal[:, jnp.newaxis]

    def train(*, adversarial: bool) -> SIPChain:
        chain = SIPChain.create(
            innovation_features=2,
            goal_features=1,
            parent_instrument_features=1,
            source_features=2,
            target_features=1,
            hidden_features=(),
            initial_noise=0.05,
            learn_noise=False,
            streams=create_streams({"parameters": jr.key(4), "inference": jr.key(5)}),
        )
        trained, _ = train_chain_adversarial(
            chain,
            training_innovation,
            goal,
            gain,
            parent_instruments,
            target,
            gain,
            steps=100,
            predictor_learning_rate=0.005,
            witness_learning_rate=0.005 if adversarial else 0.0,
            confounding_weight=2.0 if adversarial else 0.0,
            streams=create_streams({"inference": jr.key(6)}),
        )
        return trained

    def shifted_loss(chain: SIPChain) -> jnp.ndarray:
        output = chain.infer(
            shifted_innovation,
            goal,
            gain,
            parent_instruments,
            target,
            gain,
            streams=create_streams({"inference": jr.key(7)}),
            inference=True,
        )
        return jnp.mean(output.target.reconstruction_loss)

    ordinary_error = shifted_loss(train(adversarial=False))
    adversarial_error = shifted_loss(train(adversarial=True))
    assert adversarial_error < ordinary_error


def test_chain_witness_update_does_not_change_predictor_path() -> None:
    count = 8
    innovation = jr.normal(jr.key(76), (count, 2))
    goal = jnp.zeros((count, 1))
    gain = jnp.ones((count, 1))
    parent_instruments = jr.normal(jr.key(77), (count, 1))
    target = innovation[:, :1]
    initial = SIPChain.create(
        innovation_features=2,
        goal_features=1,
        parent_instrument_features=1,
        source_features=2,
        target_features=1,
        hidden_features=(),
        streams=create_streams({"parameters": jr.key(78), "inference": jr.key(79)}),
    )
    trained, _ = train_chain_adversarial(
        initial,
        innovation,
        goal,
        gain,
        parent_instruments,
        target,
        gain,
        steps=2,
        predictor_learning_rate=0.0,
        witness_learning_rate=0.01,
        streams=create_streams({"inference": jr.key(80)}),
    )

    assert all(
        jnp.allclose(before, after)
        for before, after in zip(
            jax.tree.leaves(initial.emitter),
            jax.tree.leaves(trained.emitter),
            strict=True,
        )
        if eqx.is_array(before)
    )
    assert all(
        jnp.allclose(before, after)
        for before, after in zip(
            jax.tree.leaves(initial.score.prediction_map),
            jax.tree.leaves(trained.score.prediction_map),
            strict=True,
        )
        if eqx.is_array(before)
    )
