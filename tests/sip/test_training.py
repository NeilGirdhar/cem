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
