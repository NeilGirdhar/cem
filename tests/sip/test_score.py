import jax
import jax.numpy as jnp
import jax.random as jr
from tjax import create_streams

from cem.sip import SIPScore, purification_loss, witness_loss


def _score() -> SIPScore:
    return SIPScore.create(
        predictor_observation_features=3,
        predictor_instrument_features=2,
        observation_features=4,
        hidden_features=6,
        streams=create_streams({"parameters": jr.key(10), "inference": jr.key(11)}),
    )


def _inputs() -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    return (
        jnp.array([0.2, -0.3, 0.5, 0.1]),
        jnp.array([0.4, -0.2, 0.3]),
        jnp.array([0.1, 0.6]),
        jnp.array([0.75]),
    )


def test_score_outputs_reconstruction_and_confounding_terms() -> None:
    score = _score()
    output = score.infer(
        *_inputs(),
        streams=create_streams({"inference": jr.key(12)}),
        inference=True,
    )
    assert output.prediction.shape == (4,)
    assert output.observation_score.shape == (4,)
    assert output.witness.shape == (4,)
    assert output.reconstruction_loss.shape == ()
    assert output.confounding_error.shape == ()
    assert jnp.allclose(output.reconstruction_loss, 0.5 * jnp.sum(output.observation_score**2))
    assert jnp.allclose(
        output.confounding_error,
        jnp.sum(output.observation_score * output.witness),
    )


def test_gain_scales_prediction_and_witness() -> None:
    score = _score()
    observation, predictor_observations, predictor_instruments, _ = _inputs()
    streams = create_streams({"inference": jr.key(13)})
    full = score.infer(
        observation,
        predictor_observations,
        predictor_instruments,
        jnp.array([1.0]),
        streams=streams,
        inference=True,
    )
    half = score.infer(
        observation,
        predictor_observations,
        predictor_instruments,
        jnp.array([0.5]),
        streams=streams,
        inference=True,
    )
    assert jnp.allclose(half.prediction, 0.5 * full.prediction)
    assert jnp.allclose(half.witness, 0.5 * full.witness)


def test_score_has_finite_gradients() -> None:
    score = _score()
    inputs = _inputs()
    streams = create_streams({"inference": jr.key(14)})

    def loss(model: SIPScore) -> jnp.ndarray:
        output = model.infer(*inputs, streams=streams, inference=True)
        return output.reconstruction_loss + output.confounding_error**2

    gradients = jax.grad(loss)(score)
    assert all(jnp.all(jnp.isfinite(leaf)) for leaf in jax.tree.leaves(gradients))


def test_witness_is_normalized_before_gain() -> None:
    score = _score()
    output = score.infer(
        *_inputs()[:3],
        jnp.array([1.0]),
        streams=create_streams({"inference": jr.key(15)}),
        inference=True,
    )
    assert jnp.allclose(jnp.mean(jnp.square(output.witness)), 1.0, atol=1e-3)


def test_alternating_sip_objectives_have_finite_gradients() -> None:
    score = _score()
    inputs = _inputs()
    streams = create_streams({"inference": jr.key(16)})

    def predictor_objective(model: SIPScore) -> jnp.ndarray:
        output = model.infer(*inputs, streams=streams, inference=True)
        return purification_loss(output, confounding_weight=0.5)

    def witness_objective(model: SIPScore) -> jnp.ndarray:
        output = model.infer(*inputs, streams=streams, inference=True)
        return witness_loss(output)

    predictor_gradients = jax.grad(predictor_objective)(score)
    witness_gradients = jax.grad(witness_objective)(score)
    assert all(jnp.all(jnp.isfinite(leaf)) for leaf in jax.tree.leaves(predictor_gradients))
    assert all(jnp.all(jnp.isfinite(leaf)) for leaf in jax.tree.leaves(witness_gradients))
