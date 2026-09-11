import jax.numpy as jnp
import jax.random as jr
from tjax import create_streams

from cem.sip import SIPScore


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


def test_witness_is_normalized_before_gain() -> None:
    score = _score()
    output = score.infer(
        *_inputs()[:3],
        jnp.array([1.0]),
        streams=create_streams({"inference": jr.key(15)}),
        inference=True,
    )
    assert jnp.allclose(jnp.mean(jnp.square(output.witness)), 1.0, atol=1e-3)
