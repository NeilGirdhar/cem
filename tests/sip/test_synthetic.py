"""Controlled tests for the predictive effect of SIP interventions."""

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
from tjax import create_streams

from cem.sip import SIPScore


def _fit_linear(features: jnp.ndarray, target: jnp.ndarray) -> jnp.ndarray:
    gram = features.T @ features
    return jnp.linalg.solve(gram + 1e-6 * jnp.eye(features.shape[-1]), features.T @ target)


def test_intervention_noise_reduces_nuisance_shift_error() -> None:
    """Shared emitter noise discourages a predictor from relying on a nuisance feature.

    The stable source feature is measured noisily. The nuisance feature is a much more
    precise copy during training, but becomes independent at inference. Adding the
    emitter's Gaussian intervention to both source channels acts as a controlled
    robustness regularizer.
    """
    key = jr.key(30)
    signal_key, stable_key, nuisance_key, test_key, noise_key = jr.split(key, 5)
    count = 4096
    signal = jr.normal(signal_key, (count,))
    stable = signal + 0.5 * jr.normal(stable_key, (count,))
    nuisance = signal + 0.05 * jr.normal(nuisance_key, (count,))
    training_features = jnp.stack((stable, nuisance), axis=-1)
    shifted_features = jnp.stack((stable, jr.normal(test_key, (count,))), axis=-1)

    clean_weights = _fit_linear(training_features, signal)
    intervention = 0.6 * jr.normal(noise_key, training_features.shape)
    noisy_weights = _fit_linear(training_features + intervention, signal)
    clean_error = jnp.mean(jnp.square(shifted_features @ clean_weights - signal))
    noisy_error = jnp.mean(jnp.square(shifted_features @ noisy_weights - signal))

    assert noisy_error < 0.8 * clean_error


def test_trained_score_with_noise_is_more_robust_to_nuisance_shift() -> None:
    """A score circuit trained with interventions can ignore a shifted nuisance."""
    key = jr.key(31)
    signal_key, stable_key, nuisance_key, test_key = jr.split(key, 4)
    count = 64
    signal = jr.normal(signal_key, (count,))
    stable = signal + 0.5 * jr.normal(stable_key, (count,))
    nuisance = signal + 0.05 * jr.normal(nuisance_key, (count,))
    training_features = jnp.stack((stable, nuisance), axis=-1)
    shifted_features = jnp.stack((stable, jr.normal(test_key, (count,))), axis=-1)
    instruments = jnp.zeros((count, 1))
    gain = jnp.ones((count, 1))
    target = signal[:, jnp.newaxis]

    def train(*, use_intervention: bool) -> SIPScore:
        model = SIPScore.create(
            predictor_observation_features=2,
            predictor_instrument_features=1,
            observation_features=1,
            streams=create_streams({"parameters": jr.key(32), "inference": jr.key(33)}),
        )
        for step in range(300):
            intervention = (
                0.6 * jr.normal(jr.fold_in(jr.key(34), step), training_features.shape)
                if use_intervention
                else jnp.zeros_like(training_features)
            )
            features = training_features + intervention

            def loss(current: SIPScore, predictor_observations: jnp.ndarray) -> jnp.ndarray:
                result = current.infer(
                    target,
                    predictor_observations,
                    instruments,
                    gain,
                    streams=create_streams({"inference": jr.key(35)}),
                    inference=True,
                )
                return jnp.mean(result.reconstruction_loss)

            _, gradients = eqx.filter_value_and_grad(loss)(model, features)
            model = eqx.apply_updates(model, jax.tree.map(lambda value: -0.001 * value, gradients))
        return model

    clean_model = train(use_intervention=False)
    noisy_model = train(use_intervention=True)

    def shifted_loss(model: SIPScore) -> jnp.ndarray:
        result = model.infer(
            target,
            shifted_features,
            instruments,
            gain,
            streams=create_streams({"inference": jr.key(36)}),
            inference=True,
        )
        return jnp.mean(result.reconstruction_loss)

    clean_error = shifted_loss(clean_model)
    noisy_error = shifted_loss(noisy_model)
    assert noisy_error < clean_error
