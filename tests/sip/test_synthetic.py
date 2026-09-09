"""Controlled tests for the predictive effect of SIP interventions."""

import jax.numpy as jnp
import jax.random as jr


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
