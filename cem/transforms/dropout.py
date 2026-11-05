"""Inverted dropout for stochastic regularization of activations."""

import jax.numpy as jnp
import jax.random as jr
from tjax import JaxArray, JaxRealArray


def dropout(x: JaxArray, key: JaxArray, rate: float | JaxRealArray) -> JaxArray:
    """Randomly zero activations and scale the retained values by inverse keep probability."""
    mask = jr.bernoulli(key, 1.0 - rate, shape=x.shape)
    return jnp.where(mask, x / (1.0 - rate), jnp.zeros_like(x))
