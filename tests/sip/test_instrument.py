"""Check the inherited instrument and the instrument loss that trains it."""

from collections.abc import Mapping

import jax
import jax.numpy as jnp
from efax import NormalNP
from jax import tree
from tjax import JaxRealArray, RngStream

from cem.npn.gaussian import GaussianNeuralProjection
from cem.sip.instrument import inherit_instrument, instrument_loss

from .tools import _message


def _all_zero(gradients: object) -> bool:
    return all(jnp.allclose(leaf, 0.0) for leaf in tree.leaves(gradients))


def test_inherit_instrument_holds_parent_instruments_fixed(
    streams: Mapping[str, RngStream],
) -> None:
    instrument_map = GaussianNeuralProjection.create(in_features=2, out_features=3, streams=streams)

    def output_sum(predictor_instrument: NormalNP) -> JaxRealArray:
        inherited = inherit_instrument(instrument_map, predictor_instrument)
        return jnp.sum(inherited.mean_times_precision + inherited.negative_half_precision)

    gradient = jax.grad(output_sum)(_message(jnp.asarray([0.5, -0.1]), jnp.asarray([1.0, 2.0])))
    assert _all_zero(gradient)


def test_instrument_loss_holds_observation_and_gain_fixed() -> None:
    inherited = _message(jnp.asarray([0.2, -0.4]), jnp.asarray([2.0, 0.5]))
    raw_observation = _message(jnp.asarray([0.5, 0.1]), jnp.asarray([1.0, 3.0]))
    gain = jnp.asarray([0.3, 0.9])
    gradients = jax.grad(lambda raw, g: instrument_loss(inherited, raw, g), argnums=(0, 1))(
        raw_observation, gain
    )
    assert _all_zero(gradients)


def test_instrument_loss_is_gain_weighted_cross_entropy() -> None:
    """A standard-normal fit costs the standard-normal entropy, scaled by the stopped gain."""
    standard = _message(jnp.zeros(2), jnp.ones(2))
    entropy = 0.5 * jnp.log(2.0 * jnp.pi * jnp.e)
    assert jnp.allclose(instrument_loss(standard, standard, None), 2.0 * entropy)
    gain = jnp.asarray([0.25, 0.0])
    assert jnp.allclose(instrument_loss(standard, standard, gain), 0.25 * entropy)


def test_instrument_loss_moves_instrument_towards_observation() -> None:
    inherited = _message(jnp.asarray([0.0]), jnp.asarray([1.0]))
    raw_observation = _message(jnp.asarray([1.0]), jnp.asarray([1.0]))
    gradient = jax.grad(lambda inst: instrument_loss(inst, raw_observation, None))(inherited)
    # Descent raises the precision-weighted mean towards the observed mean.
    assert gradient.mean_times_precision[0] < 0
