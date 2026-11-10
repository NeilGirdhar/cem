"""SIP inherited instrument: the first-stage prediction of a feature's raw observation."""

import jax
import jax.numpy as jnp
from efax import NormalNP
from tjax import JaxRealArray

from cem.npn.gaussian import GaussianNPN


def inherit_instrument(instrument_map: GaussianNPN, predictor_instrument: NormalNP) -> NormalNP:
    """Return the inherited instrument that the map predicts from the parents' instruments.

    Hold the parents' instruments fixed so that each feature's instrument mismatch trains only
    its own instrument map. See `sip-inherited-variation` in the thesis.
    """
    return instrument_map.infer(jax.lax.stop_gradient(predictor_instrument))


def instrument_mismatch_loss(
    inherited_instrument: NormalNP,
    raw_observation: NormalNP,
    gain: JaxRealArray | None,
) -> JaxRealArray:
    """Return the gain-gated cross-entropy of the raw observation under the inherited instrument.

    Descending this loss fits the instrument map to the raw observation's distribution given the
    parents' instruments. It holds the raw observation and the gain fixed, so it cannot move the
    observation towards the instrument or favour reducing participation. Without a gain, every
    channel has unit gain. See `sip-inherited-variation` and `sip-emitter-gain` in the thesis.

    Args:
        inherited_instrument: The inherited instrument before local noise and gain.
        raw_observation: The raw observation before local noise and gain, held fixed.
        gain: The gain observation's means, shape (n,), held fixed, or None.
    """
    cross_entropy = (
        jax.lax.stop_gradient(raw_observation).to_exp().cross_entropy(inherited_instrument)
    )
    if gain is not None:
        cross_entropy = jax.lax.stop_gradient(gain) * cross_entropy
    return jnp.sum(cross_entropy)
