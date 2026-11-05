import jax.numpy as jnp
from efax import NormalNP

from cem.npn.gaussian import normal_from_mean_and_presence
from cem.sip import InstrumentedMessage


def _message(mean: jnp.ndarray, precision: jnp.ndarray) -> NormalNP:
    return normal_from_mean_and_presence(mean, precision)


def _i_message(
    obs_mean: jnp.ndarray,
    obs_precision: jnp.ndarray,
    ins_mean: jnp.ndarray,
    ins_precision: jnp.ndarray,
) -> InstrumentedMessage:
    return InstrumentedMessage(_message(obs_mean, obs_precision), _message(ins_mean, ins_precision))
