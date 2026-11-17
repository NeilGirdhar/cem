import equinox as eqx
import jax
import jax.numpy as jnp
from efax import NormalNP, NormalPP

from cem.sip import InstrumentedMessage
from cem.sip.settings import MismatchSettings

MISMATCH_SETTINGS = MismatchSettings()


def _message(mean: jnp.ndarray, precision: jnp.ndarray) -> NormalNP:
    return NormalPP(mean, precision).to_nat()


def _i_message(observation: NormalPP, instrument: NormalPP) -> InstrumentedMessage:
    return InstrumentedMessage(observation.to_nat(), instrument.to_nat())


def _clone(state: eqx.nn.State) -> eqx.nn.State:
    """Return a copy of a state, so a test can advance it more than once."""
    leaves, treedef = jax.tree_util.tree_flatten(state)
    return jax.tree_util.tree_unflatten(treedef, leaves)
