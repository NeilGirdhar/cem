import jax.numpy as jnp
from efax import NormalNP, NormalPP

from cem.sip import InstrumentedMessage
from cem.sip.settings import CreditSettings, MismatchSettings

CREDIT_SETTINGS = CreditSettings(trace_decay_rate=0.1)
MISMATCH_SETTINGS = MismatchSettings()


def _message(mean: jnp.ndarray, precision: jnp.ndarray) -> NormalNP:
    return NormalPP(mean, precision).to_nat()


def _i_message(observation: NormalPP, instrument: NormalPP) -> InstrumentedMessage:
    return InstrumentedMessage(observation.to_nat(), instrument.to_nat())
