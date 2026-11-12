"""The training objective carried by SIP circuit outputs."""

import equinox as eqx
from tjax import JaxRealArray


class SIPLossOutputs(eqx.Module):
    """Expose a circuit's scalar training objective.

    A composite circuit's loss is the sum of its parts' losses.
    """

    loss: JaxRealArray
