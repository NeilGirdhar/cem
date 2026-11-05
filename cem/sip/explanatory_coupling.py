"""SIP explanatory coupling base class."""

import equinox as eqx


class ExplanatoryCoupling(eqx.Module):
    """Base class for feature stacks that exchange predictions and mismatch across levels.

    See `explanatory-couplings` in the thesis.
    """
