"""SIP explanatory coupling base class.

See `explanatory-couplings` in the thesis.
"""

import equinox as eqx


class ExplanatoryCoupling(eqx.Module):
    """A feature stack that exchanges predictions and mismatch between adjacent levels.

    See `explanatory-couplings` in the thesis.
    """
