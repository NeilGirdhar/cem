"""SIP explanatory coupling base class.

See `explanatory-couplings` in `~/src/typst/thesis.typ`.
"""

import equinox as eqx

from cem.sip.features import SIPFeature


class ExplanatoryCoupling(eqx.Module):
    """A stack of `SIPFeature`s, one per level.

    Subclassed by `Sensations` (bottom-up) and `Intentions` (top-down), which
    wire the stack's observation-mismatch flow and set per-feature
    controllability. The base class only holds the features; subclasses
    supply their own `create` and `infer` methods.
    """

    features: tuple[SIPFeature, ...]
