"""SIP explanatory coupling base class.

See `explanatory-couplings` in `~/src/typst/thesis.typ`.
"""

import equinox as eqx


class ExplanatoryCoupling(eqx.Module):
    """Marker base for the `Sensations` and `Intentions` feature stacks.

    The two stacks share a bottom-up / top-down symmetry but their layer
    types differ (a sensation input has no `observation_map` and no `Credit`;
    a future intention output may have its own specialization), so each
    subclass declares its own feature fields rather than inheriting a shared
    tuple here.
    """
