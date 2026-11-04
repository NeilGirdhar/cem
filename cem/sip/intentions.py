"""SIP intention coupling: top-down stack of intention features.

See `intention-coupling` in `~/src/typst/thesis.typ`.
"""

from cem.sip.explanatory_coupling import ExplanatoryCoupling


class Intentions(ExplanatoryCoupling):
    """Top-down explanatory coupling (placeholder).

    The intention stack mirrors `Sensations` with mismatch flowing downward
    and controllable features that inject local noise. Its `create` and
    `infer` methods are not yet implemented; see the sip plan.
    """
