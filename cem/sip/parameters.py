"""Parameter roles assigned during SIP network construction."""

from typing import ClassVar, Final

from tjax import JaxRealArray

from cem.structure.graph import LearnableParameter, LossBalancedParameter


class MismatchPredictionParameter(LossBalancedParameter[JaxRealArray]):
    """A mismatch prediction parameter, balanced over reconstruction and confounding."""

    # Reconstruction is MTAdam's anchor, so the balanced steps follow its gradient magnitude.
    loss_ids: ClassVar[Final[tuple[str, ...]]] = ("reconstruction", "confounding")


class TDPredictionParameter(LossBalancedParameter[JaxRealArray]):
    """A TD prediction parameter, balanced over reconstruction, confounding, and uncertainty."""

    # Reconstruction is MTAdam's anchor, so the balanced steps follow its gradient magnitude.
    loss_ids: ClassVar[Final[tuple[str, ...]]] = ("reconstruction", "confounding", "uncertainty")


class WitnessMultiplierParameter(LearnableParameter[JaxRealArray]):
    """A witness multiplier, projected onto the nonnegative reals after each update."""
