"""Parameter roles assigned during SIP network construction."""

from typing import ClassVar, Final

from tjax import JaxRealArray

from cem.structure.graph import LossBalancedParameter


class MismatchPredictionParameter(LossBalancedParameter[JaxRealArray]):
    """A mismatch prediction parameter, balanced over reconstruction and confounding."""

    # Reconstruction comes first because MTAdam takes the first loss as its reference scale.
    loss_ids: ClassVar[Final[tuple[str, ...]]] = ("reconstruction", "confounding")
