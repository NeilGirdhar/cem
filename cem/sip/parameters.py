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


class GoalObservationParameter(LossBalancedParameter[JaxRealArray]):
    """An observation-model parameter balanced over its feature's goal components.

    Each subclass names the components of one kind of goal.
    """

    loss_ids: ClassVar[tuple[str, ...]]


class DissonanceObservationParameter(GoalObservationParameter):
    """An observation-model parameter of a feature whose goal is dissonance or disutility.

    Its losses are the goal components, each bundling that component's observation credit.
    """

    # Reconstruction is MTAdam's anchor, so the balanced steps follow its gradient magnitude.
    loss_ids = ("reconstruction", "confounding")


class RewardObservationParameter(GoalObservationParameter):
    """An observation-model parameter of an intention, whose goal is reward.

    Its losses are the goal components, each bundling that component's observation credit.
    """

    # Environment reward is MTAdam's anchor, so the balanced steps follow its gradient magnitude.
    loss_ids = ("reward", "curiosity", "tracking")


class WitnessMultiplierParameter(LearnableParameter[JaxRealArray]):
    """A witness multiplier, projected onto the nonnegative reals after each update."""
