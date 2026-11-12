"""Parameter roles assigned during SIP network construction."""

from tjax import JaxRealArray

from cem.framework.graph import LearnableParameter


class WitnessMultiplierParameter(LearnableParameter[JaxRealArray]):
    """A witness multiplier, projected onto the nonnegative reals after each update."""
