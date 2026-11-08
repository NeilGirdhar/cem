"""Factories that assign SIP parameter roles during network construction."""

from tjax import JaxRealArray

from cem.structure.graph import LossBalancedParameter


def advantage_parameter(value: JaxRealArray) -> LossBalancedParameter[JaxRealArray]:
    """Subscribe an intention observation parameter to the three sources of reward credit."""
    return LossBalancedParameter(value, loss_names=("reward", "curiosity", "tracking"))


def dissonance_parameter(value: JaxRealArray) -> LossBalancedParameter[JaxRealArray]:
    """Subscribe a parameter to reconstruction and confounding."""
    return LossBalancedParameter(value, loss_names=("confounding", "reconstruction"))


disutility_parameter = dissonance_parameter
mismatch_prediction_parameter = disutility_parameter


def td_error_prediction_parameter(value: JaxRealArray) -> LossBalancedParameter[JaxRealArray]:
    """Balance reconstruction, confounding, and uncertainty on the TD prediction map."""
    return LossBalancedParameter(value, loss_names=("confounding", "reconstruction", "uncertainty"))
