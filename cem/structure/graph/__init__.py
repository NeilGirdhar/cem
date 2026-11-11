"""Model and node interfaces, parameter types, and parameter-specific gradient transforms."""

from cem.structure.graph.disassembled import (
    DisGradientState,
    DisGradientTransformation,
    DisModel,
    ParameterType,
    verify_model_has_no_free_parameters,
)
from cem.structure.graph.eligibility_training import EligibilityTraining
from cem.structure.graph.model import Model, ModelResult
from cem.structure.graph.node import NodeConfiguration, TargetConfiguration, TargetNode
from cem.structure.graph.parameters import (
    FixedParameter,
    LearnableParameter,
    LearnableParameterFactory,
    LossBalancedParameter,
    Parameter,
    SummedGradientParameter,
    apply_to_parameters,
    count_real_learnable_parameters,
    is_parameter,
)
from cem.structure.graph.surrogate import cotangent_surrogate_loss

__all__ = [
    "DisGradientState",
    "DisGradientTransformation",
    "DisModel",
    "EligibilityTraining",
    "FixedParameter",
    "LearnableParameter",
    "LearnableParameterFactory",
    "LossBalancedParameter",
    "Model",
    "ModelResult",
    "NodeConfiguration",
    "Parameter",
    "ParameterType",
    "SummedGradientParameter",
    "TargetConfiguration",
    "TargetNode",
    "apply_to_parameters",
    "cotangent_surrogate_loss",
    "count_real_learnable_parameters",
    "is_parameter",
    "verify_model_has_no_free_parameters",
]
