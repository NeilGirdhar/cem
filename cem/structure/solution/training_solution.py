"""Bundle a problem, its model partitions, and the state needed to train it."""

import equinox as eqx

from cem.structure.graph import (
    DisGradientTransformation,
    DisModel,
    Model,
    is_parameter,
    verify_model_has_no_free_parameters,
)
from cem.structure.problem import Problem
from cem.structure.solution.inference import Inference, SolutionState


class TrainingSolution(eqx.Module):
    """A problem and its trainable solution, including the optimizer and current state."""

    gradient_transformations: DisGradientTransformation
    inference: Inference
    problem: Problem
    solution_state: SolutionState

    @classmethod
    def create(
        cls,
        problem: Problem,
        model: Model,
        gradient_transformations: DisGradientTransformation,
    ) -> TrainingSolution:
        """Split the model into fixed and trainable parameters and initialize optimizer state."""
        verify_model_has_no_free_parameters(model)
        learnable_parameters, fixed_parameters = eqx.partition(
            model,
            lambda x: gradient_transformations.is_learnable(x),  # ruff:ignore[unnecessary-lambda]
            is_leaf=is_parameter,
        )
        dissassembled = DisModel.create(
            learnable_parameters, tuple(gradient_transformations.learnable_parameter_types())
        )
        return TrainingSolution(
            gradient_transformations,
            Inference(fixed_parameters),
            problem,
            SolutionState.create(gradient_transformations, dissassembled),
        )

    def assemble_model(
        self,
        *,
        fixed_parameters: bool,
        learnable_parameters: bool,
    ) -> Model:
        """Return the requested parameter partition or the complete model if both are requested."""
        assert fixed_parameters or learnable_parameters
        if not learnable_parameters:
            return self.inference.fixed_parameters
        learnable_parameters_ = self.solution_state.dis_learnable_parameters.assembled()
        if not fixed_parameters:
            return learnable_parameters_
        return self.inference.assemble_model(learnable_parameters_)
