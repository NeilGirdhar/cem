from __future__ import annotations

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
    """TrainingSolution is everything needed to keep track of the solution during training.

    Attributes:
        inference: The model and the problem.
        gradient_transformation: The way in which the state is updated.
        solution_state: The state of parameters, the gradient transformation, and the RNG.
    """

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
        verify_model_has_no_free_parameters(model)
        learnable_parameters, fixed_parameters = eqx.partition(
            model,
            lambda x: gradient_transformations.is_learnable(x),  # noqa: PLW0108
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
        assert fixed_parameters or learnable_parameters
        if not learnable_parameters:
            return self.inference.fixed_parameters
        learnable_parameters_ = self.solution_state.dis_learnable_parameters.assembled()
        if not fixed_parameters:
            return learnable_parameters_
        return self.inference.assemble_model(learnable_parameters_)
