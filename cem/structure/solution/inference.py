from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
from functools import partial
from typing import Self, cast

import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
from jax import grad, tree, vmap
from tjax import JaxAbstractClass, JaxArray, KeyArray, create_streams
from tjax.dataclasses import as_shallow_dict

from cem.structure.graph.disassembled import DisGradientState, DisGradientTransformation, DisModel
from cem.structure.graph.model import Model, ModelConfiguration
from cem.structure.graph.parameters import is_parameter
from cem.structure.problem.data_source import DataSource, ProblemState
from cem.structure.problem.problem import Problem


class InferenceResult(eqx.Module):
    """The state of one batch of inference iterates between episodes.

    The inital problem state can help with plotting.  The other fields are final values.
    """

    initial_problem_state: ProblemState
    model_configuration: ModelConfiguration
    state: eqx.nn.State


class TrainingResult(eqx.Module):
    """The state of one batch of training iterates between episodes."""

    solution_state: SolutionState
    inference_result: InferenceResult


class SolutionState(eqx.Module):
    """This class is the iterand for the SolutionTrainer during training."""

    dis_learnable_parameters: DisModel
    gradient_states: DisGradientState

    @classmethod
    def create(
        cls,
        gradient_transformations: DisGradientTransformation,
        dis_learnable_parameters: DisModel,
    ) -> Self:
        gradient_states = gradient_transformations.init(dis_learnable_parameters)
        return cls(dis_learnable_parameters, gradient_states)


class _InferenceState(eqx.Module):
    """The input of batch inference.

    This is also used by RLInference to manage the state.
    """

    # These fields have shape ().
    example_key: KeyArray
    inference_key: KeyArray
    # These fields have shape (batch_size,)
    model_configuration: ModelConfiguration
    state: eqx.nn.State  # The state stores every variable marked as eqx.nn.StateIndex.


class _InferOutput(eqx.Module):
    """The output of the infer function."""

    model_configuration: ModelConfiguration
    state: eqx.nn.State


class _TrainingState(eqx.Module):
    """The input and output of batch training."""

    solution_state: SolutionState
    inference_state: _InferenceState


class Inference(eqx.Module, JaxAbstractClass):
    """This class iterates the parameters of the model.

    Each step of the iteration corresponds to a single step of a reinforcement learning trajectory.
    An RLInference object is stored by SolutionTrainer, which is an instance IteratedFunction.
    """

    fixed_parameters: Model
    initial_memory: eqx.nn.State

    def infer_zero_episodes(
        self,
        data_source: DataSource,
        problem: Problem,
    ) -> InferenceResult:
        """Like infer_one_episode, but for zero episodes.

        This is useful when infer_one_episode produces a bad result, but you still need a result.
        """
        key = jr.key(0)
        inference_state, problem_state = self._set_up_inference(1, key, key, data_source, problem)
        return InferenceResult(
            problem_state, inference_state.model_configuration, inference_state.state
        )

    def infer_one_episode(
        self,
        batch_size: int,
        example_key: KeyArray,
        inference_key: KeyArray,
        data_source: DataSource,
        learnable_parameters: Model,
        problem: Problem,
        *,
        return_samples: bool,
    ) -> InferenceResult:
        """Start batch_size parallel agents and run them for one episode.

        The models all share parameters, which are held constant.
        """
        inference_state, problem_state = self._set_up_inference(
            batch_size, example_key, inference_key, data_source, problem
        )
        body_function = partial(
            self._inference_body_fun,
            batch_size,
            learnable_parameters,
            return_samples=return_samples,
        )
        return self._infer_one_episode(
            learnable_parameters, problem_state, problem, inference_state, body_function
        )

    def train_one_episode(
        self,
        batch_size: int,
        example_key: KeyArray,
        inference_key: KeyArray,
        gradient_transformations: DisGradientTransformation,
        data_source: DataSource,
        problem: Problem,
        solution_state: SolutionState,
        *,
        return_samples: bool,
    ) -> TrainingResult:
        """Start batch_size parallel agents and run them for one episode.

        The models all share parameters, which trained after every time step.
        """
        inference_state, problem_state = self._set_up_inference(
            batch_size, example_key, inference_key, data_source, problem
        )
        training_state = _TrainingState(solution_state, inference_state)
        training_body_function = partial(
            self._training_body_fun,
            batch_size,
            gradient_transformations,
            return_samples=return_samples,
        )
        new_training_state = self._train_one_episode(
            problem, training_state, training_body_function
        )
        new_inference_state = new_training_state.inference_state
        inference_result = InferenceResult(
            problem_state, new_inference_state.model_configuration, new_inference_state.state
        )
        return TrainingResult(new_training_state.solution_state, inference_result)

    def _infer_one_episode(
        self,
        learnable_parameters: Model,
        problem_state: ProblemState,
        problem: Problem,
        inference_state: _InferenceState,
        body_function: Callable[[_InferenceState], _InferOutput],
    ) -> InferenceResult:
        """Infer one episode.

        For Inference, this is just batch inference.  For RLInference, this runs a whole trajectory.
        """
        infer_outputs = body_function(inference_state)
        return InferenceResult(
            problem_state, infer_outputs.model_configuration, infer_outputs.state
        )

    def _train_one_episode(
        self,
        problem: Problem,
        training_state: _TrainingState,
        body_function: Callable[[_TrainingState], _TrainingState],
    ) -> _TrainingState:
        return body_function(training_state)

    def assemble_model(self, learnable_parameters: Model) -> Model:
        return eqx.combine(learnable_parameters, self.fixed_parameters, is_leaf=is_parameter)

    def _set_up_inference(
        self,
        batch_size: int,
        example_key: KeyArray,
        inference_key: KeyArray,
        data_source: DataSource,
        problem: Problem,
    ) -> tuple[_InferenceState, ProblemState]:
        model_configuration = vmap(
            self.fixed_parameters.dummy_configuration, axis_size=batch_size
        )()
        example_keys = jr.split(example_key, batch_size)

        def initial_problem_state_and_set_to_state(
            initial_memory: eqx.nn.State,
            example_key: KeyArray,
        ) -> tuple[eqx.nn.State, ProblemState]:
            problem_state = data_source.initial_problem_state(example_key)
            observation = problem.extract_observation(problem_state)
            state = self.fixed_parameters.set_input(as_shallow_dict(observation), initial_memory)
            return state, problem_state

        v_initial = vmap(initial_problem_state_and_set_to_state, in_axes=(None, 0))
        state, observation = v_initial(self.initial_memory, example_keys)
        inference_state = _InferenceState(example_key, inference_key, model_configuration, state)
        return inference_state, observation

    def _infer(
        self,
        inference_key: KeyArray,
        state: eqx.nn.State,
        learnable_parameters: Model,
        use_signal_noise: bool,  # noqa: FBT001
        return_samples: bool,  # noqa: FBT001
    ) -> tuple[JaxArray, _InferOutput]:
        keys = {"inference": inference_key}
        streams = create_streams(keys)
        model = self.assemble_model(learnable_parameters)
        model_inference_result = model.infer_one_time_step(
            streams, state, use_signal_noise=use_signal_noise, return_samples=return_samples
        )
        model_loss = model_inference_result.loss
        configuration = model_inference_result.configuration
        state = model_inference_result.state
        for batch_loss in model_inference_result.batch_losses:
            model_loss += batch_loss.loss(streams)
        infer_output = _InferOutput(configuration, state)
        return model_loss, infer_output

    def _v_infer(
        self,
        batch_size: int,
        inference_key: KeyArray,
        state: eqx.nn.State,
        learnable_parameters: Model,
        *,
        use_signal_noise: bool,
        return_samples: bool,
    ) -> tuple[JaxArray, _InferOutput]:
        inference_keys = jr.split(inference_key, batch_size)
        f = vmap(self._infer, in_axes=(0, 0, None, None, None), out_axes=(0, 0))
        model_losses, infer_outputs = f(
            inference_keys, state, learnable_parameters, use_signal_noise, return_samples
        )
        model_loss = jnp.mean(model_losses)
        return model_loss, infer_outputs

    def _v_infer_gradient(
        self,
        batch_size: int,
        inference_key: KeyArray,
        state: eqx.nn.State,
        learnable_parameters: Model,
        *,
        return_samples: bool,
    ) -> tuple[Model, _InferOutput]:
        bound_infer = partial(
            self._v_infer,
            batch_size,
            inference_key,
            state,
            use_signal_noise=True,
            return_samples=return_samples,
        )
        f = cast("Callable[[Model], tuple[Model, _InferOutput]]", grad(bound_infer, has_aux=True))
        return f(learnable_parameters)

    def _inference_body_fun(
        self,
        batch_size: int,
        learnable_parameters: Model,
        inference_state: _InferenceState,
        *,
        return_samples: bool,
    ) -> _InferOutput:
        _, infer_outputs = self._v_infer(
            batch_size,
            inference_state.inference_key,
            inference_state.state,
            learnable_parameters,
            use_signal_noise=False,
            return_samples=return_samples,
        )
        return infer_outputs

    def _training_body_fun(
        self,
        batch_size: int,
        gradient_transformations: DisGradientTransformation,
        training_state: _TrainingState,
        *,
        return_samples: bool,
    ) -> _TrainingState:
        inference_state = training_state.inference_state
        solution_state = training_state.solution_state

        # Infer parameter gradient.
        learnable_parameters_bar, infer_outputs = self._v_infer_gradient(
            batch_size,
            inference_state.inference_key,
            inference_state.state,
            solution_state.dis_learnable_parameters.assembled(),
            return_samples=return_samples,
        )

        # Update the model parameters.
        dis_learnable_parameters_bar = DisModel.create(
            learnable_parameters_bar, tuple(gradient_transformations.learnable_parameter_types())
        )
        new_dis_learnable_parameters_bar, new_gradient_states = gradient_transformations.update(
            dis_learnable_parameters_bar,
            solution_state.gradient_states,
            solution_state.dis_learnable_parameters,
        )
        new_dis_learnable_parameters = tree.map(
            jnp.add, solution_state.dis_learnable_parameters, new_dis_learnable_parameters_bar
        )

        # Return the new training state.
        new_inference_state = replace(
            inference_state,
            model_configuration=infer_outputs.model_configuration,
            state=infer_outputs.state,
        )
        return _TrainingState(
            SolutionState(new_dis_learnable_parameters, new_gradient_states), new_inference_state
        )
