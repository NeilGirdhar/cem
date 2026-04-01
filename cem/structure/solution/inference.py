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
    """Result of running inference for one batch.

    Stores the initial problem state together with the final model configuration
    derived from the resulting model state.
    """

    initial_problem_state: ProblemState
    model_configuration: ModelConfiguration


class TrainingResult(eqx.Module):
    """Result of running one training episode for one batch."""

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
    """Internal state carried through batched inference.

    Holds RNG keys and the model state for the current batch. RLInference extends
    this with environment-specific rollout state.
    """

    # These fields have shape ().
    example_key: KeyArray
    inference_key: KeyArray
    # These fields have shape (batch_size,)
    state: eqx.nn.State  # The state stores every variable marked as eqx.nn.StateIndex.


class _TrainingState(eqx.Module):
    """The input and output of batch training."""

    solution_state: SolutionState
    inference_state: _InferenceState


class Inference(eqx.Module, JaxAbstractClass):
    """Base class for batched model inference and training.

    The base implementation performs a single batched inference or training step.
    Subclasses such as RLInference can override this to run multi-step episodes.
    """

    fixed_parameters: Model
    initial_memory: eqx.nn.State

    def infer_zero_episodes(
        self,
        data_source: DataSource,
        problem: Problem,
    ) -> InferenceResult:
        """Initialize inference state without running any model steps."""
        key = jr.key(0)
        inference_state, problem_state = self._set_up_inference(1, key, key, data_source, problem)
        model_configuration = self.fixed_parameters.configuration_from_state(inference_state.state)
        return InferenceResult(problem_state, model_configuration)

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
        """Run one batched inference episode with shared model parameters.

        In the base class this is a single batched forward pass; subclasses may
        interpret an episode as a multi-step rollout.
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
        """Run one batched training episode.

        All batch elements share parameters, and the training body may update
        those parameters during the episode.
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
        model_configuration = self.fixed_parameters.configuration_from_state(
            new_inference_state.state
        )
        inference_result = InferenceResult(problem_state, model_configuration)
        return TrainingResult(new_training_state.solution_state, inference_result)

    def _infer_one_episode(
        self,
        learnable_parameters: Model,
        problem_state: ProblemState,
        problem: Problem,
        inference_state: _InferenceState,
        body_function: Callable[[_InferenceState], eqx.nn.State],
    ) -> InferenceResult:
        """Run the inference body for one episode.

        In the base class this is a single batched inference step. Subclasses
        such as RLInference can override this to execute a full rollout.
        """
        state = body_function(inference_state)
        return InferenceResult(
            problem_state,
            self.fixed_parameters.configuration_from_state(state),
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
        inference_state = _InferenceState(example_key, inference_key, state)
        return inference_state, observation

    def _infer(
        self,
        inference_key: KeyArray,
        state: eqx.nn.State,
        learnable_parameters: Model,
        use_signal_noise: bool,  # noqa: FBT001
        return_samples: bool,  # noqa: FBT001
    ) -> tuple[JaxArray, eqx.nn.State]:
        keys = {"inference": inference_key}
        streams = create_streams(keys)
        model = self.assemble_model(learnable_parameters)
        model_inference_result = model.infer_one_time_step(
            streams, state, use_signal_noise=use_signal_noise, return_samples=return_samples
        )
        model_loss = model_inference_result.loss
        state = model_inference_result.state
        for batch_loss in model_inference_result.batch_losses:
            model_loss += batch_loss.loss(streams)
        return model_loss, state

    def _v_infer(
        self,
        batch_size: int,
        inference_key: KeyArray,
        state: eqx.nn.State,
        learnable_parameters: Model,
        *,
        use_signal_noise: bool,
        return_samples: bool,
    ) -> tuple[JaxArray, eqx.nn.State]:
        inference_keys = jr.split(inference_key, batch_size)
        f = vmap(self._infer, in_axes=(0, 0, None, None, None), out_axes=(0, 0))
        model_losses, state = f(
            inference_keys, state, learnable_parameters, use_signal_noise, return_samples
        )
        model_loss = jnp.mean(model_losses)
        return model_loss, state

    def _v_infer_gradient(
        self,
        batch_size: int,
        inference_key: KeyArray,
        state: eqx.nn.State,
        learnable_parameters: Model,
        *,
        return_samples: bool,
    ) -> tuple[Model, eqx.nn.State]:
        bound_infer = partial(
            self._v_infer,
            batch_size,
            inference_key,
            state,
            use_signal_noise=True,
            return_samples=return_samples,
        )
        f = cast("Callable[[Model], tuple[Model, eqx.nn.State]]", grad(bound_infer, has_aux=True))
        return f(learnable_parameters)

    def _inference_body_fun(
        self,
        batch_size: int,
        learnable_parameters: Model,
        inference_state: _InferenceState,
        *,
        return_samples: bool,
    ) -> eqx.nn.State:
        _, state = self._v_infer(
            batch_size,
            inference_state.inference_key,
            inference_state.state,
            learnable_parameters,
            use_signal_noise=False,
            return_samples=return_samples,
        )
        return state

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
        learnable_parameters_bar, new_state = self._v_infer_gradient(
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
        new_inference_state = replace(inference_state, state=new_state)
        return _TrainingState(
            SolutionState(new_dis_learnable_parameters, new_gradient_states), new_inference_state
        )
