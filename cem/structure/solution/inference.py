"""Batched model steps and the optimizer state carried between training episodes."""

from collections.abc import Callable
from dataclasses import replace
from functools import partial
from typing import Any, Self

import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
from jax import tree, vmap
from tjax import JaxAbstractClass, JaxArray, KeyArray, create_streams

from cem.structure.graph.disassembled import DisGradientState, DisGradientTransformation, DisModel
from cem.structure.graph.losses import LossGradients, differentiate_losses
from cem.structure.graph.model import Model
from cem.structure.graph.node import NodeConfiguration
from cem.structure.graph.parameters import is_parameter
from cem.structure.problem.data_source import DataSource, ProblemState
from cem.structure.problem.problem import Problem


class InferenceResult(eqx.Module):
    """The batch's initial problem states and evaluated node outputs."""

    initial_problem_state: ProblemState
    model_configuration: frozendict[str, NodeConfiguration]


class SolutionState(eqx.Module):
    """The learnable parameter partitions and their current optimizer states."""

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


class TrainingResult(eqx.Module):
    """Updated parameters and optimizer state together with the batch's evaluated node outputs."""

    solution_state: SolutionState
    inference_result: InferenceResult


class _InferenceState(eqx.Module):
    """Random keys, observations, and model state carried through a batched episode."""

    # These fields have shape ().
    example_key: KeyArray
    inference_key: KeyArray
    # These fields have shape (batch_size, ...).
    state: Any  # model's recurrent state
    observation: Any  # current environment observation


class _TrainingState(eqx.Module):
    """The model and optimizer state carried through a training step."""

    solution_state: SolutionState
    inference_state: _InferenceState


class Inference(eqx.Module, JaxAbstractClass):
    """Run batched model evaluation and training with fixed parameters shared across episodes."""

    fixed_parameters: Model

    def infer_one_episode(
        self,
        batch_size: int,
        example_key: KeyArray,
        inference_key: KeyArray,
        data_source: DataSource,
        learnable_parameters: Model,
        problem: Problem,
    ) -> InferenceResult:
        """Sample and evaluate a batch without updating model parameters."""
        inference_state, problem_state = self._set_up_inference(
            batch_size, example_key, inference_key, data_source, problem
        )
        body_function = partial(self._inference_body_fun, batch_size, learnable_parameters)
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
    ) -> TrainingResult:
        """Train on one batch and evaluate its node outputs with the updated parameters."""
        inference_state, problem_state = self._set_up_inference(
            batch_size, example_key, inference_key, data_source, problem
        )
        training_state = _TrainingState(solution_state, inference_state)
        training_body_function = partial(
            self._training_body_fun, batch_size, gradient_transformations
        )
        new_training_state = self._train_one_episode(
            problem, training_state, training_body_function
        )
        new_inference_state = new_training_state.inference_state
        new_solution_state = new_training_state.solution_state
        _, configurations = self._inference_body_fun(
            batch_size,
            new_solution_state.dis_learnable_parameters.assembled(),
            new_inference_state,
        )
        inference_result = InferenceResult(problem_state, frozendict(configurations))
        return TrainingResult(new_training_state.solution_state, inference_result)

    def _infer_one_episode(
        self,
        learnable_parameters: Model,
        problem_state: ProblemState,
        problem: Problem,
        inference_state: _InferenceState,
        body_function: Callable[[_InferenceState], tuple[Any, frozendict[str, NodeConfiguration]]],
    ) -> InferenceResult:
        """Run a single forward step by default. Rollout subclasses can advance the environment."""
        _, configurations = body_function(inference_state)
        return InferenceResult(problem_state, frozendict(configurations))

    def _train_one_episode(
        self,
        problem: Problem,
        training_state: _TrainingState,
        body_function: Callable[[_TrainingState], _TrainingState],
    ) -> _TrainingState:
        """Run one update by default. Rollout subclasses can perform several updates."""
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
        """Sample initial observations and replicate the model's initial state across the batch."""
        example_keys = jr.split(example_key, batch_size)
        initial_model_state = self.fixed_parameters.initial_state()

        def make_one(example_key: KeyArray) -> tuple[Any, ProblemState]:
            problem_state = data_source.initial_problem_state(example_key)
            observation = problem.extract_observation(problem_state)
            return observation, problem_state

        observations, problem_states = vmap(make_one)(example_keys)
        states = vmap(lambda _: initial_model_state)(example_keys)
        inference_state = _InferenceState(example_key, inference_key, states, observations)
        return inference_state, problem_states

    def _infer(
        self,
        inference_key: KeyArray,
        observation: object,
        state: object,
        learnable_parameters: Model,
        inference: bool,  # ruff:ignore[boolean-type-hint-positional-argument]
        loss_ids: tuple[str, ...],
    ) -> tuple[tuple[dict[str, JaxArray], JaxArray], Any, frozendict[str, NodeConfiguration]]:
        streams = create_streams({"inference": inference_key})
        model = self.assemble_model(learnable_parameters)
        result = model.infer(observation, state, streams=streams, inference=inference)
        missing = set(loss_ids) - result.balanced_losses.keys()
        if missing:
            msg = f"The model did not return the requested losses: {sorted(missing)}"
            raise ValueError(msg)
        named = {name: result.balanced_losses[name] for name in loss_ids}
        remaining = sum(
            (loss for name, loss in result.balanced_losses.items() if name not in named),
            result.fixed_weight_loss,
        )
        return (named, remaining), result.state, result.configurations

    def _v_infer(
        self,
        batch_size: int,
        inference_key: KeyArray,
        observation: object,
        state: object,
        learnable_parameters: Model,
        *,
        inference: bool,
        loss_ids: tuple[str, ...] = (),
    ) -> tuple[tuple[dict[str, JaxArray], JaxArray], Any, frozendict[str, NodeConfiguration]]:
        """Evaluate each example with its own key and average each requested objective."""
        inference_keys = jr.split(inference_key, batch_size)
        f = vmap(self._infer, in_axes=(0, 0, 0, None, None, None), out_axes=(0, 0, 0))
        losses, new_states, configurations = f(
            inference_keys, observation, state, learnable_parameters, inference, loss_ids
        )
        return tree.map(jnp.mean, losses), new_states, configurations

    def _v_infer_gradient(
        self,
        batch_size: int,
        inference_key: KeyArray,
        observation: object,
        state: object,
        learnable_parameters: Model,
        *,
        loss_ids: tuple[str, ...] = (),
    ) -> tuple[LossGradients[Model], tuple[Any, frozendict[str, NodeConfiguration]]]:
        """Differentiate the batch objectives and retain the forward pass's outputs."""

        def forward(
            params: Model,
        ) -> tuple[
            tuple[dict[str, JaxArray], JaxArray], tuple[Any, frozendict[str, NodeConfiguration]]
        ]:
            losses, new_state, configurations = self._v_infer(
                batch_size,
                inference_key,
                observation,
                state,
                params,
                inference=False,
                loss_ids=loss_ids,
            )
            return losses, (new_state, configurations)

        return differentiate_losses(forward, learnable_parameters)

    def _inference_body_fun(
        self,
        batch_size: int,
        learnable_parameters: Model,
        inference_state: _InferenceState,
    ) -> tuple[Any, frozendict[str, NodeConfiguration]]:
        _, new_state, configurations = self._v_infer(
            batch_size,
            inference_state.inference_key,
            inference_state.observation,
            inference_state.state,
            learnable_parameters,
            inference=True,
        )
        return new_state, configurations

    def _training_body_fun(
        self,
        batch_size: int,
        gradient_transformations: DisGradientTransformation,
        training_state: _TrainingState,
    ) -> _TrainingState:
        """Advance the batch's model state and apply the parameter-specific optimizer updates."""
        inference_state = training_state.inference_state
        solution_state = training_state.solution_state

        learnable_parameters_bar, (new_state, _) = self._v_infer_gradient(
            batch_size,
            inference_state.inference_key,
            inference_state.observation,
            inference_state.state,
            solution_state.dis_learnable_parameters.assembled(),
            loss_ids=gradient_transformations.requested_loss_ids(),
        )
        parameter_types = tuple(gradient_transformations.learnable_parameter_types())

        def partition(gradient: Model) -> DisModel:
            # Conjugate JAX's complex gradients so the optimizer follows steepest descent.
            return DisModel.create(tree.map(jnp.conj, gradient), parameter_types)

        dis_learnable_parameters_bar = LossGradients(
            {
                name: partition(gradient)
                for name, gradient in learnable_parameters_bar.balanced_gradients.items()
            },
            partition(learnable_parameters_bar.fixed_weight_gradient),
        )
        new_dis_learnable_parameters_bar, new_gradient_states = gradient_transformations.update(
            dis_learnable_parameters_bar,
            solution_state.gradient_states,
            solution_state.dis_learnable_parameters,
        )
        new_dis_learnable_parameters = tree.map(
            jnp.add, solution_state.dis_learnable_parameters, new_dis_learnable_parameters_bar
        )

        new_inference_state = replace(inference_state, state=new_state)
        return _TrainingState(
            SolutionState(new_dis_learnable_parameters, new_gradient_states), new_inference_state
        )
