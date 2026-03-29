from __future__ import annotations

from collections.abc import Callable
from functools import partial
from typing import override

import jax.numpy as jnp
import jax.random as jr
from equinox.internal import while_loop
from jax.lax import stop_gradient
from tjax import JaxArray, JaxBooleanArray, JaxIntegralArray, KeyArray
from tjax.dataclasses import as_shallow_dict

from cem.structure.graph.model import Model
from cem.structure.problem.data_source import DataSource, ProblemState
from cem.structure.problem.problem import Problem
from cem.structure.solution.inference import Inference, InferenceResult
from cem.structure.solution.inference import _InferenceState, _InferOutput, _TrainingState

from .problem import ProblemAction, ProblemReward, RLProblem


class _RLInferenceState(_InferenceState):
    """The state of one batch of inference iterates between time steps."""

    # These fields have shape ().
    step: JaxIntegralArray
    total_reward: JaxArray
    done: JaxBooleanArray
    # These fields have shape (batch_size,)
    problem_state: ProblemState


class RLInference(Inference):
    """This class iterates the parameters of the model.

    Each step of the iteration corresponds to a single step of a reinforcement learning trajectory.
    An RLInference object is stored by SolutionTrainer, which is an instance IteratedFunction.
    """

    @override
    def _set_up_inference(
        self,
        batch_size: int,
        example_key: KeyArray,
        inference_key: KeyArray,
        data_source: DataSource,
        problem: Problem,
    ) -> tuple[_RLInferenceState, ProblemState]:
        inference_state, problem_state = super()._set_up_inference(
            batch_size, example_key, inference_key, data_source, problem
        )
        z = jnp.zeros(())
        rl_inference_state = _RLInferenceState(
            inference_state.example_key,
            inference_state.inference_key,
            inference_state.model_configuration,
            inference_state.state,
            z,
            z,
            jnp.asarray(False),  # noqa: FBT003
            problem_state,
        )
        return rl_inference_state, problem_state

    @override
    def _infer_one_episode(
        self,
        learnable_parameters: Model,
        problem_state: ProblemState,
        problem: Problem,
        inference_state: _InferenceState,
        body_function: Callable[[_InferenceState], _InferOutput],
    ) -> InferenceResult:
        assert isinstance(problem, RLProblem)
        assert isinstance(inference_state, _RLInferenceState)
        rl_body_function = partial(
            self._rl_inference_body_fun, body_function, problem, learnable_parameters
        )
        inference_state = while_loop(
            self._rl_inference_cond_fun,
            rl_body_function,
            inference_state,
            max_steps=problem.max_episode_steps(),  # ty: ignore
            kind="lax",
        )
        return InferenceResult(
            problem_state, inference_state.model_configuration, inference_state.state
        )

    @override
    def _train_one_episode(
        self,
        problem: Problem,
        training_state: _TrainingState,
        body_function: Callable[[_TrainingState], _TrainingState],
    ) -> _TrainingState:
        assert isinstance(problem, RLProblem)
        return while_loop(
            self._rl_training_cond_fun,
            body_function,
            training_state,
            max_steps=problem.max_episode_steps(),  # ty: ignore
            kind="lax",
        )

    def _rl_inference_cond_fun(self, inference_state: _RLInferenceState) -> JaxBooleanArray:
        return jnp.logical_not(inference_state.done)

    def _rl_training_cond_fun(self, training_state: _TrainingState) -> JaxBooleanArray:
        inference_state = training_state.inference_state
        assert isinstance(inference_state, _RLInferenceState)
        return self._rl_inference_cond_fun(inference_state)

    def _rl_inference_body_fun(
        self,
        body_function: Callable[[_InferenceState], _InferOutput],
        problem: RLProblem[ProblemState, ProblemAction, ProblemReward],
        learnable_parameters: Model,
        inference_state: _RLInferenceState,
    ) -> _RLInferenceState:
        # Run one time step.
        infer_outputs = body_function(inference_state)
        # Extract action.
        model = self.assemble_model(learnable_parameters)
        action_fields = model.get_output(
            problem.action_distributions().keys(), infer_outputs.model_configuration
        )
        action = problem.produce_action(action_fields, inference_state.example_key)
        action = stop_gradient(action)
        # Update environment.
        new_problem_state, reward, new_done = problem.iterate_state(
            inference_state.problem_state, action
        )
        new_problem_state = stop_gradient(new_problem_state)
        # Inject new observation.
        new_observation = problem.extract_observation(new_problem_state)
        new_state = model.set_input(as_shallow_dict(new_observation), infer_outputs.state)
        # Produce result.
        new_step = inference_state.step + 1
        new_total_reward = inference_state.total_reward + reward.total_reward()
        (new_example_key,) = jr.split(inference_state.example_key, 1)
        (new_inference_key,) = jr.split(inference_state.inference_key, 1)
        return _RLInferenceState(
            new_example_key,
            new_inference_key,
            infer_outputs.model_configuration,
            new_state,
            new_step,
            new_total_reward,
            new_done,
            new_problem_state,
        )
