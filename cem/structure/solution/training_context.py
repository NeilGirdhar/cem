from __future__ import annotations

import logging
from dataclasses import replace
from typing import Any

import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
from jax import tree
from tjax import Array, GenericString, JaxArray, KeyArray, PyTree, dynamic_tree_all, jit

from cem.structure.problem.data_source import DataSource
from cem.structure.solution.inference import SolutionState, TrainingResult

from .execution_context import ExecutionContext, ExecutionPacket
from .results import TrainingResults
from .telemetry import Telemetry
from .training_solution import TrainingSolution

log = logging.getLogger(__name__)


def all_finite(x: Array, /) -> JaxArray:
    return jnp.all(jnp.isfinite(x))


def is_all_finite_tree(x: PyTree, /) -> JaxArray:
    return dynamic_tree_all(tree.map(all_finite, x))


def training_snapshots(
    solution: TrainingSolution,
    telemetries: tuple[Telemetry, ...],
    result: TrainingResult,
) -> dict[Telemetry, Any]:
    snapshots: dict[Telemetry, Any] = {}
    for telemetry in telemetries:
        snapshot = telemetry.training_snapshot(solution, result, snapshots)
        if snapshot is not None:
            snapshots[telemetry] = snapshot
    return snapshots


def train_one_episode(
    solution: TrainingSolution,
    batch_size: int,
    example_key: KeyArray,
    inference_key: KeyArray,
    data_source: DataSource,
    telemetries: tuple[Telemetry, ...],
    *,
    return_samples: bool,
) -> tuple[SolutionState, dict[Telemetry, Any], JaxArray, JaxArray]:
    result = solution.inference.train_one_episode(
        batch_size,
        example_key,
        inference_key,
        solution.gradient_transformations,
        data_source,
        solution.problem,
        solution.solution_state,
        return_samples=return_samples,
    )
    snapshots = training_snapshots(solution, telemetries, result)
    finite_parameters = is_all_finite_tree(result.solution_state.dis_learnable_parameters)
    finite_inference_result = is_all_finite_tree(result.inference_result)
    return result.solution_state, snapshots, finite_parameters, finite_inference_result


j_train_one_episode = jit(train_one_episode, static_argnames=("batch_size", "return_samples"))


def train_episodes(
    solver_name: str | None,
    batch_size: int,
    solution: TrainingSolution,
    packet: ExecutionPacket,
    key: KeyArray,
    episodes: int,
) -> TrainingResults:
    """Train episodes."""
    log.info("Training")
    solution_state = solution.solution_state
    data_source = solution.problem.create_data_source()
    default_result = TrainingResult(
        solution_state, solution.inference.infer_zero_episodes(data_source, solution.problem)
    )
    default_snapshots = training_snapshots(solution, packet.telemetries.telemetries, default_result)
    example_key_base, inference_key_base = jr.split(key)
    example_keys = jr.split(example_key_base, episodes)
    inference_keys = jr.split(inference_key_base, episodes)
    with ExecutionContext.create(
        solver_name=solver_name,
        default_snapshots=default_snapshots,
        episodes=episodes,
        packet=packet,
        job_type="training",
        use_wandb=True,
    ) as execution_context:
        for example_key, inference_key in zip(example_keys, inference_keys, strict=True):
            (solution_state, snapshots, finite_parameters, finite_inference_result) = (
                j_train_one_episode(
                    solution,
                    batch_size,
                    example_key,
                    inference_key,
                    data_source,
                    packet.telemetries.telemetries,
                    return_samples=False,
                )
            )
            solution = replace(solution, solution_state=solution_state)
            if not finite_parameters or not finite_inference_result:
                if not finite_parameters:
                    infinite_parameters = eqx.filter(
                        solution_state.dis_learnable_parameters, all_finite, inverse=True
                    )
                    log.error(
                        "Non-finite parameters encountered at example "
                        f"{execution_context.episodes_done()} at these elements:"
                    )
                    log.info(GenericString(infinite_parameters))
                if not finite_inference_result:
                    result = solution.inference.train_one_episode(
                        batch_size,
                        example_key,
                        inference_key,
                        solution.gradient_transformations,
                        data_source,
                        solution.problem,
                        solution_state,
                        return_samples=False,
                    )
                    infinite_inference_result = eqx.filter(
                        result.inference_result, all_finite, inverse=True
                    )
                    log.error(
                        "Non-finite configuration encountered at example "
                        f"{execution_context.episodes_done()} at these elements:"
                    )
                    log.info(GenericString(infinite_inference_result))
                break
            execution_context.append_result(snapshots)
    return TrainingResults(
        execution_context.episodes_done(), execution_context.telemetries(), solution_state
    )
