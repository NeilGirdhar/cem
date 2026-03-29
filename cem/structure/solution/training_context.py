from __future__ import annotations

import logging
from collections.abc import Generator
from dataclasses import replace
from typing import Any, override

import equinox as eqx
import jax.numpy as jnp
import rich.progress as rp
from jax import tree
from tjax import Array, GenericString, JaxArray, KeyArray, PyTree, dynamic_tree_all, jit
from wandb.sdk.wandb_run import Run

from cem.structure.model import DataSource, SolutionState, TrainingResult
from .execution_context import ExecutionContext, ExecutionPacket
from .results import TrainingResults
from .segment import segment_keys
from .telemetry import Telemetries, TrainingTelemetry
from .training_solution import TrainingSolution

log = logging.getLogger(__name__)


class TrainingSegment(eqx.Module):
    key: KeyArray  # Used for producing example and doing inference.
    episodes: int  # The number of episodes that belong to this segment.


def all_finite(x: Array, /) -> JaxArray:
    return jnp.all(jnp.isfinite(x))


def is_all_finite_tree(x: PyTree, /) -> JaxArray:
    return dynamic_tree_all(tree.map(all_finite, x))


def training_segments_iterable(
    segments: list[TrainingSegment],
) -> Generator[tuple[KeyArray, KeyArray]]:
    for segment in segments:
        example_keys, inference_keys = segment_keys(segment.key, segment.episodes)
        yield from zip(example_keys, inference_keys, strict=True)


def training_snapshots(
    solution: TrainingSolution,
    training_telemetries: tuple[TrainingTelemetry, ...],
    result: TrainingResult,
) -> dict[TrainingTelemetry, Any]:
    snapshots = {}
    for telemetry in training_telemetries:
        snapshot = telemetry.training_snapshot(solution, result, snapshots)
        snapshots[telemetry] = snapshot
    return snapshots


class TrainingExecutionContext(ExecutionContext[TrainingTelemetry]):
    @override
    def __init__(
        self,
        *,
        default_result: TrainingResult,
        progress_manager: rp.Progress | None,
        task_id: rp.TaskID | None,
        telemetries: Telemetries,
        wandb_run: Run | None,
        wandb_log_period: int,
        solution: TrainingSolution,
    ) -> None:
        super().__init__(
            default_result=default_result,
            progress_manager=progress_manager,
            task_id=task_id,
            telemetries=telemetries,
            wandb_run=wandb_run,
            wandb_log_period=wandb_log_period,
        )
        self.solution = solution
        self._training_telemetries = telemetries.training_telemetries

    @override
    def snapshots(self, result: Any) -> dict[TrainingTelemetry, Any]:
        return training_snapshots(self.solution, self._training_telemetries, result)


def train_one_episode(
    solution: TrainingSolution,
    batch_size: int,
    example_key: KeyArray,
    inference_key: KeyArray,
    data_source: DataSource,
    training_telemetries: tuple[TrainingTelemetry, ...],
    *,
    return_samples: bool,
) -> tuple[SolutionState, dict[TrainingTelemetry, Any], JaxArray, JaxArray]:
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
    snapshots = training_snapshots(solution, training_telemetries, result)
    finite_parameters = is_all_finite_tree(result.solution_state.dis_learnable_parameters)
    finite_inference_result = is_all_finite_tree(result.inference_result)
    return result.solution_state, snapshots, finite_parameters, finite_inference_result


j_train_one_episode = jit(train_one_episode, static_argnames=("batch_size", "return_samples"))


def train_episodes(
    solver_name: str | None,
    batch_size: int,
    solution: TrainingSolution,
    packet: ExecutionPacket,
    segments: list[TrainingSegment],
) -> TrainingResults:
    """Train episodes."""
    log.info("Training")
    total_episodes = sum(segment.episodes for segment in segments)
    solution_state = solution.solution_state
    data_source = solution.problem.create_data_source()
    default_result = TrainingResult(
        solution_state, solution.inference.infer_zero_episodes(data_source, solution.problem)
    )
    with TrainingExecutionContext.create(
        solver_name=solver_name,
        default_result=default_result,
        episodes=total_episodes,
        packet=packet,
        job_type="training",
        solution=solution,
        use_wandb=True,
    ) as execution_context:
        for example_key, inference_key in training_segments_iterable(segments):
            (solution_state, snapshots, finite_parameters, finite_inference_result) = (
                j_train_one_episode(
                    solution,
                    batch_size,
                    example_key,
                    inference_key,
                    data_source,
                    packet.telemetries.training_telemetries,
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
