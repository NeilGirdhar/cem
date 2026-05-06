from __future__ import annotations

import logging
from dataclasses import replace
from functools import partial
from operator import itemgetter
from typing import Any

import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from jax import lax, tree
from tjax import Array, GenericString, JaxArray, KeyArray, PyTree, dynamic_tree_all, jit

from cem.structure.problem.data_source import DataSource
from cem.structure.solution.inference import SolutionState, TrainingResult

from .execution_context import ExecutionContext, ExecutionPacket
from .results import TrainingResults
from .telemetry import Telemetry
from .training_solution import TrainingSolution

log = logging.getLogger(__name__)
_logged_training_jit_signatures: set[tuple[int, int, tuple[str, ...]]] = set()


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


@partial(jit, static_argnames=("batch_size", "max_chunk_size"))
def train_episode_chunk(  # noqa: PLR0917
    solution: TrainingSolution,
    batch_size: int,
    example_key_base: KeyArray,
    inference_key_base: KeyArray,
    chunk_start: int,
    valid_count: int,
    max_chunk_size: int,
    data_source: DataSource,
    telemetries: tuple[Telemetry, ...],
    solution_state: SolutionState,
) -> tuple[SolutionState, dict[Telemetry, Any], JaxArray, JaxArray]:
    def step(
        carry: tuple[SolutionState, JaxArray], _: None
    ) -> tuple[tuple[SolutionState, JaxArray], tuple[dict[Telemetry, Any], JaxArray, JaxArray]]:
        carry_solution_state, episode_index = carry
        valid = episode_index < chunk_start + valid_count
        example_key = jr.fold_in(example_key_base, episode_index)
        inference_key = jr.fold_in(inference_key_base, episode_index)
        step_solution = replace(solution, solution_state=carry_solution_state)
        result = step_solution.inference.train_one_episode(
            batch_size,
            example_key,
            inference_key,
            step_solution.gradient_transformations,
            data_source,
            step_solution.problem,
            carry_solution_state,
        )
        snapshots = training_snapshots(step_solution, telemetries, result)
        finite_parameters = is_all_finite_tree(result.solution_state.dis_learnable_parameters)
        finite_inference_result = is_all_finite_tree(result.inference_result)
        new_solution_state = lax.cond(
            valid,
            lambda: result.solution_state,
            lambda: carry_solution_state,
        )
        true = jnp.ones((), dtype=bool)
        return (
            new_solution_state,
            episode_index + 1,
        ), (
            snapshots,
            jnp.where(valid, finite_parameters, true),
            jnp.where(valid, finite_inference_result, true),
        )

    (solution_state, _), (snapshots, finite_parameters, finite_inference_result) = lax.scan(
        step,
        (solution_state, jnp.asarray(chunk_start)),
        None,
        length=max_chunk_size,
    )
    return solution_state, snapshots, finite_parameters, finite_inference_result


def _slice_chunk(snapshots: dict[Telemetry, Any], count: int) -> dict[Telemetry, Any]:
    return tree.map(itemgetter(slice(count)), snapshots)


def _first_nonfinite_index(
    finite_parameters: JaxArray,
    finite_inference_result: JaxArray,
) -> int | None:
    finite = np.asarray(finite_parameters) & np.asarray(finite_inference_result)
    if np.all(finite):
        return None
    return int(np.argmax(~finite))


def _replay_to_nonfinite(
    solution: TrainingSolution,
    batch_size: int,
    example_key_base: KeyArray,
    inference_key_base: KeyArray,
    chunk_start: int,
    data_source: DataSource,
    solution_state: SolutionState,
    failure_index: int,
) -> tuple[SolutionState, TrainingResult]:
    result = None
    for episode_index in range(chunk_start, chunk_start + failure_index + 1):
        result = solution.inference.train_one_episode(
            batch_size,
            jr.fold_in(example_key_base, episode_index),
            jr.fold_in(inference_key_base, episode_index),
            solution.gradient_transformations,
            data_source,
            solution.problem,
            solution_state,
        )
        solution_state = result.solution_state
    assert result is not None
    return solution_state, result


def _log_nonfinite_result(
    result: TrainingResult,
    *,
    finite_parameters: bool,
    finite_inference_result: bool,
    episode_index: int,
) -> None:
    if not finite_parameters:
        infinite_parameters = eqx.filter(
            result.solution_state.dis_learnable_parameters, all_finite, inverse=True
        )
        log.error(
            f"Non-finite parameters encountered at example {episode_index} at these elements:"
        )
        log.info(GenericString(infinite_parameters))
    if not finite_inference_result:
        infinite_inference_result = eqx.filter(result.inference_result, all_finite, inverse=True)
        log.error(
            f"Non-finite configuration encountered at example {episode_index} at these elements:"
        )
        log.info(GenericString(infinite_inference_result))


def _log_training_jit_once(
    batch_size: int,
    max_chunk_size: int,
    telemetries: tuple[Telemetry, ...],
) -> None:
    signature = (batch_size, max_chunk_size, tuple(repr(t) for t in telemetries))
    if signature in _logged_training_jit_signatures:
        return
    _logged_training_jit_signatures.add(signature)
    log.info(
        "JIT compiling training chunk: batch_size=%d max_chunk_size=%d telemetries=%s",
        batch_size,
        max_chunk_size,
        tuple(type(t).__name__ for t in telemetries),
    )


def train_episodes(
    solver_name: str | None,
    batch_size: int,
    solution: TrainingSolution,
    packet: ExecutionPacket,
    key: KeyArray,
    episodes: int,
) -> TrainingResults:
    """Train episodes."""
    if episodes <= 0:
        msg = f"training_examples must be > 0, got {episodes}"
        raise ValueError(msg)
    if packet.scan_chunk_size <= 0:
        msg = f"scan_chunk_size must be > 0, got {packet.scan_chunk_size}"
        raise ValueError(msg)
    solution_state = solution.solution_state
    data_source = solution.problem.create_data_source()
    example_key_base, inference_key_base = jr.split(key)
    with ExecutionContext.create(
        solver_name=solver_name,
        episodes=episodes,
        packet=packet,
        job_type="training",
        use_wandb=True,
    ) as execution_context:
        for chunk_start in range(0, episodes, packet.scan_chunk_size):
            chunk_stop = min(chunk_start + packet.scan_chunk_size, episodes)
            chunk_size = chunk_stop - chunk_start
            chunk_start_state = solution_state
            _log_training_jit_once(
                batch_size, packet.scan_chunk_size, packet.telemetries.telemetries
            )
            (solution_state, snapshots, finite_parameters, finite_inference_result) = (
                train_episode_chunk(
                    solution,
                    batch_size,
                    example_key_base,
                    inference_key_base,
                    chunk_start,
                    chunk_size,
                    packet.scan_chunk_size,
                    data_source,
                    packet.telemetries.telemetries,
                    chunk_start_state,
                )
            )
            first_failure = _first_nonfinite_index(finite_parameters, finite_inference_result)
            if first_failure is None:
                execution_context.append_chunk(_slice_chunk(snapshots, chunk_size), chunk_size)
                continue
            execution_context.append_chunk(_slice_chunk(snapshots, first_failure), first_failure)
            solution_state, result = _replay_to_nonfinite(
                solution,
                batch_size,
                example_key_base,
                inference_key_base,
                chunk_start,
                data_source,
                chunk_start_state,
                first_failure,
            )
            _log_nonfinite_result(
                result,
                finite_parameters=bool(np.asarray(finite_parameters)[first_failure]),
                finite_inference_result=bool(np.asarray(finite_inference_result)[first_failure]),
                episode_index=execution_context.episodes_done(),
            )
            break
    return TrainingResults(
        execution_context.episodes_done(), execution_context.telemetries(), solution_state
    )
