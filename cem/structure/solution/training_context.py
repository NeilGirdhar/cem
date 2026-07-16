import logging
from dataclasses import replace
from functools import partial
from typing import Any

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jax import lax, tree
from tjax import Array, GenericString, JaxArray, KeyArray, PyTree, dynamic_tree_all, jit

from cem.structure.problem.data_source import DataSource
from cem.structure.solution.inference import SolutionState, TrainingResult

from .execution_loop import (
    collect_snapshots,
    fold_episode_keys,
    log_jit_once,
    run_execution_chunks,
)
from .execution_packet import ExecutionPacket
from .results import TrainingResults
from .telemetry import Telemetry
from .training_solution import TrainingSolution

log = logging.getLogger(__name__)
_logged_training_jit_signatures: set[tuple[int, int, tuple[str, ...]]] = set()


def all_finite(x: Array, /) -> JaxArray:
    return jnp.all(jnp.isfinite(x))


def is_all_finite_tree(x: PyTree, /) -> JaxArray:
    return dynamic_tree_all(tree.map(all_finite, x))


@partial(jit, static_argnames=("batch_size", "max_chunk_size"))
def train_episode_chunk(  # ruff:ignore[too-many-positional-arguments]
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
        example_key, inference_key = fold_episode_keys(
            example_key_base, inference_key_base, episode_index
        )
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
        snapshots = collect_snapshots(
            telemetries,
            lambda telemetry, snapshots: telemetry.training_snapshot(
                step_solution, result, snapshots
            ),
        )
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
        example_key, inference_key = fold_episode_keys(
            example_key_base, inference_key_base, episode_index
        )
        result = solution.inference.train_one_episode(
            batch_size,
            example_key,
            inference_key,
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
    solution_state = solution.solution_state
    data_source = solution.problem.create_data_source()

    def handle_chunk(
        example_key_base: KeyArray,
        inference_key_base: KeyArray,
        chunk_start: int,
        chunk_size: int,
        episodes_done: int,
    ) -> tuple[dict[Telemetry, Any], int, bool]:
        nonlocal solution_state
        chunk_start_state = solution_state
        log_jit_once(
            _logged_training_jit_signatures,
            log,
            "training",
            batch_size,
            packet.scan_chunk_size,
            packet.telemetries.telemetries,
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
            return snapshots, chunk_size, True
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
            episode_index=episodes_done + first_failure,
        )
        return snapshots, first_failure, False

    execution_context = run_execution_chunks(
        solver_name=solver_name,
        episodes=episodes,
        packet=packet,
        key=key,
        job_type="training",
        handle_chunk=handle_chunk,
    )
    return TrainingResults(
        execution_context.episodes_done(), execution_context.telemetries(), solution_state
    )
