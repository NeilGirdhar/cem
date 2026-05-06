from __future__ import annotations

import logging
from functools import partial
from operator import itemgetter
from typing import Any

import jax.numpy as jnp
import jax.random as jr
from jax import lax, tree
from tjax import JaxArray, KeyArray, jit

from cem.structure.graph.model import Model
from cem.structure.problem.data_source import DataSource
from cem.structure.problem.problem import Problem
from cem.structure.solution.inference import Inference, InferenceResult, SolutionState

from .execution_context import ExecutionContext, ExecutionPacket
from .results import InferenceResults
from .telemetry import Telemetry

log = logging.getLogger(__name__)
_logged_inference_jit_signatures: set[tuple[int, int, tuple[str, ...]]] = set()


def inference_snapshots(
    inference: Inference,
    telemetries: tuple[Telemetry, ...],
    result: InferenceResult,
) -> dict[Telemetry, Any]:
    snapshots: dict[Telemetry, Any] = {}
    for telemetry in telemetries:
        snapshot = telemetry.inference_snapshot(inference, result, snapshots)
        if snapshot is not None:
            snapshots[telemetry] = snapshot
    return snapshots


@partial(jit, static_argnames=("batch_size", "max_chunk_size"))
def infer_episode_chunk(  # noqa: PLR0917
    inference: Inference,
    batch_size: int,
    example_key_base: KeyArray,
    inference_key_base: KeyArray,
    chunk_start: int,
    max_chunk_size: int,
    data_source: DataSource,
    learnable_parameters: Model,
    problem: Problem,
    telemetries: tuple[Telemetry, ...],
) -> dict[Telemetry, Any]:
    def step(episode_index: JaxArray, _: None) -> tuple[JaxArray, dict[Telemetry, Any]]:
        example_key = jr.fold_in(example_key_base, episode_index)
        inference_key = jr.fold_in(inference_key_base, episode_index)
        result = inference.infer_one_episode(
            batch_size,
            example_key,
            inference_key,
            data_source,
            learnable_parameters,
            problem,
        )
        return episode_index + 1, inference_snapshots(inference, telemetries, result)

    _, snapshots = lax.scan(step, jnp.asarray(chunk_start), None, length=max_chunk_size)
    return snapshots


def _slice_chunk(snapshots: dict[Telemetry, Any], count: int) -> dict[Telemetry, Any]:
    return tree.map(itemgetter(slice(count)), snapshots)


def _log_inference_jit_once(
    batch_size: int,
    max_chunk_size: int,
    telemetries: tuple[Telemetry, ...],
) -> None:
    signature = (batch_size, max_chunk_size, tuple(repr(t) for t in telemetries))
    if signature in _logged_inference_jit_signatures:
        return
    _logged_inference_jit_signatures.add(signature)
    log.info(
        "JIT compiling inference chunk: batch_size=%d max_chunk_size=%d telemetries=%s",
        batch_size,
        max_chunk_size,
        tuple(type(t).__name__ for t in telemetries),
    )


def infer_episodes(
    solver_name: str | None,
    batch_size: int,
    episodes: int,
    key: KeyArray,
    inference: Inference,
    problem: Problem,
    packet: ExecutionPacket,
    solution_state: SolutionState,
) -> InferenceResults:
    """Infer episodes.

    Args:
        solver_name: The name of the solver.
        batch_size: The batch size.
        key: The random number generation key used for inference.
        episodes: The number of RL episodes to run.
        inference: The object that runs the inference.
        problem: The problem being inferred.
        packet: The inference packet.
        solution_state: The inference parameters.
    """
    if episodes < 0:
        msg = f"inference_examples must be >= 0, got {episodes}"
        raise ValueError(msg)
    if episodes == 0:
        return InferenceResults(count=0, telemetries={})
    if packet.scan_chunk_size <= 0:
        msg = f"scan_chunk_size must be > 0, got {packet.scan_chunk_size}"
        raise ValueError(msg)
    data_source = problem.create_data_source()
    with ExecutionContext.create(
        solver_name=solver_name,
        episodes=episodes,
        packet=packet,
        job_type="inference",
        use_wandb=True,
    ) as execution_context:
        example_key_base, inference_key_base = jr.split(key)
        for chunk_start in range(0, episodes, packet.scan_chunk_size):
            chunk_stop = min(chunk_start + packet.scan_chunk_size, episodes)
            chunk_size = chunk_stop - chunk_start
            learnable_parameters = solution_state.dis_learnable_parameters.assembled()
            _log_inference_jit_once(
                batch_size, packet.scan_chunk_size, packet.telemetries.telemetries
            )
            snapshots = infer_episode_chunk(
                inference,
                batch_size,
                example_key_base,
                inference_key_base,
                chunk_start,
                packet.scan_chunk_size,
                data_source,
                learnable_parameters,
                problem,
                packet.telemetries.telemetries,
            )
            execution_context.append_chunk(_slice_chunk(snapshots, chunk_size), chunk_size)
    return InferenceResults(execution_context.episodes_done(), execution_context.telemetries())
