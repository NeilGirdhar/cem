"""Run held-out inference in compiled chunks and collect episode telemetry."""

import logging
from functools import partial
from typing import Any

import jax.numpy as jnp
from jax import lax
from tjax import JaxArray, KeyArray, jit

from cem.structure.graph.model import Model
from cem.structure.problem.data_source import DataSource
from cem.structure.problem.problem import Problem
from cem.structure.solution.inference import Inference, SolutionState

from .execution_loop import (
    collect_snapshots,
    fold_episode_keys,
    log_jit_once,
    run_execution_chunks,
)
from .execution_packet import ExecutionPacket
from .results import InferenceResults
from .telemetry import Telemetry

log = logging.getLogger(__name__)
_logged_inference_jit_signatures: set[tuple[int, int, tuple[str, ...]]] = set()


@partial(jit, static_argnames=("batch_size", "max_chunk_size"))
def infer_episode_chunk(  # ruff:ignore[too-many-arguments, too-many-positional-arguments]
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
    """Evaluate a fixed-size chunk and stack its snapshots along the episode axis."""

    def step(episode_index: JaxArray, _: None) -> tuple[JaxArray, dict[Telemetry, Any]]:
        example_key, inference_key = fold_episode_keys(
            example_key_base, inference_key_base, episode_index
        )
        result = inference.infer_one_episode(
            batch_size,
            example_key,
            inference_key,
            data_source,
            learnable_parameters,
            problem,
        )
        return episode_index + 1, collect_snapshots(
            telemetries,
            lambda telemetry, snapshots: telemetry.inference_snapshot(inference, result, snapshots),
        )

    _, snapshots = lax.scan(step, jnp.asarray(chunk_start), None, length=max_chunk_size)
    return snapshots


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
    """Evaluate trained parameters on the inference split and collect requested telemetry.

    The final compiled chunk may evaluate extra episodes to keep the scan size fixed.
    Only snapshots for the requested episodes enter the result.
    """
    if episodes < 0:
        msg = f"inference_examples must be >= 0, got {episodes}"
        raise ValueError(msg)
    if episodes == 0:
        return InferenceResults(count=0, telemetries={})
    data_source = problem.create_data_source(inference=True)

    def handle_chunk(
        example_key_base: KeyArray,
        inference_key_base: KeyArray,
        chunk_start: int,
        chunk_size: int,
        episodes_done: int,
    ) -> tuple[dict[Telemetry, Any], int, bool]:
        del episodes_done
        learnable_parameters = solution_state.dis_learnable_parameters.assembled()
        log_jit_once(
            _logged_inference_jit_signatures,
            log,
            "inference",
            batch_size,
            packet.scan_chunk_size,
            packet.telemetries.telemetries,
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
        return snapshots, chunk_size, True

    execution_context = run_execution_chunks(
        solver_name=solver_name,
        episodes=episodes,
        packet=packet,
        key=key,
        job_type="inference",
        handle_chunk=handle_chunk,
    )
    return InferenceResults(execution_context.episodes_done(), execution_context.telemetries())
