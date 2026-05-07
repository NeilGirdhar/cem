from __future__ import annotations

import logging
from collections.abc import Callable
from operator import itemgetter
from typing import Any

import jax.random as jr
from jax import tree
from tjax import JaxArray, KeyArray

from .execution_context import ExecutionContext
from .execution_packet import ExecutionPacket
from .telemetry import Telemetry

type ChunkHandler = Callable[
    [KeyArray, KeyArray, int, int, int], tuple[dict[Telemetry, Any], int, bool]
]


def collect_snapshots(
    telemetries: tuple[Telemetry, ...],
    snapshot: Callable[[Telemetry, dict[Telemetry, Any]], Any],
) -> dict[Telemetry, Any]:
    snapshots: dict[Telemetry, Any] = {}
    for telemetry in telemetries:
        value = snapshot(telemetry, snapshots)
        if value is not None:
            snapshots[telemetry] = value
    return snapshots


def fold_episode_keys(
    example_key_base: KeyArray,
    inference_key_base: KeyArray,
    episode_index: int | JaxArray,
) -> tuple[KeyArray, KeyArray]:
    return jr.fold_in(example_key_base, episode_index), jr.fold_in(
        inference_key_base, episode_index
    )


def slice_chunk(snapshots: dict[Telemetry, Any], count: int) -> dict[Telemetry, Any]:
    return tree.map(itemgetter(slice(count)), snapshots)


def log_jit_once(
    logged_signatures: set[tuple[int, int, tuple[str, ...]]],
    logger: logging.Logger,
    job_type: str,
    batch_size: int,
    max_chunk_size: int,
    telemetries: tuple[Telemetry, ...],
) -> None:
    signature = (batch_size, max_chunk_size, tuple(repr(t) for t in telemetries))
    if signature in logged_signatures:
        return
    logged_signatures.add(signature)
    logger.info(
        "JIT compiling %s chunk: batch_size=%d max_chunk_size=%d telemetries=%s",
        job_type,
        batch_size,
        max_chunk_size,
        tuple(type(t).__name__ for t in telemetries),
    )


def _validate_scan_chunk_size(scan_chunk_size: int) -> None:
    if scan_chunk_size <= 0:
        msg = f"scan_chunk_size must be > 0, got {scan_chunk_size}"
        raise ValueError(msg)


def run_execution_chunks(
    *,
    solver_name: str | None,
    episodes: int,
    packet: ExecutionPacket,
    key: KeyArray,
    job_type: str,
    handle_chunk: ChunkHandler,
) -> ExecutionContext:
    _validate_scan_chunk_size(packet.scan_chunk_size)
    example_key_base, inference_key_base = jr.split(key)
    with ExecutionContext.create(
        solver_name=solver_name,
        episodes=episodes,
        packet=packet,
        job_type=job_type,
        use_wandb=True,
    ) as execution_context:
        for chunk_start in range(0, episodes, packet.scan_chunk_size):
            chunk_size = min(packet.scan_chunk_size, episodes - chunk_start)
            snapshots, count, keep_going = handle_chunk(
                example_key_base,
                inference_key_base,
                chunk_start,
                chunk_size,
                execution_context.episodes_done(),
            )
            execution_context.append_chunk(slice_chunk(snapshots, count), count)
            if not keep_going:
                break
    return execution_context
