from __future__ import annotations

import logging
from functools import partial
from typing import Any

import jax.random as jr
from tjax import KeyArray, jit

from cem.structure.model import (
    DataSource,
    Inference,
    InferenceResult,
    Model,
    Problem,
    SolutionState,
)

from .execution_context import ExecutionContext, ExecutionPacket
from .results import InferenceResults
from .telemetry import Telemetry

log = logging.getLogger(__name__)


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


@partial(jit, static_argnames=("batch_size", "return_samples"))
def infer_one_episode(
    inference: Inference,
    batch_size: int,
    example_key: KeyArray,
    inference_key: KeyArray,
    data_source: DataSource,
    learnable_parameters: Model,
    problem: Problem,
    telemetries: tuple[Telemetry, ...],
    *,
    return_samples: bool,
) -> dict[Telemetry, Any]:
    result = inference.infer_one_episode(
        batch_size,
        example_key,
        inference_key,
        data_source,
        learnable_parameters,
        problem,
        return_samples=False,
    )
    return inference_snapshots(inference, telemetries, result)


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
    log.info("Inferring")
    data_source = problem.create_data_source()
    default_result = inference.infer_zero_episodes(data_source, problem)
    default_snapshots = inference_snapshots(
        inference, packet.telemetries.telemetries, default_result
    )
    with ExecutionContext.create(
        solver_name=solver_name,
        default_snapshots=default_snapshots,
        episodes=episodes,
        packet=packet,
        job_type="inference",
        use_wandb=True,
    ) as execution_context:
        example_key_base, inference_key_base = jr.split(key)
        example_keys = jr.split(example_key_base, episodes)
        inference_keys = jr.split(inference_key_base, episodes)
        for example_key, inference_key in zip(example_keys, inference_keys, strict=True):
            learnable_parameters = solution_state.dis_learnable_parameters.assembled()
            snapshots = infer_one_episode(
                inference,
                batch_size,
                example_key,
                inference_key,
                data_source,
                learnable_parameters,
                problem,
                packet.telemetries.telemetries,
                return_samples=False,
            )
            execution_context.append_result(snapshots)
    return InferenceResults(execution_context.episodes_done(), execution_context.telemetries())
