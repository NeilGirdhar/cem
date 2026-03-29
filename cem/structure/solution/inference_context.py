from __future__ import annotations

import logging
from functools import partial
from typing import Any

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
from .segment import segment_keys
from .telemetry import InferenceTelemetry

log = logging.getLogger(__name__)


def inference_snapshots(
    inference: Inference,
    inference_telemetries: tuple[InferenceTelemetry, ...],
    result: InferenceResult,
) -> dict[InferenceTelemetry, Any]:
    snapshots = {}
    for telemetry in inference_telemetries:
        snapshot = telemetry.inference_snapshot(inference, result, snapshots)
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
    inference_telemetries: tuple[InferenceTelemetry, ...],
    *,
    return_samples: bool,
) -> dict[InferenceTelemetry, Any]:
    result = inference.infer_one_episode(
        batch_size,
        example_key,
        inference_key,
        data_source,
        learnable_parameters,
        problem,
        return_samples=False,
    )
    return inference_snapshots(inference, inference_telemetries, result)


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
    snapshots_fn = partial(inference_snapshots, inference, packet.telemetries.inference_telemetries)
    with ExecutionContext.create(
        solver_name=solver_name,
        default_result=default_result,
        episodes=episodes,
        packet=packet,
        job_type="inference",
        use_wandb=True,
        snapshots_fn=snapshots_fn,
    ) as execution_context:
        example_keys, inference_keys = segment_keys(key, episodes)
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
                packet.telemetries.inference_telemetries,
                return_samples=False,
            )
            execution_context.append_result(snapshots)
    return InferenceResults(execution_context.episodes_done(), execution_context.telemetries())
