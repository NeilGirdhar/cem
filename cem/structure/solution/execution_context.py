import tempfile
from collections.abc import Generator
from contextlib import ExitStack, contextmanager
from dataclasses import fields, is_dataclass, replace
from pathlib import Path
from typing import Any, Self

import jax.numpy as jnp
import rich.progress as rp
from jax import tree
from jax.profiler import trace
from tjax.dataclasses import DataclassInstance
from wandb.sdk.wandb_run import Run

from .execution_packet import ExecutionPacket
from .telemetry import Telemetry
from .wandb_tools import WandBDict, wandb_init

_JOB_TYPE_TITLES = {
    "inference": "Inferring",
    "training": "Training",
}


def _task_title(job_type: str, solver_name: str | None) -> str:
    title = _JOB_TYPE_TITLES.get(job_type, job_type.title())
    if solver_name is None:
        return title
    return f"{title} {solver_name}"


def _snapshots_to_wandb_dict(x: DataclassInstance | dict[Any, Any], /) -> WandBDict:
    """Recursively convert a snapshots structure into a W&B-compatible dict.

    Dataclass instances are converted to dicts keyed by field name.  Plain dicts
    have their keys stringified (e.g. Telemetry objects become their repr).
    Recursion stops at any value that is neither a dataclass nor a dict.

    Args:
        x: A dataclass instance or a dict whose values may themselves be dataclasses
            or nested dicts.

    Returns:
        A nested ``WandBDict`` suitable for passing to ``wandb.Run.log``.
    """
    if is_dataclass(x):
        return {f.name: _snapshots_to_wandb_dict(getattr(x, f.name)) for f in fields(x)}
    assert not isinstance(x, DataclassInstance)
    return {str(k): _snapshots_to_wandb_dict(v) for k, v in x.items()}


class ExecutionContext[T: Telemetry]:
    def __init__(
        self,
        *,
        progress_manager: rp.Progress | None,
        task_id: rp.TaskID | None,
        wandb_run: Run | None,
    ) -> None:
        super().__init__()
        self._progress_manager = progress_manager
        self._task_id = task_id
        self._chunks: list[dict[T, Any]] = []
        self._episodes_done = 0
        self._wandb_run = wandb_run
        self._telemetries: dict[T, Any] = {}

    def append_chunk(self, snapshots: dict[T, Any], count: int) -> None:
        if count == 0:
            return
        self._chunks.append(snapshots)
        self._episodes_done += count
        if self._progress_manager is not None:
            assert self._task_id is not None
            self._progress_manager.advance(self._task_id, count)
        if self._wandb_run is not None and snapshots:
            self._wandb_run.log(_snapshots_to_wandb_dict(snapshots))

    def _stack_telemetries(self) -> None:
        if not self._chunks:
            self._telemetries = {}
            return
        result, *more_results = self._chunks
        if not more_results:
            self._telemetries = result
            return
        self._telemetries = tree.map(lambda *xs: jnp.concatenate(xs, axis=0), result, *more_results)

    def episodes_done(self) -> int:
        return self._episodes_done

    def telemetries(self) -> dict[T, Any]:
        return self._telemetries

    @classmethod
    @contextmanager
    def create(
        cls,
        *,
        solver_name: str | None,
        episodes: int,
        packet: ExecutionPacket,
        job_type: str,
        use_wandb: bool,
    ) -> Generator[Self]:
        exit_stack = ExitStack()
        task_id: rp.TaskID | None = None
        if packet.progress_manager is not None:
            task_id = packet.progress_manager.add_task(
                _task_title(job_type, solver_name), total=episodes
            )
            if not packet.progress_manager.live.is_started:
                exit_stack.enter_context(packet.progress_manager)
        if packet.wandb_settings is not None and use_wandb:
            wandb_settings = replace(packet.wandb_settings, job_type=job_type, group=solver_name)
            assert isinstance(wandb_settings.dir, Path)
            wandb_settings.dir.mkdir(parents=True, exist_ok=True)
            wandb_run = exit_stack.enter_context(wandb_init(wandb_settings))
        else:
            wandb_run = None
        if packet.enable_profiling:
            temp_log_dir = tempfile.TemporaryDirectory(prefix="jax")
            exit_stack.enter_context(trace(temp_log_dir.name, create_perfetto_link=True))
        inference_manager = cls(
            progress_manager=packet.progress_manager,
            task_id=task_id,
            wandb_run=wandb_run,
        )
        with exit_stack:
            yield inference_manager

        inference_manager._stack_telemetries()

        if packet.progress_manager is not None:
            assert task_id is not None
            packet.progress_manager.remove_task(task_id)
