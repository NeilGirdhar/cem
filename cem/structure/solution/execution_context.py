from __future__ import annotations

import logging
import operator
import tempfile
from collections.abc import Generator
from contextlib import ExitStack, contextmanager
from dataclasses import fields, is_dataclass, replace
from typing import Any, Self, override

import jax.numpy as jnp
import numpy as np
import rich.progress as rp
from jax import tree
from jax.profiler import trace
from tjax import JaxArray, Timer, display_time
from tjax.dataclasses import DataclassInstance
from wandb.sdk.wandb_run import Run

from .wandb_tools import WandBDict, wandb_init

from .execution_packet import ExecutionPacket
from .telemetry import Telemetries, Telemetry

log = logging.getLogger(__name__)


def _stack(*elements: JaxArray) -> JaxArray:
    return jnp.asarray(np.stack(elements))  # np.stack is much faster than jnp.stack


class ExecutionContext[T: Telemetry]:
    @override
    def __init__(
        self,
        *,
        default_result: Any,
        progress_manager: rp.Progress | None,
        task_id: rp.TaskID | None,
        telemetries: Telemetries,  # Only part of the interface for child classes.
        wandb_run: Run | None,
        wandb_log_period: int,
    ) -> None:
        super().__init__()
        self._default_result = default_result
        self._progress_manager = progress_manager
        self._task_id = task_id
        self._results: list[dict[T, Any]] = []
        self._wandb_run = wandb_run
        self._telemetries: dict[T, Any] = {}
        self._wandb_log_period = wandb_log_period

    def snapshots(self, result: object) -> dict[T, Any]:
        raise NotImplementedError

    def append_result(self, snapshots: dict[T, Any]) -> None:
        self._results.append(snapshots)
        if self._progress_manager is not None:
            assert self._task_id is not None
            self._progress_manager.advance(self._task_id, 1)
        if (
            self._wandb_run is not None
            and snapshots
            and self.episodes_done() % self._wandb_log_period == 0
        ):

            def dc_to_dict(x: DataclassInstance | dict[T, Any], /) -> WandBDict:
                """Transform dataclasses into dicts."""
                if is_dataclass(x):
                    return {f.name: dc_to_dict(getattr(x, f.name)) for f in fields(x)}
                assert not isinstance(x, DataclassInstance)
                return {str(k): dc_to_dict(v) for k, v in x.items()}

            self._wandb_run.log(dc_to_dict(snapshots))

    def _stack_telemetries(self) -> None:
        results = self._results
        if results:
            result, *more_results = results
            self._telemetries = tree.map(_stack, result, *more_results)
        else:
            assert not self._telemetries
            self._telemetries = self._empty_telemetries()

    def episodes_done(self) -> int:
        return len(self._results)

    def telemetries(self) -> dict[T, Any]:
        return self._telemetries

    def _empty_telemetries(self) -> dict[T, Any]:
        return tree.map(operator.itemgetter(jnp.newaxis), self.snapshots(self._default_result))

    @classmethod
    @contextmanager
    def create(
        cls,
        *,
        solver_name: str | None,
        default_result: object,
        episodes: int,
        packet: ExecutionPacket,
        job_type: str,
        use_wandb: bool,
        **kwargs: object,
    ) -> Generator[Self]:
        exit_stack = ExitStack()
        task_id: rp.TaskID | None = None
        if packet.progress_manager is not None:
            task_title = (
                job_type.title() if solver_name is None else f"{solver_name} {job_type.title()}"
            )
            task_id = packet.progress_manager.add_task(task_title, total=episodes)
            exit_stack.enter_context(packet.progress_manager)
        if packet.wandb_settings is not None and use_wandb:
            wandb_settings = replace(packet.wandb_settings, job_type=job_type, group=solver_name)
            wandb_run = exit_stack.enter_context(wandb_init(wandb_settings))
        else:
            wandb_run = None
        if packet.enable_profiling:
            temp_log_dir = tempfile.TemporaryDirectory(prefix="jax")
            exit_stack.enter_context(trace(temp_log_dir.name, create_perfetto_link=True))
        inference_manager = cls(
            default_result=default_result,
            progress_manager=packet.progress_manager,
            task_id=task_id,
            telemetries=packet.telemetries,
            wandb_run=wandb_run,
            wandb_log_period=packet.wandb_log_period,
            **kwargs,
        )
        with exit_stack:
            yield inference_manager

        with Timer("Stacking"):
            inference_manager._stack_telemetries()

        if packet.progress_manager is not None:
            assert task_id is not None
            task = packet.progress_manager._tasks[task_id]  # noqa: SLF001
            if task.elapsed is not None and episodes != 0:
                log.info(f"Average iteration period {display_time(task.elapsed * 1e6 / episodes)}")
            packet.progress_manager.remove_task(task_id)
