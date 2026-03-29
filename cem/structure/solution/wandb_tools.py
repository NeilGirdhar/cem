from collections.abc import Generator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, fields, is_dataclass
from os import PathLike
from types import NoneType
from typing import Any, Literal, TypeGuard, overload

import wandb
from matplotlib.figure import Figure
from tjax import JaxArray
from tjax.dataclasses import DataclassInstance
from wandb import Image
from wandb.sdk.wandb_run import Run


@dataclass
class WAndBInitSettings:
    entity: str | None = None
    project: str | None = None
    dir: str | PathLike[str] | None = None
    id: str | None = None
    name: str | None = None
    notes: str | None = None
    tags: Sequence[Any] | None = None
    config: dict[str, Any] | str | None = None
    config_exclude_keys: list[str] | None = None
    config_include_keys: list[str] | None = None
    allow_val_change: bool | None = None
    group: str | None = None
    job_type: str | None = None
    mode: Literal["online", "offline", "disabled"] | None = None
    force: bool | None = None
    anonymous: Literal["never", "allow", "must"] | None = None
    reinit: bool | None = None
    resume: bool | Literal["allow", "never", "must", "auto"] | None = None
    resume_from: str | None = None
    fork_from: str | None = None
    save_code: bool | None = None
    tensorboard: bool | None = None
    sync_tensorboard: bool | None = None
    monitor_gym: bool | None = None
    settings: wandb.Settings | dict[str, Any] | None = None


@contextmanager
def wandb_init(settings: WAndBInitSettings) -> Generator[Run]:
    run = wandb.init(
        job_type=settings.job_type,
        dir=settings.dir,
        config=settings.config,
        project=settings.project,
        entity=settings.entity,
        reinit=settings.reinit,
        tags=settings.tags,
        group=settings.group,
        name=settings.name,
        notes=settings.notes,
        config_exclude_keys=settings.config_exclude_keys,
        config_include_keys=settings.config_include_keys,
        anonymous=settings.anonymous,
        mode=settings.mode,
        allow_val_change=settings.allow_val_change,
        resume=settings.resume,
        force=settings.force,
        tensorboard=settings.tensorboard,
        sync_tensorboard=settings.sync_tensorboard,
        monitor_gym=settings.monitor_gym,
        save_code=settings.save_code,
        id=settings.id,
        fork_from=settings.fork_from,
        resume_from=settings.resume_from,
        settings=settings.settings,
    )
    assert isinstance(run, Run)
    yield run
    run.finish()


type _WandBLeaves = str | int | float | bool | JaxArray | Figure | Image | None
type _WandBNodes = _WandBLeaves | tuple[_WandBNodes, ...] | list[_WandBNodes] | "WandBDict"
type WandBDict = dict[str, _WandBNodes]


def as_wandb_dict(obj: DataclassInstance | dict[Any, Any]) -> WandBDict:
    """Return the fields of a dataclass instance as a new dictionary."""
    if not _is_dataclass_instance(obj) and not isinstance(obj, dict):
        raise TypeError
    return _asdict_inner(obj)


def _is_dataclass_instance(x: object) -> TypeGuard[DataclassInstance]:
    return not isinstance(x, type) and is_dataclass(x)


@overload
def _asdict_inner(obj: DataclassInstance | dict[Any, Any]) -> WandBDict: ...
@overload
def _asdict_inner(obj: object) -> _WandBNodes: ...
def _asdict_inner(obj: object) -> _WandBNodes:
    if _is_dataclass_instance(obj):
        return {f.name: _asdict_inner(getattr(obj, f.name)) for f in fields(obj)}
    if isinstance(obj, dict):
        return {str(k): _asdict_inner(v) for k, v in obj.items()}
    if isinstance(obj, tuple):
        return tuple(_asdict_inner(v) for v in obj)
    if isinstance(obj, list):
        return [_asdict_inner(v) for v in obj]
    if isinstance(obj, str | int | float | JaxArray | NoneType):
        return obj
    msg = f"Cannot convert object of type {type(obj).__name__}"
    raise TypeError(msg)
