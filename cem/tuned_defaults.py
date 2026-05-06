from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Mapping
from operator import itemgetter
from pathlib import Path
from typing import Any

TUNED_DEFAULTS_PATH = Path(__file__).resolve().parent / "demos" / "tuned_defaults.json"


def load_tuned_defaults() -> dict[str, dict[str, Any]]:
    """Load committed best-known hyperparameter defaults by demo name."""
    if not TUNED_DEFAULTS_PATH.exists():
        return {}
    with TUNED_DEFAULTS_PATH.open() as f:
        data = json.load(f)
    if not isinstance(data, dict):
        msg = f"{TUNED_DEFAULTS_PATH} must contain a JSON object"
        raise TypeError(msg)
    return {str(k): _validate_hyperparameter_map(v, demo_name=str(k)) for k, v in data.items()}


def tuned_defaults_for_demo(demo_name: str) -> dict[str, Any]:
    """Return committed best-known hyperparameter defaults for one demo."""
    return load_tuned_defaults().get(demo_name, {})


def update_tuned_defaults(demo_name: str, hyperparameters: Mapping[str, Any]) -> None:
    """Persist best-known hyperparameter defaults for one demo."""
    data = load_tuned_defaults()
    data[demo_name] = {
        str(k): _json_scalar(v) for k, v in sorted(hyperparameters.items(), key=itemgetter(0))
    }
    _write_tuned_defaults(data)


def _write_tuned_defaults(data: Mapping[str, Mapping[str, Any]]) -> None:
    TUNED_DEFAULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        prefix=f"{TUNED_DEFAULTS_PATH.name}.",
        suffix=".tmp",
        dir=TUNED_DEFAULTS_PATH.parent,
        text=True,
    )
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(data, f, indent=2, sort_keys=True)
            f.write("\n")
        Path(tmp_name).replace(TUNED_DEFAULTS_PATH)
    except BaseException:
        Path(tmp_name).unlink()
        raise


def _validate_hyperparameter_map(value: object, *, demo_name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        msg = f"{TUNED_DEFAULTS_PATH} entry for {demo_name!r} must be a JSON object"
        raise TypeError(msg)
    return {str(k): _json_scalar(v) for k, v in value.items()}


def _json_scalar(value: object) -> int | float | str | bool | None:
    if value is None or isinstance(value, int | float | str | bool):
        return value
    msg = f"hyperparameter values must be JSON scalars, got {type(value).__name__}"
    raise TypeError(msg)
