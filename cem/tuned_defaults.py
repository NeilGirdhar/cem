"""Read and persist tuned hyperparameters that experiments use as defaults."""

import json
import os
import tempfile
from collections.abc import Mapping
from operator import itemgetter
from pathlib import Path
from typing import Any

TUNED_DEFAULTS_PATH = Path(__file__).resolve().parent / "tuned_defaults.json"


def load_tuned_defaults() -> dict[str, dict[str, Any]]:
    """Read and validate saved hyperparameters by experiment name.

    Return an empty mapping if the defaults file is absent.
    """
    if not TUNED_DEFAULTS_PATH.exists():
        return {}
    with TUNED_DEFAULTS_PATH.open() as f:
        data = json.load(f)
    if not isinstance(data, dict):
        msg = f"{TUNED_DEFAULTS_PATH} must contain a JSON object"
        raise TypeError(msg)
    return {
        str(k): _validate_hyperparameter_map(v, experiment_name=str(k)) for k, v in data.items()
    }


def tuned_defaults_for_experiment(experiment_name: str) -> dict[str, Any]:
    """Return one experiment's saved defaults, or an empty mapping if it has none."""
    return load_tuned_defaults().get(experiment_name, {})


def update_tuned_defaults(experiment_name: str, hyperparameters: Mapping[str, Any]) -> None:
    """Replace one experiment's saved defaults while preserving the other entries."""
    data = load_tuned_defaults()
    data[experiment_name] = {
        str(k): _json_scalar(v) for k, v in sorted(hyperparameters.items(), key=itemgetter(0))
    }
    _write_tuned_defaults(data)


def _write_tuned_defaults(data: Mapping[str, Mapping[str, Any]]) -> None:
    """Write defaults to a sibling temporary file and replace the destination atomically."""
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


def _validate_hyperparameter_map(value: object, *, experiment_name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        msg = f"{TUNED_DEFAULTS_PATH} entry for {experiment_name!r} must be a JSON object"
        raise TypeError(msg)
    return {str(k): _json_scalar(v) for k, v in value.items()}


def _json_scalar(value: object) -> int | float | str | bool | None:
    if value is None or isinstance(value, int | float | str | bool):
        return value
    msg = f"hyperparameter values must be JSON scalars, got {type(value).__name__}"
    raise TypeError(msg)
