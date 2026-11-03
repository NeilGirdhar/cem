import json
from pathlib import Path

import pytest
from optuna.distributions import CategoricalDistribution

from cem import tuned_defaults
from cem.npn.experiments.supervised.missingness import supervised_missingness_experiment

_TUNED_HIDDEN_SIZE = 42
_DEFAULT_LEARNING_RATE = 0.01


def test_tuned_defaults_for_experiment_loads_committed_json(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "tuned_defaults.json"
    path.write_text(
        json.dumps(
            {
                "experiment-a": {
                    "hidden_size": 12,
                    "learning_rate": 0.25,
                }
            }
        )
    )
    monkeypatch.setattr(tuned_defaults, "TUNED_DEFAULTS_PATH", path)

    assert tuned_defaults.tuned_defaults_for_experiment("experiment-a") == {
        "hidden_size": 12,
        "learning_rate": 0.25,
    }
    assert tuned_defaults.tuned_defaults_for_experiment("experiment-b") == {}


def test_update_tuned_defaults_rewrites_one_experiment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "tuned_defaults.json"
    path.write_text(
        json.dumps(
            {
                "experiment-a": {
                    "hidden_size": 12,
                }
            }
        )
    )
    monkeypatch.setattr(tuned_defaults, "TUNED_DEFAULTS_PATH", path)

    tuned_defaults.update_tuned_defaults(
        "experiment-b",
        {
            "hidden_size": 24,
            "learning_rate": 0.125,
        },
    )

    assert json.loads(path.read_text()) == {
        "experiment-a": {
            "hidden_size": 12,
        },
        "experiment-b": {
            "hidden_size": 24,
            "learning_rate": 0.125,
        },
    }


def test_experiment_default_hyperparameters_overlay_tuned_defaults(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "tuned_defaults.json"
    path.write_text(
        json.dumps(
            {
                supervised_missingness_experiment.name: {
                    "npn.hidden_size": _TUNED_HIDDEN_SIZE,
                }
            }
        )
    )
    monkeypatch.setattr(tuned_defaults, "TUNED_DEFAULTS_PATH", path)

    defaults = supervised_missingness_experiment.default_hyperparameters()

    assert defaults["npn.hidden_size"] == _TUNED_HIDDEN_SIZE
    assert defaults["npn.learning_rate"] == _DEFAULT_LEARNING_RATE


def test_supervised_shape_hyperparameters_include_tuned_choices() -> None:
    hyperparameters = supervised_missingness_experiment.create_hyperparameters()

    mask_mlp_hidden_size = hyperparameters["mask-mlp.hidden_size"]
    npn_hidden_size = hyperparameters["npn.hidden_size"]

    assert isinstance(mask_mlp_hidden_size, CategoricalDistribution)
    assert mask_mlp_hidden_size.choices == (
        2,
        3,
        4,
        5,
        6,
        8,
        10,
        12,
        16,
        20,
        24,
        27,
        32,
        40,
        48,
        64,
        73,
        80,
        85,
        96,
        98,
        128,
        139,
        160,
        192,
        220,
        256,
    )
    assert isinstance(npn_hidden_size, CategoricalDistribution)
    assert npn_hidden_size.choices == mask_mlp_hidden_size.choices
