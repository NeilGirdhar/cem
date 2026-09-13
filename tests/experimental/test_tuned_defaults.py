import json
from pathlib import Path

import pytest
from optuna.distributions import CategoricalDistribution

from cem import tuned_defaults
from cem.demos.supervised.demo import supervised_bike_sharing_demand_demo

_TUNED_HIDDEN_SIZE = 42
_DEFAULT_LEARNING_RATE = 0.01


def test_tuned_defaults_for_demo_loads_committed_json(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "tuned_defaults.json"
    path.write_text(
        json.dumps(
            {
                "demo-a": {
                    "hidden_size": 12,
                    "learning_rate": 0.25,
                }
            }
        )
    )
    monkeypatch.setattr(tuned_defaults, "TUNED_DEFAULTS_PATH", path)

    assert tuned_defaults.tuned_defaults_for_demo("demo-a") == {
        "hidden_size": 12,
        "learning_rate": 0.25,
    }
    assert tuned_defaults.tuned_defaults_for_demo("demo-b") == {}


def test_update_tuned_defaults_rewrites_one_demo(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "tuned_defaults.json"
    path.write_text(
        json.dumps(
            {
                "demo-a": {
                    "hidden_size": 12,
                }
            }
        )
    )
    monkeypatch.setattr(tuned_defaults, "TUNED_DEFAULTS_PATH", path)

    tuned_defaults.update_tuned_defaults(
        "demo-b",
        {
            "hidden_size": 24,
            "learning_rate": 0.125,
        },
    )

    assert json.loads(path.read_text()) == {
        "demo-a": {
            "hidden_size": 12,
        },
        "demo-b": {
            "hidden_size": 24,
            "learning_rate": 0.125,
        },
    }


def test_demo_default_hyperparameters_overlay_tuned_defaults(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "tuned_defaults.json"
    path.write_text(
        json.dumps(
            {
                supervised_bike_sharing_demand_demo.name: {
                    "natural_parameter.hidden_size": _TUNED_HIDDEN_SIZE,
                }
            }
        )
    )
    monkeypatch.setattr(tuned_defaults, "TUNED_DEFAULTS_PATH", path)

    defaults = supervised_bike_sharing_demand_demo.default_hyperparameters()

    assert defaults["natural_parameter.hidden_size"] == _TUNED_HIDDEN_SIZE
    assert defaults["natural_parameter.learning_rate"] == _DEFAULT_LEARNING_RATE


def test_supervised_shape_hyperparameters_include_tuned_choices() -> None:
    hyperparameters = supervised_bike_sharing_demand_demo.create_hyperparameters()

    perceptron_hidden_size = hyperparameters["perceptron.hidden_size"]
    natural_parameter_hidden_size = hyperparameters["natural_parameter.hidden_size"]

    assert isinstance(perceptron_hidden_size, CategoricalDistribution)
    assert perceptron_hidden_size.choices == (
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
    assert isinstance(natural_parameter_hidden_size, CategoricalDistribution)
    assert natural_parameter_hidden_size.choices == perceptron_hidden_size.choices
