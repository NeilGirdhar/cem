from __future__ import annotations

import json
from pathlib import Path

import pytest
from optuna.distributions import CategoricalDistribution

from cem import tuned_defaults
from cem.demos.afp.demo import afp_synthetic_iv_demo
from cem.demos.supervised.demo import supervised_bike_sharing_demand_demo

_TUNED_HIDDEN_SIZE = 42
_TUNED_N_FREQUENCIES = 3
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
                    "phasor.learning_rate": 0.25,
                }
            }
        )
    )
    monkeypatch.setattr(tuned_defaults, "TUNED_DEFAULTS_PATH", path)

    assert tuned_defaults.tuned_defaults_for_demo("demo-a") == {
        "hidden_size": 12,
        "phasor.learning_rate": 0.25,
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
            "phasor.n_frequencies": 2,
            "phasor.learning_rate": 0.125,
        },
    )

    assert json.loads(path.read_text()) == {
        "demo-a": {
            "hidden_size": 12,
        },
        "demo-b": {
            "phasor.learning_rate": 0.125,
            "phasor.n_frequencies": 2,
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
                    "hidden_size": _TUNED_HIDDEN_SIZE,
                    "phasor.n_frequencies": _TUNED_N_FREQUENCIES,
                }
            }
        )
    )
    monkeypatch.setattr(tuned_defaults, "TUNED_DEFAULTS_PATH", path)

    defaults = supervised_bike_sharing_demand_demo.default_hyperparameters()

    assert defaults["hidden_size"] == _TUNED_HIDDEN_SIZE
    assert defaults["phasor.n_frequencies"] == _TUNED_N_FREQUENCIES
    assert defaults["perceptron.learning_rate"] == _DEFAULT_LEARNING_RATE


def test_supervised_shape_hyperparameters_use_hardware_friendly_choices() -> None:
    hyperparameters = supervised_bike_sharing_demand_demo.create_hyperparameters()

    hidden_size = hyperparameters["hidden_size"]
    n_frequencies = hyperparameters["phasor.n_frequencies"]

    assert isinstance(hidden_size, CategoricalDistribution)
    assert hidden_size.choices == (4, 5, 6, 8, 10, 12, 16, 20, 24, 32, 40, 48, 64, 80, 96, 128)
    assert isinstance(n_frequencies, CategoricalDistribution)
    assert n_frequencies.choices == (2, 3, 4, 5, 6, 8, 10, 12, 16)


def test_afp_shape_hyperparameters_use_hardware_friendly_choices() -> None:
    hyperparameters = afp_synthetic_iv_demo.create_hyperparameters()

    endo_latent = hyperparameters["endo_latent"]
    exo_latent = hyperparameters["exo_latent"]
    n_frequencies = hyperparameters["n_frequencies"]

    assert isinstance(endo_latent, CategoricalDistribution)
    assert endo_latent.choices == (1, 2, 3, 4, 5, 6, 8, 10, 12, 16)
    assert isinstance(exo_latent, CategoricalDistribution)
    assert exo_latent.choices == (1, 2, 3, 4, 5, 6, 8, 10, 12, 16)
    assert isinstance(n_frequencies, CategoricalDistribution)
    assert n_frequencies.choices == (2, 3, 4, 5, 6, 8, 10, 12, 16)
