import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

import cem.commands.sip_identification as command
from cem.sip import (
    CausalBenchmarkResult,
    CausalBenchmarkTrajectory,
    CreditBenchmarkResult,
)


def test_sip_identification_cli_writes_json(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    zero = CausalBenchmarkResult(
        1.7,
        1.6,
        0.01,
        0.02,
        trajectory=CausalBenchmarkTrajectory((0, 32), (0.0, 1.6), (1.0, 0.02)),
    )
    random = CausalBenchmarkResult(
        1.7,
        1.7,
        0.001,
        0.01,
        trajectory=CausalBenchmarkTrajectory((0, 32), (0.0, 1.7), (1.0, 0.01)),
    )
    inherited = CausalBenchmarkResult(
        1.7,
        1.65,
        0.02,
        0.03,
        true_first_stage_effect=1.3,
        estimated_first_stage_effect=1.2,
        first_stage_residual_covariance=0.001,
        trajectory=CausalBenchmarkTrajectory(
            (0, 32),
            (0.0, 1.65),
            (1.0, 0.03),
            (0.1, 0.2),
        ),
    )
    policy = CausalBenchmarkResult(
        1.7,
        1.9,
        0.0,
        0.03,
        trajectory=CausalBenchmarkTrajectory(
            (0, 32),
            (0.0, 1.9),
            (1.0, 0.03),
            (0.0, 0.2),
        ),
    )
    monkeypatch.setattr(
        command,
        "run_direct_injection_benchmark",
        lambda **_kwargs: {"zero": zero, "random": random},
    )
    monkeypatch.setattr(
        command,
        "run_inherited_instrument_benchmark",
        lambda **_kwargs: {"policy": policy, "injected": inherited},
    )
    ordinary_credit = CreditBenchmarkResult(
        1.7,
        0.4,
        0.05,
        trajectory=CausalBenchmarkTrajectory((0, 32), (1.8, 0.4), (1.0, 0.05)),
    )
    td_credit = CreditBenchmarkResult(
        1.7,
        1.67,
        0.1,
        trajectory=CausalBenchmarkTrajectory(
            (0, 64, 128, 192, 256),
            (1.67,) * 5,
            (0.1,) * 5,
            link_strengths=(0.0,) * 5,
            expected_td_errors=(1.0,) * 5,
            td_error_by_step=((1.0, 0.9),) * 5,
            td_error_mean_by_step=((-0.1, 0.1),) * 5,
        ),
    )
    monkeypatch.setattr(
        command,
        "run_td_error_benchmark",
        lambda **_kwargs: {"ordinary": ordinary_credit, "td": td_credit},
    )
    output = tmp_path / "sip-identification.json"

    result = CliRunner().invoke(
        command.app,
        ["--output", str(output), "--count", "8", "--steps", "4"],
    )

    assert result.exit_code == 0
    payload = json.loads(output.read_text())
    balance = payload.pop("td-error-balance")
    trajectory_count = 8
    trajectory_steps = 16
    assert balance["iteration"] == list(range(trajectory_steps + 1))
    assert len(balance["line plots"]) == trajectory_count
    assert all(len(balance[name]) == trajectory_steps + 1 for name in balance["line plots"])
    assert all(balance[name][0] == pytest.approx(1.0) for name in balance["line plots"])
    assert payload == {
        "configuration": {
            "count": 8,
            "steps": 4,
            "direct_seed": 200,
            "inherited_seed": 300,
            "td_error_seed": 400,
        },
        "direct-injection": {
            "zero": {
                "true_effect": 1.7,
                "estimated_effect": 1.6,
                "residual_instrument_covariance": 0.01,
                "reconstruction_loss": 0.02,
                "effect_error": pytest.approx(0.1),
            },
            "random": {
                "true_effect": 1.7,
                "estimated_effect": 1.7,
                "residual_instrument_covariance": 0.001,
                "reconstruction_loss": 0.01,
                "effect_error": 0.0,
            },
        },
        "direct-injection-effect": {
            "iteration": [0, 32],
            "line plots": {
                "zero": "Noise magnitude 0",
                "random": "Noise magnitude 1",
                "true": "True effect",
            },
            "zero": [0.0, 1.6],
            "random": [0.0, 1.7],
            "true": [1.7, 1.7],
        },
        "direct-injection-reconstruction-loss": {
            "iteration": [0, 32],
            "line plots": {
                "zero": "Noise magnitude 0",
                "random": "Noise magnitude 1",
            },
            "zero": [1.0, 0.02],
            "random": [1.0, 0.01],
        },
        "inherited-instrument-effect": {
            "iteration": [0, 32],
            "line plots": {
                "policy": "Policy",
                "injected": "Injected",
                "true": "True effect",
                "instrument-y": "instrument(Y) magnitude",
            },
            "line styles": {"instrument-y": "dashed"},
            "line colors": {
                "policy": "dark-peach",
                "injected": "dark-blue",
                "true": "dark-green",
                "instrument-y": "dark-blue",
            },
            "policy": [0.0, 1.9],
            "injected": [0.0, 1.65],
            "true": [1.7, 1.7],
            "instrument-y": [0.1, 0.2],
        },
        "inherited-instrument-reconstruction-loss": {
            "iteration": [0, 32],
            "line plots": {
                "policy": "Policy Z reconstruction",
                "injected": "Injected Z reconstruction",
            },
            "line styles": {},
            "line colors": {
                "policy": "dark-peach",
                "injected": "dark-blue",
            },
            "policy": [1.0, 0.03],
            "injected": [1.0, 0.03],
        },
        "instrument-inheritance": {
            "policy": {
                "true_effect": 1.7,
                "estimated_effect": 1.9,
                "residual_instrument_covariance": 0.0,
                "reconstruction_loss": 0.03,
                "effect_error": pytest.approx(0.2),
            },
            "injected": {
                "true_effect": 1.7,
                "estimated_effect": 1.65,
                "residual_instrument_covariance": 0.02,
                "reconstruction_loss": 0.03,
                "true_first_stage_effect": 1.3,
                "estimated_first_stage_effect": 1.2,
                "first_stage_residual_covariance": 0.001,
                "effect_error": pytest.approx(0.05),
            },
        },
        "td-error": {
            "ordinary": {
                "true_effect": 1.7,
                "estimated_effect": 0.4,
                "reconstruction_loss": 0.05,
                "effect_error": pytest.approx(1.3),
            },
            "td": {
                "true_effect": 1.7,
                "estimated_effect": 1.67,
                "reconstruction_loss": 0.1,
                "effect_error": pytest.approx(0.03),
            },
        },
        "td-error-link-strength": {
            "iteration": [0, 100, 200, 300, 400, 500, 600],
            "line plots": {
                "td": "P to R link strength",
            },
            "td": [0.0] * 7,
        },
        "td-error-expected-error": {
            "iteration": [0, 100, 200, 300, 400, 500, 600],
            "line plots": {
                "td": "Expected TD-error magnitude",
            },
            "td": [1.0] * 7,
        },
        "td-error-by-step": {
            "iteration": [1, 2],
            "line plots": {
                "0": "0 episodes",
                "100": "100 episodes",
                "200": "200 episodes",
                "300": "300 episodes",
                "400": "400 episodes",
                "500": "500 episodes",
                "600": "600 episodes",
            },
            **{str(example): [1.0, 0.9] for example in (0, 100, 200, 300, 400, 500, 600)},
        },
        "td-error-mean-by-step": {
            "iteration": [1, 2],
            "line plots": {
                str(example): f"{example} episodes" for example in (0, 100, 200, 300, 400, 500, 600)
            },
            **{str(example): [-0.1, 0.1] for example in (0, 100, 200, 300, 400, 500, 600)},
        },
    }
