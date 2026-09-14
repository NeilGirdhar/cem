import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

import cem.commands.sip_identification as command
from cem.sip import CausalBenchmarkResult


def test_sip_identification_cli_writes_json(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    direct = CausalBenchmarkResult(1.7, 1.6, 0.01, 0.02)
    inherited = CausalBenchmarkResult(
        1.7,
        1.65,
        0.02,
        0.03,
        true_first_stage_effect=1.3,
        estimated_first_stage_effect=1.2,
        first_stage_residual_covariance=0.001,
    )
    monkeypatch.setattr(
        command,
        "run_direct_injection_benchmark",
        lambda **_kwargs: {"zero": direct},
    )
    monkeypatch.setattr(
        command,
        "run_inherited_instrument_benchmark",
        lambda **_kwargs: {"injected": inherited},
    )
    output = tmp_path / "sip-identification.json"

    result = CliRunner().invoke(
        command.app,
        ["--output", str(output), "--count", "8", "--steps", "4"],
    )

    assert result.exit_code == 0
    assert json.loads(output.read_text()) == {
        "configuration": {
            "count": 8,
            "steps": 4,
            "direct_seed": 200,
            "inherited_seed": 300,
        },
        "direct-injection": {
            "zero": {
                "true_effect": 1.7,
                "estimated_effect": 1.6,
                "residual_instrument_covariance": 0.01,
                "reconstruction_loss": 0.02,
                "effect_error": pytest.approx(0.1),
            }
        },
        "instrument-inheritance": {
            "injected": {
                "true_effect": 1.7,
                "estimated_effect": 1.65,
                "residual_instrument_covariance": 0.02,
                "reconstruction_loss": 0.03,
                "true_first_stage_effect": 1.3,
                "estimated_first_stage_effect": 1.2,
                "first_stage_residual_covariance": 0.001,
                "effect_error": pytest.approx(0.05),
            }
        },
    }
