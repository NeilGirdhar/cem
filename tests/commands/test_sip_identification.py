import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from cem.commands import app
from cem.sip.experiments import sip_identification
from cem.sip.experiments.benchmark import (
    CausalBenchmarkResult,
    CausalBenchmarkTrajectory,
    CreditBenchmarkResult,
)


def test_generate_sip_identification_writes_json(
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
        sip_identification,
        "run_direct_injection_benchmark",
        lambda **_kwargs: {"zero": zero, "random": random},
    )
    monkeypatch.setattr(
        sip_identification,
        "run_inherited_instrument_benchmark",
        lambda **_kwargs: {"policy": policy, "injected": inherited},
    )
    monkeypatch.setattr(
        sip_identification,
        "run_td_error_benchmark",
        lambda **_kwargs: {"ordinary": ordinary_credit, "td": td_credit},
    )
    output_dir = tmp_path / "out"

    result = CliRunner().invoke(
        app,
        ["generate", "sip-identification", "--output-dir", str(output_dir), "--no-log"],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads((output_dir / "sip-identification.json").read_text())
    balance = payload.pop("td-error-balance")
    trajectory_count = 8
    trajectory_steps = 16
    assert balance["iteration"] == list(range(trajectory_steps + 1))
    assert len(balance["line plots"]) == trajectory_count
    assert all(len(balance[name]) == trajectory_steps + 1 for name in balance["line plots"])
    assert all(balance[name][0] == pytest.approx(1.0) for name in balance["line plots"])
    assert payload["direct-injection"] == {
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
    }
