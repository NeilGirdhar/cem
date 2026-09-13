import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

import cem.commands.supervised_missingness as command
from cem.demos.supervised.missingness import summarize_missingness_losses


def test_summarize_missingness_losses_reports_means_and_deviations() -> None:
    losses = {
        "npn": [[1.0, 3.0], [2.0, 6.0]],
        "mask-mlp": [[2.0, 4.0], [4.0, 8.0]],
    }

    result = summarize_missingness_losses(losses)

    assert result["npn"] == [2.0, 4.0]
    assert result["mask-mlp"] == [3.0, 6.0]
    assert result["bar errors"] == {
        "npn": [1.0, 2.0],
        "mask-mlp": [1.0, 2.0],
    }
    assert result["seed losses"] == losses


def test_supervised_missingness_cli_writes_json(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expected = {"missing-data-distributional": {"npn": [0.1]}}
    monkeypatch.setattr(command, "run_missingness_comparison", lambda **_kwargs: expected)
    output = tmp_path / "missingness.json"

    result = CliRunner().invoke(
        command.app,
        ["--output", str(output), "--seeds", "2"],
    )

    assert result.exit_code == 0
    assert json.loads(output.read_text()) == expected
