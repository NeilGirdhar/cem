import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

import cem.commands.supervised_missingness as command


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
