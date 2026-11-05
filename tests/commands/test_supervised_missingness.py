"""Check that the CLI writes chart JSON for the missing-input experiment."""

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from cem.commands import app
from cem.npn.experiments.supervised.missingness import SupervisedMissingnessExperiment


def test_generate_supervised_missingness_writes_json(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expected = {"missing-data-distributional": {"npn": [0.1]}}
    monkeypatch.setattr(
        SupervisedMissingnessExperiment,
        "generate",
        lambda _self, _hyperparameters=None: expected,
    )
    output_dir = tmp_path / "out"

    result = CliRunner().invoke(
        app,
        ["generate", "supervised-missingness", "--output-dir", str(output_dir), "--no-log"],
    )

    assert result.exit_code == 0, result.output
    assert json.loads((output_dir / "supervised-missingness.json").read_text()) == expected
