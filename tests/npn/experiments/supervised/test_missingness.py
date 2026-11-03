from typing import cast

import jax.numpy as jnp

from cem.npn.experiments.supervised.missingness import MissingnessBarPlotter
from cem.npn.experiments.supervised.models import DatasetKind
from cem.structure.plotter import Run
from cem.structure.solution import (
    InferenceResults,
    LossTelemetry,
    SolutionState,
    TrainingResults,
)

_INFERENCE_BATCH_SIZE = 128


def _run(variant_label: str, dataset_kind: DatasetKind, seed: int, loss: float) -> Run:
    telemetry = LossTelemetry(selected_node="target")
    inference = InferenceResults(
        count=1,
        telemetries={telemetry: jnp.array([loss * _INFERENCE_BATCH_SIZE])},
    )
    training = TrainingResults(
        count=1,
        telemetries={telemetry: jnp.array([0.0])},
        final_state=cast("SolutionState", None),
    )
    return Run(
        variant_label=variant_label,
        sweep_entry={"dataset_kind": dataset_kind, "seed": seed},
        training_results=training,
        inference_results=inference,
    )


def test_missingness_bar_plotter_reports_means_and_deviations() -> None:
    runs = [
        _run("npn", DatasetKind.iris, 0, 1.0),
        _run("npn", DatasetKind.iris, 1, 3.0),
        _run("npn", DatasetKind.bike_sharing_demand, 0, 2.0),
        _run("npn", DatasetKind.bike_sharing_demand, 1, 6.0),
        _run("mask-mlp", DatasetKind.iris, 0, 2.0),
        _run("mask-mlp", DatasetKind.iris, 1, 4.0),
        _run("mask-mlp", DatasetKind.bike_sharing_demand, 0, 4.0),
        _run("mask-mlp", DatasetKind.bike_sharing_demand, 1, 8.0),
    ]

    result = MissingnessBarPlotter().plotted_series(runs)

    assert result["npn"][:2] == [2.0, 4.0]
    assert result["mask-mlp"][:2] == [3.0, 6.0]
    assert result["bar errors"]["npn"][:2] == [1.0, 2.0]
    assert result["bar errors"]["mask-mlp"][:2] == [1.0, 2.0]
    assert result["seed losses"]["npn"][:2] == [[1.0, 3.0], [2.0, 6.0]]
    assert result["seed losses"]["mask-mlp"][:2] == [[2.0, 4.0], [4.0, 8.0]]
