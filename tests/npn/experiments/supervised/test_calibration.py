"""Check calibration experiment registration and chart data from held-out predictions."""

from typing import cast

import jax.numpy as jnp
import numpy as np

from cem.commands.registry import experiments
from cem.framework.plotter import Run
from cem.framework.solution import (
    InferenceResults,
    PredictionTelemetry,
    SolutionState,
    TrainingResults,
)
from cem.npn.experiments.supervised.calibration import (
    CalibrationScatterPlotter,
    SupervisedCalibrationExperiment,
    supervised_calibration_experiment,
)
from cem.npn.experiments.supervised.models import DatasetKind


def _run(dataset: DatasetKind, seed: int, snapshot: jnp.ndarray) -> Run:
    telemetry = PredictionTelemetry(selected_node="target", selected_field="y")
    inference = InferenceResults(count=snapshot.shape[0], telemetries={telemetry: snapshot})
    training = TrainingResults(
        count=snapshot.shape[0],
        telemetries={telemetry: snapshot},
        final_state=cast("SolutionState", None),
    )
    return Run(
        variant_label="",
        sweep_entry={"dataset_kind": dataset, "seed": seed},
        training_results=training,
        inference_results=inference,
    )


def test_supervised_calibration_experiment_is_registered() -> None:
    assert experiments["supervised-calibration"] is supervised_calibration_experiment
    assert isinstance(supervised_calibration_experiment, SupervisedCalibrationExperiment)


def test_calibration_plotter_bins_by_variance_and_emits_per_dataset_series() -> None:
    iris_snapshot = jnp.asarray(
        [
            # [predicted_mean, predicted_variance, observed_value]
            [0.0, 0.1, 1.0],
            [0.0, 0.2, 2.0],
            [0.0, 0.3, 3.0],
            [0.0, 0.4, 4.0],
        ]
    )
    bike_snapshot = jnp.asarray(
        [
            [0.0, 0.05, 0.3],
            [0.0, 0.15, 0.6],
        ]
    )
    runs = [
        _run(DatasetKind.iris, 0, iris_snapshot),
        _run(DatasetKind.bike_sharing_demand, 0, bike_snapshot),
    ]

    result = CalibrationScatterPlotter(bins=2).plotted_series(runs)

    assert set(result["scatter series"]) == {"Iris", "Bike"}
    iris_series = result["scatter series"]["Iris"]
    assert iris_series["x"][0] < iris_series["x"][1]
    assert iris_series["y"] == [np.mean([1.0, 4.0]), np.mean([9.0, 16.0])]
    bike_series = result["scatter series"]["Bike"]
    assert bike_series["y"] == [0.3 * 0.3, 0.6 * 0.6]
    assert result["min"] > 0
    assert result["max"] >= max(iris_series["y"] + iris_series["x"])


def test_calibration_plotter_returns_empty_series_when_no_runs() -> None:
    result = CalibrationScatterPlotter().plotted_series([])

    assert result["scatter series"] == {}
    assert result["min"] == 0.0
    assert result["max"] == 1.0
