from typing import cast

import jax.numpy as jnp
import numpy as np

from cem.commands.registry import experiments
from cem.npn.experiments.supervised.calibration import (
    CalibrationPlotter,
    SupervisedCalibrationExperiment,
    supervised_calibration_experiment,
)
from cem.npn.experiments.supervised.models import DatasetKind
from cem.structure.plotter import Run
from cem.structure.solution import (
    InferenceResults,
    PredictionTelemetry,
    SolutionState,
    TrainingResults,
)


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


def test_calibration_plotter_filters_by_dataset_and_bins_by_variance() -> None:
    iris_snapshot = jnp.array(
        [
            # [predicted_mean, predicted_variance, observed_value]
            [0.0, 0.1, 1.0],
            [0.0, 0.2, 2.0],
            [0.0, 0.3, 3.0],
            [0.0, 0.4, 4.0],
        ]
    )
    bike_snapshot = jnp.array(
        [
            [0.0, 99.0, 99.0],
        ]
    )
    runs = [
        _run(DatasetKind.iris, 0, iris_snapshot),
        _run(DatasetKind.bike_sharing_demand, 0, bike_snapshot),
    ]

    result = CalibrationPlotter(
        name="calibration-iris",
        title="Iris calibration",
        dataset=DatasetKind.iris,
        bins=2,
    ).plotted_series(runs)

    assert result["iteration"] == [0, 1]
    assert result["counts"] == [2, 2]
    assert result["predicted"][0] < result["predicted"][1]
    assert result["empirical"] == [np.mean([1.0, 4.0]), np.mean([9.0, 16.0])]


def test_calibration_plotter_returns_empty_when_dataset_absent() -> None:
    result = CalibrationPlotter(
        name="calibration-iris",
        title="Iris",
        dataset=DatasetKind.iris,
        bins=4,
    ).plotted_series([])

    assert result == {"iteration": [], "line plots": {}}
