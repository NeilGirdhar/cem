"""Reliability curve for NPN predicted variance across the four supervised datasets."""

import itertools as it
from collections.abc import Mapping, Sequence
from dataclasses import KW_ONLY, replace
from typing import Any, override

import numpy as np
from tjax.dataclasses import field

from cem.structure.plotter import Experiment, Run, Variant
from cem.structure.plotter.plotter import PlottedSeries, Plotter
from cem.structure.solution import PredictionTelemetry, Telemetries
from cem.structure.solver import Solver

from .models import DatasetKind, LinkKind, SupervisedSolver

_SEED_COUNT = 3
_BINS = 10
_TARGET_NODE = "target"
_TARGET_FIELD = "y"
_HIDDEN_SIZES: dict[DatasetKind, int] = {
    DatasetKind.iris: 220,
    DatasetKind.bike_sharing_demand: 98,
    DatasetKind.cpu_activity: 20,
    DatasetKind.elevators: 85,
}
_TRAINING_EXAMPLES = {
    DatasetKind.iris: 1168,
    DatasetKind.bike_sharing_demand: 1216,
    DatasetKind.cpu_activity: 882,
    DatasetKind.elevators: 2456,
}
_DATASET_TITLES = {
    DatasetKind.iris: "Iris",
    DatasetKind.bike_sharing_demand: "Bike-Sharing Demand",
    DatasetKind.cpu_activity: "CPU Activity",
    DatasetKind.elevators: "Elevators",
}


def _bin_by_quantile(
    predicted_variance: np.ndarray,
    squared_residual: np.ndarray,
    bins: int,
) -> tuple[list[float], list[float], list[int]]:
    """Return per-bin (mean predicted variance, mean squared residual, count) by quantile."""
    order = np.argsort(predicted_variance)
    variance_sorted = predicted_variance[order]
    residual_sorted = squared_residual[order]
    edges = np.linspace(0, variance_sorted.shape[0], bins + 1).astype(int)
    predicted_means: list[float] = []
    empirical_means: list[float] = []
    counts: list[int] = []
    for start, end in it.pairwise(edges):
        if end <= start:
            continue
        predicted_means.append(float(variance_sorted[start:end].mean()))
        empirical_means.append(float(residual_sorted[start:end].mean()))
        counts.append(int(end - start))
    return predicted_means, empirical_means, counts


class CalibrationPlotter(Plotter):
    """Reliability curve for one dataset: mean predicted σ² vs. mean squared residual per bin."""

    _: KW_ONLY
    name: str = field(static=True)
    title: str = field(static=True)
    dataset: DatasetKind = field(static=True)
    bins: int = field(static=True, default=_BINS)

    @override
    def telemetries(self) -> Telemetries:
        return Telemetries(
            (PredictionTelemetry(selected_node=_TARGET_NODE, selected_field=_TARGET_FIELD),)
        )

    @override
    def plotted_series(self, runs: Sequence[Run]) -> PlottedSeries:
        telemetry = PredictionTelemetry(selected_node=_TARGET_NODE, selected_field=_TARGET_FIELD)
        predicted_variance_batches: list[np.ndarray] = []
        squared_residual_batches: list[np.ndarray] = []
        for run in runs:
            if DatasetKind(run.sweep_entry["dataset_kind"]) is not self.dataset:
                continue
            snapshot = np.asarray(run.inference_results.telemetries[telemetry])
            if snapshot.size == 0:
                continue
            flat = snapshot.reshape(-1, 3)
            predicted_mean = flat[:, 0]
            predicted_variance = flat[:, 1]
            observed = flat[:, 2]
            residual = predicted_mean - observed
            predicted_variance_batches.append(predicted_variance)
            squared_residual_batches.append(residual * residual)
        if not predicted_variance_batches:
            return {"iteration": [], "line plots": {}}
        predicted_variance = np.concatenate(predicted_variance_batches)
        squared_residual = np.concatenate(squared_residual_batches)
        predicted_means, empirical_means, counts = _bin_by_quantile(
            predicted_variance, squared_residual, self.bins
        )
        return {
            "iteration": list(range(len(predicted_means))),
            "line plots": {
                "predicted": f"{_DATASET_TITLES[self.dataset]} predicted σ²",
                "empirical": f"{_DATASET_TITLES[self.dataset]} empirical squared residual",
            },
            "predicted": predicted_means,
            "empirical": empirical_means,
            "counts": counts,
        }


class SupervisedCalibrationExperiment(Experiment):
    """NPN predicted-variance calibration across the four tabular regression datasets."""

    def __init__(self) -> None:
        super().__init__(
            name="supervised-calibration",
            variants=[
                Variant(
                    label="",
                    solver=SupervisedSolver(
                        dataset_kind=DatasetKind.iris,
                        link_kind=LinkKind.natural_parameter,
                        training_batch_size=128,
                        inference_examples=256,
                        inference_batch_size=128,
                        learning_rate=0.01,
                    ),
                ),
            ],
            plotters=[
                CalibrationPlotter(
                    name=f"calibration-{dataset.value.replace('_', '-')}",
                    title=f"{_DATASET_TITLES[dataset]} calibration",
                    dataset=dataset,
                )
                for dataset in _HIDDEN_SIZES
            ],
            sweep=[
                {"dataset_kind": dataset, "seed": seed}
                for dataset in _HIDDEN_SIZES
                for seed in range(_SEED_COUNT)
            ],
        )

    @override
    def resolve_solver(
        self,
        variant: Variant,
        sweep_entry: Mapping[str, Any],
        hyperparameters: Mapping[str, Any],
    ) -> Solver[Any]:
        del hyperparameters
        dataset_kind = sweep_entry["dataset_kind"]
        seed = sweep_entry["seed"]
        base = variant.solver
        assert isinstance(base, SupervisedSolver)
        return replace(
            base,
            dataset_kind=dataset_kind,
            hidden_size=_HIDDEN_SIZES[dataset_kind],
            training_examples=_TRAINING_EXAMPLES[dataset_kind],
            parameters_seed=seed,
            training_seed=100 + seed,
            inference_seed=200 + seed,
        )


supervised_calibration_experiment = SupervisedCalibrationExperiment()
