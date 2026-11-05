"""Compare capacity-matched supervised models on missing inputs across datasets and seeds."""

from collections.abc import Mapping, Sequence
from dataclasses import KW_ONLY, replace
from statistics import fmean, pstdev
from typing import Any, override

import jax.numpy as jnp
from tjax.dataclasses import field

from cem.structure.plotter import Experiment, Run, Variant
from cem.structure.plotter.plotter import PlottedSeries, Plotter
from cem.structure.solution import Telemetries
from cem.structure.solution.loss_telemetry import LossTelemetry
from cem.structure.solver import Solver

from .models import DatasetKind, LinkKind, SupervisedSolver

_SEED_COUNT = 3
_INFERENCE_BATCH_SIZE = 128
_MODEL_LABELS = {
    "npn": "Gaussian NPN",
    "mask-mlp": "Mask-aware MLP",
}
_HIDDEN_SIZES: dict[str, dict[DatasetKind, int]] = {
    "npn": {
        DatasetKind.iris: 220,
        DatasetKind.bike_sharing_demand: 98,
        DatasetKind.cpu_activity: 20,
        DatasetKind.elevators: 85,
    },
    "mask-mlp": {
        DatasetKind.iris: 133,
        DatasetKind.bike_sharing_demand: 57,
        DatasetKind.cpu_activity: 11,
        DatasetKind.elevators: 46,
    },
}
_TRAINING_EXAMPLES = {
    DatasetKind.iris: 1168,
    DatasetKind.bike_sharing_demand: 1216,
    DatasetKind.cpu_activity: 882,
    DatasetKind.elevators: 2456,
}
_DATASET_LABELS = {
    DatasetKind.iris: "Iris",
    DatasetKind.bike_sharing_demand: "Bike",
    DatasetKind.cpu_activity: "CPU",
    DatasetKind.elevators: "Elevators",
}


class MissingnessBarPlotter(Plotter):
    """Compare mean held-out losses by dataset and model, with variation across matched seeds."""

    _: KW_ONLY
    name: str = field(static=True, default="missing-data-distributional")
    title: str = field(
        static=True,
        default="Held-out distributional loss, 50% missing inputs",
    )

    @override
    def telemetries(self) -> Telemetries:
        return Telemetries((LossTelemetry(selected_node="target"),))

    @override
    def plotted_series(self, runs: Sequence[Run]) -> PlottedSeries:
        """Build per-example loss bars with one population standard deviation across seeds."""
        telemetry = LossTelemetry(selected_node="target")
        seen = {DatasetKind(run.sweep_entry["dataset_kind"]) for run in runs}
        datasets_in_order = [dataset for dataset in _DATASET_LABELS if dataset in seen]
        losses: dict[str, dict[DatasetKind, list[float]]] = {
            label: {dataset: [] for dataset in datasets_in_order} for label in _MODEL_LABELS
        }
        for run in runs:
            dataset = DatasetKind(run.sweep_entry["dataset_kind"])
            inference = run.inference_results
            loss = float(jnp.mean(inference.telemetries[telemetry])) / _INFERENCE_BATCH_SIZE
            losses[run.variant_label][dataset].append(loss)
        means = {
            label: [fmean(losses[label][dataset]) for dataset in datasets_in_order]
            for label in _MODEL_LABELS
        }
        deviations = {
            label: [pstdev(losses[label][dataset]) for dataset in datasets_in_order]
            for label in _MODEL_LABELS
        }
        seed_losses = {
            label: [list(losses[label][dataset]) for dataset in datasets_in_order]
            for label in _MODEL_LABELS
        }
        return {
            "iteration": list(range(len(datasets_in_order))),
            "x labels": [_DATASET_LABELS[dataset] for dataset in datasets_in_order],
            "bar plots": dict(_MODEL_LABELS),
            "bar errors": deviations,
            **means,
            "seed losses": seed_losses,
            "error bars": "one population standard deviation across matched seeds",
        }


class SupervisedMissingnessExperiment(Experiment):
    """Compare a Gaussian NPN and a mask-aware MLP when half the inputs are missing on average."""

    def __init__(self) -> None:
        super().__init__(
            name="supervised-missingness",
            variants=[
                Variant(
                    label="npn",
                    solver=SupervisedSolver(
                        dataset_kind=DatasetKind.iris,
                        link_kind=LinkKind.natural_parameter,
                        training_batch_size=128,
                        inference_examples=64,
                        inference_batch_size=_INFERENCE_BATCH_SIZE,
                        learning_rate=0.01,
                        missing_probability=0.5,
                        random_missing_values=False,
                        include_missingness_mask=False,
                    ),
                ),
                Variant(
                    label="mask-mlp",
                    solver=SupervisedSolver(
                        dataset_kind=DatasetKind.iris,
                        link_kind=LinkKind.perceptron,
                        training_batch_size=128,
                        inference_examples=64,
                        inference_batch_size=_INFERENCE_BATCH_SIZE,
                        learning_rate=0.01,
                        missing_probability=0.5,
                        random_missing_values=False,
                        include_missingness_mask=True,
                    ),
                ),
            ],
            plotters=[MissingnessBarPlotter()],
            sweep=[
                {"dataset_kind": dataset, "seed": seed}
                for dataset in _HIDDEN_SIZES["npn"]
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
        """Apply the dataset budget and model-specific width while matching seeds across models."""
        del hyperparameters
        dataset_kind = sweep_entry["dataset_kind"]
        seed = sweep_entry["seed"]
        base = variant.solver
        assert isinstance(base, SupervisedSolver)
        return replace(
            base,
            dataset_kind=dataset_kind,
            hidden_size=_HIDDEN_SIZES[variant.label][dataset_kind],
            training_examples=_TRAINING_EXAMPLES[dataset_kind],
            parameters_seed=seed,
            training_seed=100 + seed,
            inference_seed=200 + seed,
        )


supervised_missingness_experiment = SupervisedMissingnessExperiment()
