"""Shared pieces of the SIP experiments: seeded runs and bar-chart series."""

from collections.abc import Mapping, Sequence
from dataclasses import replace
from statistics import fmean, pstdev
from typing import Any, override

from cem.framework.plotter import Experiment, Variant
from cem.framework.plotter.plotter import PlottedSeries
from cem.framework.solver import Solver

SEED_COUNT = 5


def reference_seed(seed: int) -> int:
    """Return the seed of the sample that fits a reference estimator for one experiment seed.

    The reference sample is independent of the seed's training samples (seed 100 + seed) and its
    held-out samples (seed 200 + seed), so a reference fitted on it is scored out of sample.
    """
    return 300 + seed


def bar_chart(
    x_labels: Sequence[str], series: Mapping[str, tuple[str, Sequence[Sequence[float]]]]
) -> PlottedSeries:
    """Plot the mean of each series at each x position with one standard deviation across seeds.

    Each series maps its key to its legend label and to its seed values at each x position.
    """
    return {
        "iteration": list(range(len(x_labels))),
        "x labels": list(x_labels),
        "bar plots": {key: label for key, (label, _) in series.items()},
        "bar errors": {
            key: [pstdev(values) for values in seed_values]
            for key, (_, seed_values) in series.items()
        },
        **{
            key: [fmean(values) for values in seed_values]
            for key, (_, seed_values) in series.items()
        },
        "seed values": {
            key: [list(values) for values in seed_values]
            for key, (_, seed_values) in series.items()
        },
        "error bars": "one population standard deviation across seeds",
    }


class SeededExperiment(Experiment):
    """Run one solver over several seeds.

    Each seed trains from fresh initial parameters and evaluates on its own held-out samples,
    where every estimator sees the same samples.
    """

    @override
    def resolve_solver(
        self,
        variant: Variant,
        sweep_entry: Mapping[str, Any],
        hyperparameters: Mapping[str, Any],
    ) -> Solver[Any]:
        """Give each seed its own initial parameters and its own training and held-out samples."""
        solver = super().resolve_solver(variant, sweep_entry, hyperparameters)
        seed = sweep_entry["seed"]
        return replace(
            solver, parameters_seed=seed, training_seed=100 + seed, inference_seed=200 + seed
        )
