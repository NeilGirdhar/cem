"""Run solver variants across a sweep and collect their results for charting."""

from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from functools import reduce
from typing import Any

from optuna.distributions import BaseDistribution
from rich.progress import Progress

from cem.framework.reporting import console_progress_bar
from cem.framework.solution import ExecutionPacket, Telemetries
from cem.framework.solution.wandb_tools import WAndBInitSettings
from cem.framework.solver import Solver
from cem.tuned_defaults import tuned_defaults_for_experiment

from .plotter import Plotter, Run

_EMPTY_TELEMETRIES = Telemetries()
_EMPTY_SWEEP: tuple[Mapping[str, Any], ...] = ({},)


@dataclass(frozen=True)
class Variant:
    """A solver configuration and its label for comparing alternatives in one experiment.

    The label identifies chart series and scopes variant-specific hyperparameters.
    A single-variant experiment can leave the label empty.
    """

    label: str
    solver: Solver[Any]


class Experiment:
    """Compare solver variants under the same sweep and collect results for plotters.

    Each combination of variant and sweep entry produces a separate run. A
    single-variant experiment can use an empty variant label.
    """

    def __init__(
        self,
        *,
        name: str,
        variants: Sequence[Variant],
        plotters: Sequence[Plotter],
        sweep: Sequence[Mapping[str, Any]] = (),
        shared_hyperparameter_names: frozenset[str] = frozenset(),
        extra_telemetries: Telemetries = _EMPTY_TELEMETRIES,
    ) -> None:
        """Set up the comparison grid and the charts that consume its results.

        Supply at least one variant and label each variant in a multi-variant
        experiment. An empty sweep runs each variant once without sweep overrides.
        """
        if not variants:
            msg = "Experiment requires at least one Variant"
            raise ValueError(msg)
        if len(variants) > 1 and not all(v.label for v in variants):
            msg = "each Variant in a multi-variant Experiment must have a non-empty label"
            raise ValueError(msg)
        self.name = name
        self.variants = list(variants)
        self.plotters = list(plotters)
        self.sweep: tuple[Mapping[str, Any], ...] = (
            tuple(dict(entry) for entry in sweep) if sweep else _EMPTY_SWEEP
        )
        self.shared_hyperparameter_names = shared_hyperparameter_names
        self.extra_telemetries = extra_telemetries

    def all_telemetries(self) -> Telemetries:
        """Combine the telemetry requested by all plotters with explicitly requested telemetry."""
        return reduce(
            Telemetries.combine,
            (plotter.telemetries() for plotter in self.plotters),
            self.extra_telemetries,
        )

    def create_hyperparameters(self) -> dict[str, BaseDistribution]:
        """Return the Optuna search space for the experiment.

        With multiple variants, shared names use the first solver's distribution.
        All other names are prefixed with their variant label and a dot.
        """
        if len(self.variants) == 1:
            return self.variants[0].solver.create_hyperparameters()
        shared_names = self.shared_hyperparameter_names
        first_hyper = self.variants[0].solver.create_hyperparameters()
        result: dict[str, BaseDistribution] = {
            k: v for k, v in first_hyper.items() if k in shared_names
        }
        for variant in self.variants:
            prefix = f"{variant.label}."
            hyper = variant.solver.create_hyperparameters()
            result.update({f"{prefix}{k}": v for k, v in hyper.items() if k not in shared_names})
        return result

    def default_hyperparameters(self) -> dict[str, Any]:
        """Overlay saved tuned values on the solver defaults using experiment-level names."""
        first_defaults = self.variants[0].solver.default_hyperparameters()
        shared_names = (
            frozenset(first_defaults)
            if len(self.variants) == 1
            else self.shared_hyperparameter_names
        )
        result: dict[str, Any] = {k: v for k, v in first_defaults.items() if k in shared_names}
        for variant in self.variants:
            prefix = f"{variant.label}."
            defaults = variant.solver.default_hyperparameters()
            result.update({f"{prefix}{k}": v for k, v in defaults.items() if k not in shared_names})
        return result | tuned_defaults_for_experiment(self.name)

    def variant_hyperparameters(
        self,
        variant: Variant,
        hyperparameters: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Select the shared values and the variant's own values, stripping its name prefix."""
        if len(self.variants) == 1:
            return dict(hyperparameters)
        prefix = f"{variant.label}."
        resolved = {
            k: v for k, v in hyperparameters.items() if k in self.shared_hyperparameter_names
        }
        resolved.update(
            {k[len(prefix) :]: v for k, v in hyperparameters.items() if k.startswith(prefix)}
        )
        return resolved

    def resolve_solver(
        self,
        variant: Variant,
        sweep_entry: Mapping[str, Any],
        hyperparameters: Mapping[str, Any],
    ) -> Solver[Any]:
        """Configure a solver for one variant and sweep entry.

        The base implementation applies the selected hyperparameters. Override it
        when a sweep entry also sets fields such as the dataset or seed.
        """
        del sweep_entry
        return variant.solver.populate_from_hyperparameters(
            self.variant_hyperparameters(variant, hyperparameters)
        )

    def variant_for(self, run: Run) -> Variant:
        """Return the variant that produced ``run``."""
        for variant in self.variants:
            if variant.label == run.variant_label:
                return variant
        msg = f"no variant with label {run.variant_label!r} in experiment {self.name!r}"
        raise KeyError(msg)

    def solver_for(self, run: Run, hyperparameters: Mapping[str, Any]) -> Solver[Any]:
        """Reconstruct the solver for a run using its variant and sweep overrides."""
        return self.resolve_solver(self.variant_for(run), run.sweep_entry, hyperparameters)

    def runs(
        self,
        hyperparameters: Mapping[str, Any],
        *,
        progress_manager: Progress | None = None,
        wandb_settings: WAndBInitSettings | None = None,
        enable_profiling: bool = False,
    ) -> list[Run]:
        """Train and evaluate each variant at every sweep entry with the requested telemetry."""
        telemetries = self.all_telemetries()
        collected: list[Run] = []
        for variant in self.variants:
            for sweep_entry in self.sweep:
                solver = self.resolve_solver(variant, sweep_entry, hyperparameters)
                packet = ExecutionPacket(
                    progress_manager=progress_manager,
                    run_label=variant.label or None,
                    telemetries=telemetries,
                    wandb_settings=wandb_settings,
                    enable_profiling=enable_profiling,
                )
                training_results, inference_results = solver.training_and_inference_result(
                    packet=packet,
                )
                collected.append(
                    Run(
                        variant_label=variant.label,
                        sweep_entry=dict(sweep_entry),
                        training_results=training_results,
                        inference_results=inference_results,
                    )
                )
        return collected

    def generate(self, hyperparameters: Mapping[str, Any] | None = None) -> dict[str, Any]:
        """Run the comparison and return chart data keyed by plotter name."""
        resolved = (
            dict(hyperparameters) if hyperparameters is not None else self.default_hyperparameters()
        )
        collected = self.runs(resolved, progress_manager=console_progress_bar())
        return {plotter.name: plotter.plotted_series(collected) for plotter in self.plotters}


class OptimizableExperiment(Experiment, ABC):
    """An experiment with an objective that can be used to tune its hyperparameters."""

    @abstractmethod
    def experiment_loss(
        self,
        runs: Sequence[Run],
        hyperparameters: Mapping[str, Any],
    ) -> float:
        """Return the scalar optimization objective for a complete experiment run."""
        raise NotImplementedError
