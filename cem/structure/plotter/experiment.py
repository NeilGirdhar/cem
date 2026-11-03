"""The :class:`Experiment` harness: variants crossed with a sweep, rendered by plotters."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from functools import reduce
from typing import Any

from optuna.distributions import BaseDistribution
from rich.progress import Progress

from cem.structure.reporting import console_progress_bar
from cem.structure.solution import ExecutionPacket, Telemetries
from cem.structure.solution.wandb_tools import WAndBInitSettings
from cem.structure.solver import Solver
from cem.tuned_defaults import tuned_defaults_for_experiment

from .plotter import Plotter, Run

_EMPTY_TELEMETRIES = Telemetries()
_EMPTY_SWEEP: tuple[Mapping[str, Any], ...] = ({},)


@dataclass(frozen=True)
class Variant:
    """One alternative solver compared inside an :class:`Experiment`.

    Attributes:
        label: Non-empty for multi-variant experiments; used as the hyperparameter
            prefix and chart series key. Empty for single-variant experiments.
        solver: Base solver carrying this variant's problem and model. The
            hyperparameters (and any :meth:`Experiment.resolve_solver` overrides)
            are applied on top at run time.
    """

    label: str
    solver: Solver[Any]


class Experiment:
    r"""A named experiment: ``variants × sweep`` evaluated and rendered by plotters.

    Single-variant: ``Experiment(name=..., variants=[Variant(label="", ...)], ...)``.
    Multi-variant: ``Experiment(name=..., variants=[Variant(label="a", ...), ...], ...)``.

    The sweep is a sequence of parameter overrides applied to every variant. Each
    (variant, sweep entry) pair is one :class:`Run`; plotters receive all runs and
    pivot them into chart data.

    Multi-variant experiments optimize jointly: :meth:`create_hyperparameters`
    prefixes each variant's parameters with ``"{variant.label}."`` except for the
    names listed in ``shared_hyperparameter_names``, and the loss is scored by
    subclasses that override :meth:`experiment_loss`.
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
        return reduce(
            Telemetries.combine,
            (plotter.telemetries() for plotter in self.plotters),
            self.extra_telemetries,
        )

    def create_hyperparameters(self) -> dict[str, BaseDistribution]:
        """Return the Optuna search space for this experiment.

        For single-variant experiments this is the solver's hyperparameter space.
        For multi-variant experiments, names in ``shared_hyperparameter_names``
        appear once (no prefix); all other names are prefixed with
        ``"{variant.label}."``.
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
        """Return default values for :meth:`create_hyperparameters` overlaid with tuned ones."""
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
        """Project one variant's slice out of the experiment-level hyperparameters."""
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
        """Build the solver for one (variant, sweep entry) combination.

        Subclasses override this to apply sweep-dependent fields that aren't
        exposed as hyperparameters (e.g. :attr:`parameters_seed` or
        :attr:`dataset_kind`).
        """
        del sweep_entry
        return variant.solver.populate_from_hyperparameters(
            self.variant_hyperparameters(variant, hyperparameters)
        )

    def runs(
        self,
        hyperparameters: Mapping[str, Any],
        *,
        progress_manager: Progress | None = None,
        wandb_settings: WAndBInitSettings | None = None,
        enable_profiling: bool = False,
    ) -> list[Run]:
        """Train every (variant, sweep entry) pair and collect :class:`Run` objects."""
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
        """Train every (variant, sweep entry) pair and return the chart JSON."""
        resolved = (
            dict(hyperparameters) if hyperparameters is not None else self.default_hyperparameters()
        )
        collected = self.runs(resolved, progress_manager=console_progress_bar())
        return {plotter.name: plotter.plotted_series(collected) for plotter in self.plotters}

    def experiment_loss(
        self,
        runs: Sequence[Run],
        hyperparameters: Mapping[str, Any],
    ) -> float:
        """Return the scalar optimization objective for a complete experiment run."""
        raise NotImplementedError
