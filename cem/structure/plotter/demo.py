from collections.abc import Sequence
from dataclasses import dataclass
from functools import reduce
from typing import Any

from optuna.distributions import BaseDistribution

from cem.structure.solution import InferenceResults, Telemetries, TrainingResults
from cem.structure.solver import Solver
from cem.tuned_defaults import tuned_defaults_for_demo

from .plotter import Plotter

_EMPTY_TELEMETRIES = Telemetries()


@dataclass(frozen=True)
class Variant:
    """One alternative configuration compared inside a :class:`Demo`.

    Attributes:
        label: Non-empty for multi-variant demos; used as the hyperparameter prefix
            and chart series key. Empty for single-variant demos.
        solver: Base solver carrying this variant's problem and model.
            Hyperparameters, including sweep overrides, are applied on top.
    """

    label: str
    solver: Solver[Any]


class Demo:
    r"""A named demo of one or more :class:`Variant`\ s with shared plotters.

    Single-variant: ``Demo(name=..., variants=[Variant(label="", solver=...)])``.
    Multi-variant:  ``Demo(name=..., variants=[Variant(label="a", ...), Variant(label="b", ...)])``.

    The name is used as the Optuna study identifier and CLI key. Multi-variant demos
    optimize jointly: :meth:`create_hyperparameters` prefixes each variant's parameters
    with ``"{variant.label}."`` except for the names listed in
    ``shared_hyperparameter_names`` and the optimization loss is scored by subclasses.
    """

    def __init__(
        self,
        *,
        name: str,
        variants: Sequence[Variant],
        plotters: Sequence[Plotter],
        shared_hyperparameter_names: frozenset[str] = frozenset(),
        extra_telemetries: Telemetries = _EMPTY_TELEMETRIES,
    ) -> None:
        if not variants:
            msg = "Demo requires at least one Variant"
            raise ValueError(msg)
        if len(variants) > 1 and not all(v.label for v in variants):
            msg = "each Variant in a multi-variant Demo must have a non-empty label"
            raise ValueError(msg)
        self.name = name
        self.variants = list(variants)
        self.plotters = list(plotters)
        self.shared_hyperparameter_names = shared_hyperparameter_names
        self.extra_telemetries = extra_telemetries

    def all_telemetries(self) -> Telemetries:
        return reduce(
            Telemetries.combine,
            (plotter.telemetries() for plotter in self.plotters),
            self.extra_telemetries,
        )

    def create_hyperparameters(self) -> dict[str, BaseDistribution]:
        """Return the hyperparameter search space for this demo.

        For single-variant demos this is the solver's hyperparameter space. For
        multi-variant demos, names in ``shared_hyperparameter_names`` appear once
        (no prefix); all other names are prefixed with ``"{variant.label}."``.
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
        """Return default values for :meth:`create_hyperparameters`, overlaid with tuned ones."""
        first_defaults = self.variants[0].solver.default_hyperparameters()
        # With one variant, every hyperparameter is effectively shared because no variant prefix
        # should appear in the public hyperparameter names.
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
        return result | tuned_defaults_for_demo(self.name)

    def demo_loss(
        self,
        variant_results: Sequence[tuple[Variant, TrainingResults, InferenceResults]],
        hyperparameters: dict[str, Any],
    ) -> float:
        """Return the scalar optimization objective for a complete demo run."""
        raise NotImplementedError
