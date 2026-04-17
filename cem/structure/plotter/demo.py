from __future__ import annotations

from abc import abstractmethod
from collections.abc import Sequence
from functools import reduce
from typing import Any

from optuna.distributions import BaseDistribution

from cem.structure.solution import InferenceResults, Telemetries, TrainingResults
from cem.structure.solver import Solver

from .plotter import Plotter


class Variant:
    """Abstract base for a specific experimental configuration.

    A Variant knows how to build a solver, define the loss, provide plotters, and
    declare telemetries.  It carries no identity of its own — names live on ``Demo``.
    For multi-variant demos, ``label`` must be set to a non-empty string to distinguish
    this variant in plots and hyperparameter prefixes.
    """

    label: str = ""

    def all_telemetries(self) -> Telemetries:
        return reduce(
            Telemetries.combine,
            (plotter.telemetries() for plotter in self.plotters()),
            self.extra_telemetries(),
        )

    @abstractmethod
    def create_solver(self) -> Solver[Any]:
        """Return the solver that defines the model, problem, and hyperparameters."""
        raise NotImplementedError

    @abstractmethod
    def demo_loss(
        self, training_results: TrainingResults, inference_results: InferenceResults
    ) -> float:
        """Return a scalar loss used to drive hyperparameter optimisation.

        Args:
            training_results: Results from the training run.
            inference_results: Results from the inference run.

        Returns:
            A scalar loss; lower is better.
        """
        raise NotImplementedError

    @abstractmethod
    def plotters(self) -> Sequence[Plotter]:
        """Return the plotters that visualise results for this variant."""
        raise NotImplementedError

    @abstractmethod
    def extra_telemetries(self) -> Telemetries:
        """Return additional telemetries beyond those declared by the plotters."""
        raise NotImplementedError

    def shared_hyperparameter_names(self) -> frozenset[str]:
        """Names of hyperparameters shared across all variants in a multi-variant Demo.

        Shared parameters appear once in the joint search space (no variant prefix) and
        are passed unchanged to every variant's ``populate_from_hyperparameters``.
        Override to declare which parameter names should be shared.
        """
        return frozenset()


class Demo:
    """A named demo containing one or more Variants.

    Single-variant: ``Demo(name=..., variants=[SomeVariant()])``
    Multi-variant:  ``Demo(name=..., variants=[V1(label="a"), V2(label="b")])``

    The name is used as the Optuna study identifier and CLI key.  Multi-variant demos
    optimise jointly: ``create_hyperparameters`` prefixes each variant's parameters with
    ``"{variant.label}."`` and the optimisation loss is ``max`` over variants.
    """

    name: str
    variants: list[Variant]

    def __init__(self, *, name: str, variants: Sequence[Variant]) -> None:
        if not variants:
            msg = "Demo requires at least one Variant"
            raise ValueError(msg)
        if len(variants) > 1 and not all(v.label for v in variants):
            msg = "each Variant in a multi-variant Demo must have a non-empty label"
            raise ValueError(msg)
        self.name = name
        self.variants = list(variants)

    def create_hyperparameters(self) -> dict[str, BaseDistribution]:
        """Return the hyperparameter search space for this demo.

        For single-variant demos this is the solver's hyperparameter space.  For
        multi-variant demos, parameters named in ``Variant.shared_hyperparameter_names``
        appear once (no prefix); all other parameters are prefixed with
        ``"{variant.label}."``.
        """
        if len(self.variants) == 1:
            return self.variants[0].create_solver().create_hyperparameters()
        shared_names = frozenset.intersection(
            *(v.shared_hyperparameter_names() for v in self.variants)
        )
        first_hyper = self.variants[0].create_solver().create_hyperparameters()
        result: dict[str, BaseDistribution] = {
            k: v for k, v in first_hyper.items() if k in shared_names
        }
        for variant in self.variants:
            prefix = f"{variant.label}."
            hyper = variant.create_solver().create_hyperparameters()
            result.update({f"{prefix}{k}": v for k, v in hyper.items() if k not in shared_names})
        return result

    def plotters(self) -> Sequence[Plotter]:
        return self.variants[0].plotters()

    def all_telemetries(self) -> Telemetries:
        return reduce(
            Telemetries.combine,
            (v.all_telemetries() for v in self.variants),
            Telemetries(),
        )
