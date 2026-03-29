from __future__ import annotations

from abc import abstractmethod
from collections.abc import Sequence
from typing import Any, ClassVar

from cem.structure.solution import InferenceResults, Telemetries, TrainingResults
from cem.structure.solver import Solver

from .plotter import Plotter


class Demo:
    name: ClassVar[str]
    title: ClassVar[str]

    @classmethod
    @abstractmethod
    def solver(cls) -> Solver[Any]:
        """Return the solver that defines the model, problem, and hyperparameters for this demo."""
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def demo_loss(
        cls, training_results: TrainingResults, inference_results: InferenceResults
    ) -> float:
        """Return a scalar loss used to drive hyperparameter optimisation.

        Args:
            training_results: Results from the training run.
            inference_results: Results from the inference run.

        Returns:
            A scalar loss; lower is better.
        """
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def plotters(cls) -> Sequence[Plotter]:
        """Return the plotters that visualise results for this demo."""
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def create_telemetries(cls) -> Telemetries:
        """Return additional telemetries beyond those declared by the plotters.

        The telemetries returned here are automatically combined with those
        from each plotter in ``plotters()``.
        """
        raise NotImplementedError
