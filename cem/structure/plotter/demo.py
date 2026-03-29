from __future__ import annotations

from abc import abstractmethod
from collections.abc import Sequence
from typing import Any, ClassVar

from cem.structure.solution import ExecutionPacket, Telemetries
from cem.structure.solver import Solver, Trainer
from .plotter import Plotter


class Demo[T: Trainer[Any]]:
    name: ClassVar[str]
    title: ClassVar[str]

    @classmethod
    @abstractmethod
    def create_solvers(cls) -> Sequence[Solver[T]]:
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def demo_loss(cls, solvers: Sequence[Solver[T]], packet: ExecutionPacket) -> float:
        """The demo loss is used to optimize hyperparameters.

        The default value of zero means that hyperparameters can't be optimized.
        """
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def plotters(cls) -> Sequence[Plotter]:
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def telemetries(cls) -> Telemetries:
        raise NotImplementedError
