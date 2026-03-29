from __future__ import annotations

from dataclasses import KW_ONLY
from typing import Any

import equinox as eqx

from cem.structure.solution import InferenceResults, Telemetries, TrainingResults
from cem.structure.solver import Solver
from .plot import Plot


class ExecutedSolverResults(eqx.Module):
    solver: Solver[Any]
    training_results: TrainingResults
    inference_results: InferenceResults


class Plotter(eqx.Module):
    _: KW_ONLY
    name: str = eqx.field(static=True)
    title: str = eqx.field(static=True)

    def plot(
        self, plot: Plot, results: tuple[ExecutedSolverResults, ...], solver_index: int
    ) -> None:
        raise NotImplementedError

    def telemetries(self) -> Telemetries:
        return Telemetries()
