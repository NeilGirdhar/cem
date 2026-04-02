from __future__ import annotations

from collections.abc import Mapping
from dataclasses import KW_ONLY, fields, is_dataclass, replace
from typing import Any, Self, cast

import equinox as eqx
import jax.random as jr
from optuna.distributions import BaseDistribution, FloatDistribution, IntDistribution
from tjax import RngStream, create_streams
from tjax.gradient import Adam

from cem.structure.graph import (
    DisGradientTransformation,
    FixedParameter,
    LearnableParameter,
    Model,
    ParameterType,
)
from cem.structure.problem import DataSource, Problem
from cem.structure.solution import (
    ExecutionPacket,
    InferenceResults,
    TrainingResults,
    TrainingSolution,
    infer_episodes,
    train_episodes,
)
from cem.structure.solution.inference import SolutionState

from .hp_field import float_field, int_field

__all__ = ["Solver"]


class Solver[P: Problem](eqx.Module):
    _: KW_ONLY
    title: str = ""
    name: str | None = None
    inference_examples: int = int_field(
        default=0, domain=IntDistribution(1, (1 << 64) - 1, log=True)
    )
    inference_batch_size: int = int_field(
        default=0, domain=IntDistribution(1, (1 << 32) - 1, log=True)
    )
    inference_seed: int = int_field(default=0, domain=IntDistribution(1, (1 << 32) - 1))

    training_examples: int = int_field(
        default=0, domain=IntDistribution(1, (1 << 64) - 1, log=True)
    )
    training_seed: int = int_field(default=0, domain=IntDistribution(1, (1 << 32) - 1))
    training_batch_size: int = int_field(
        default=1, domain=IntDistribution(1, (1 << 32) - 1, log=True)
    )
    parameters_seed: int = int_field(default=0, domain=IntDistribution(1, (1 << 32) - 1))
    learning_rate: float = float_field(
        default=0.002, domain=FloatDistribution(1e-4, 1.0, log=True), optimize=True
    )

    def training_results(self, *, packet: ExecutionPacket) -> TrainingResults:
        solution = self.solution()
        return train_episodes(
            self.name,
            self.training_batch_size,
            solution,
            packet,
            jr.key(self.training_seed),
            self.training_examples,
        )

    def problem(self) -> P:
        raise NotImplementedError

    def create_model(
        self,
        data_source: DataSource,
        problem: Problem,
        *,
        streams: Mapping[str, RngStream],
    ) -> Model:
        raise NotImplementedError

    def solution(self) -> TrainingSolution:
        """Return the solution for this solver.

        This is typically called in the following places:
        * NamesAndPaths.extract,
        * Solver.training_results,
        * Solver.inference_results, and
        * various plotters.

        Therefore, it should always create the same pytree, which means it must not have closures.
        """
        problem = self.problem()
        parameters_key = jr.key(self.parameters_seed)
        data_source = problem.create_data_source()
        keys = {"parameters": parameters_key, "example": jr.key(0)}
        streams = create_streams(keys)
        model = self.create_model(data_source, problem, streams=streams)
        return TrainingSolution.create(problem, model, self.gradient_transformations())

    def gradient_transformations(self) -> DisGradientTransformation:
        return DisGradientTransformation(
            [
                (ParameterType(FixedParameter), None),
                (ParameterType(LearnableParameter), Adam[Model](self.learning_rate)),
            ]
        )

    def inference_results(
        self, solution_state: SolutionState, *, packet: ExecutionPacket
    ) -> InferenceResults:
        solution = self.solution()
        inference_key = jr.key(self.inference_seed)
        return infer_episodes(
            self.name,
            self.inference_batch_size,
            self.inference_examples,
            inference_key,
            solution.inference,
            solution.problem,
            packet,
            solution_state,
        )

    def training_and_inference_result(
        self, *, packet: ExecutionPacket
    ) -> tuple[TrainingResults, InferenceResults]:
        training_results = self.training_results(packet=packet)
        inference_results = self.inference_results(training_results.final_state, packet=packet)
        return training_results, inference_results

    def create_hyperparameters(self) -> dict[str, BaseDistribution]:
        return _create_hyperparameters(self, prefix="")

    def populate_from_hyperparameters(self, hyper: dict[str, Any]) -> Self:
        return _populate_from_hyperparameters(self, hyper, prefix="")


def _create_hyperparameters(x: object, *, prefix: str) -> dict[str, BaseDistribution]:
    assert is_dataclass(x)
    assert not isinstance(x, type)
    hyper = {}
    for f in fields(x):
        y = getattr(x, f.name)
        if not f.metadata.get("optimize", True):
            continue
        if (domain := f.metadata.get("domain", None)) is not None:
            hyper[f"{prefix}{f.name}"] = domain
        elif is_dataclass(y):
            hyper |= _create_hyperparameters(y, prefix=f"{prefix}{f.name}.")
    return hyper


def _populate_from_hyperparameters[T](x: T, hyper: dict[str, Any], *, prefix: str) -> T:
    assert is_dataclass(x)
    assert not isinstance(x, type)
    kwargs = {}
    for f in fields(x):
        y = getattr(x, f.name)
        if not f.metadata.get("optimize", True):
            continue
        if f.metadata.get("domain", None) is not None:
            kwargs[f.name] = hyper[f"{prefix}{f.name}"]
        elif is_dataclass(y):
            kwargs[f.name] = _populate_from_hyperparameters(y, hyper, prefix=f"{prefix}{f.name}.")
    return cast("T", replace(x, **kwargs))
