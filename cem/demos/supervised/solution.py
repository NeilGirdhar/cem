"""Supervised learning solver: perceptron and phasor variants."""  # noqa: INP001

from __future__ import annotations

from collections.abc import Mapping
from enum import Enum
from typing import Any, Self, override

from efax import Flattener
from optuna.distributions import IntDistribution
from tjax import JaxRealArray, RngStream, frozendict

from cem import perceptron, phasor
from cem.perceptron.target_node import PerceptronTargetNode
from cem.phasor.frequency import geometric_frequencies
from cem.phasor.message import PhasorMessage
from cem.phasor.target_node import PhasorTargetNode
from cem.structure.graph import FixedParameter, Model, ModelResult
from cem.structure.problem import DataSource, Problem
from cem.structure.solver import Solver, int_field

from .problem import (
    SupervisedProblem,
    SupervisedProblemState,
    load_iris,
    load_synthetic_regression,
)


class DatasetKind(Enum):
    iris = "iris"
    synthetic_regression = "synthetic_regression"


class LinkKind(Enum):
    perceptron = "perceptron"
    phasor = "phasor"


class PerceptronSupervisedModel(Model):
    """Supervised model: flat-encoded features → Nonlinear → PerceptronTargetNode."""

    link: perceptron.Nonlinear
    target: PerceptronTargetNode

    @classmethod
    def create(
        cls,
        sup: SupervisedProblem,
        hidden_size: int,
        *,
        streams: Mapping[str, RngStream],
    ) -> Self:
        in_size = sup.n_features
        out_size = sup.n_targets
        return cls(
            link=perceptron.Nonlinear.create(
                in_size, out_size, mid_features=hidden_size, streams=streams
            ),
            target=PerceptronTargetNode.create({"y": sup.y_prior}),
        )

    @override
    def infer(
        self,
        observation: object,
        state: object,
        *,
        streams: Mapping[str, RngStream],
        inference: bool,
    ) -> ModelResult:
        assert isinstance(observation, SupervisedProblemState)
        y_hat = self.link.infer(observation.x, streams=streams, inference=inference)
        config = self.target.infer(frozendict({"y": observation.y}), y_hat)
        return ModelResult(
            loss=config.total_loss(),
            configurations=frozendict({"target": config}),
            state=state,
        )


class PhasorSupervisedModel(Model):
    """Supervised model: phasor-encoded features → Nonlinear → PhasorTargetNode."""

    link: phasor.Nonlinear
    target: PhasorTargetNode
    _x_flattener: FixedParameter[Flattener[Any]]
    _frequencies: FixedParameter[JaxRealArray]

    @classmethod
    def create(
        cls,
        sup: SupervisedProblem,
        n_frequencies: int,
        hidden_size: int,
        *,
        streams: Mapping[str, RngStream],
    ) -> Self:
        freqs = geometric_frequencies(n_frequencies)
        in_size = sup.n_features * n_frequencies
        out_size = sup.n_targets * n_frequencies
        num_groups = max(1, out_size // 8)
        x_flattener, _ = Flattener.flatten(sup.x_prior, mapped_to_plane=True)
        return cls(
            link=phasor.Nonlinear.create(
                in_size, out_size, num_groups, mid_features=hidden_size, streams=streams
            ),
            target=PhasorTargetNode.create({"y": sup.y_prior}, freqs),
            _x_flattener=FixedParameter(x_flattener),
            _frequencies=FixedParameter(freqs),
        )

    @override
    def infer(
        self,
        observation: object,
        state: object,
        *,
        streams: Mapping[str, RngStream],
        inference: bool,
    ) -> ModelResult:
        assert isinstance(observation, SupervisedProblemState)
        x_dist = self._x_flattener.value.unflatten(observation.x)
        x_phasor = PhasorMessage.from_distribution(x_dist, self._frequencies.value)
        z_hat = self.link.infer(x_phasor, streams=streams, inference=inference)
        config = self.target.infer(frozendict({"y": observation.y}), z_hat)
        return ModelResult(
            loss=config.total_loss(),
            configurations=frozendict({"target": config}),
            state=state,
        )


class SupervisedSolver(Solver[SupervisedProblem]):
    """Solver for supervised learning using a single nonlinear link.

    Attributes:
        dataset_kind: Which dataset to use.
        link_kind: Whether to use a perceptron or phasor link.
        hidden_size: Hidden dimension of the nonlinear layer.
        n_frequencies: Number of phasor frequencies (only used when ``link_kind == phasor``).
    """

    dataset_kind: DatasetKind = DatasetKind.iris
    link_kind: LinkKind = LinkKind.perceptron
    hidden_size: int = int_field(default=64, domain=IntDistribution(4, 128))
    n_frequencies: int = int_field(default=8, domain=IntDistribution(2, 16))

    @override
    def problem(self) -> SupervisedProblem:
        if self.dataset_kind == DatasetKind.iris:
            return load_iris()
        return load_synthetic_regression()

    @override
    def create_model(
        self,
        data_source: DataSource,
        problem: Problem,
        *,
        streams: Mapping[str, RngStream],
    ) -> Model:
        assert isinstance(problem, SupervisedProblem)
        if self.link_kind == LinkKind.perceptron:
            return PerceptronSupervisedModel.create(problem, self.hidden_size, streams=streams)
        return PhasorSupervisedModel.create(
            problem, self.n_frequencies, self.hidden_size, streams=streams
        )
