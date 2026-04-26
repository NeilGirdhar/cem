"""Supervised learning solver: perceptron and phasor variants."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import KW_ONLY
from enum import Enum
from typing import Any, Self, override

import equinox as eqx
from efax import Flattener
from optuna.distributions import BaseDistribution, FloatDistribution, IntDistribution
from tjax import JaxRealArray, RngStream, frozendict

from cem import perceptron, phasor
from cem.perceptron.target_node import PerceptronTargetNode
from cem.phasor.frequency import geometric_frequencies
from cem.phasor.message import PhasorMessage
from cem.phasor.target_node import PhasorTargetNode
from cem.structure.graph import FixedParameter, Model, ModelResult
from cem.structure.problem import DataSource, Problem
from cem.structure.solver import Solver, float_field, int_field

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
        use_spectral_loss: bool = False,
        streams: Mapping[str, RngStream],
    ) -> Self:
        freqs = geometric_frequencies(n_frequencies, base=1 / 4)
        in_size = sup.n_features * n_frequencies
        out_size = sup.n_targets * n_frequencies
        num_groups = max(1, out_size // 8)
        x_flattener, _ = Flattener.flatten(sup.x_prior, mapped_to_plane=True)
        return cls(
            link=phasor.Nonlinear.create(
                in_size, out_size, num_groups, mid_features=hidden_size, streams=streams
            ),
            target=PhasorTargetNode.create(
                {"y": sup.y_prior}, freqs, use_spectral_loss=use_spectral_loss
            ),
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
        n_frequencies = self._frequencies.value.shape[0]
        x_dist = self._x_flattener.value.unflatten(observation.x, return_vector=True)
        x_phasor = PhasorMessage.from_distribution(x_dist, self._frequencies.value, raveled=True)
        z_hat = self.link.infer(x_phasor, streams=streams, inference=inference)
        z_hat_2d = z_hat.split_frequencies(n_frequencies)
        config = self.target.infer(frozendict({"y": observation.y}), z_hat_2d)
        return ModelResult(
            loss=config.total_loss(),
            configurations=frozendict({"target": config}),
            state=state,
        )


class SupervisedSolver(Solver[SupervisedProblem]):
    """Solver for supervised learning using a single nonlinear link.

    Attributes:
        dataset_kind: Which dataset to use (set by the demo, not optimised).
        link_kind: Whether to use a perceptron or phasor link (set by the demo, not optimised).
        hidden_size: Hidden dimension of the nonlinear layer.
        n_frequencies: Number of phasor frequencies (only used when ``link_kind == phasor``).
    """

    _: KW_ONLY
    dataset_kind: DatasetKind = eqx.field(static=True)
    link_kind: LinkKind = eqx.field(static=True)
    use_spectral_loss: bool = eqx.field(default=False, static=True)
    training_examples: int = int_field(
        default=1000, domain=IntDistribution(1, 1 << 16, log=True), optimize=True
    )
    training_batch_size: int = int_field(
        default=32, domain=IntDistribution(1, 1 << 10, log=True), optimize=True
    )
    inference_examples: int = int_field(
        default=200, domain=IntDistribution(1, 1 << 12, log=True), optimize=True
    )
    inference_batch_size: int = int_field(
        default=32, domain=IntDistribution(1, 1 << 10, log=True), optimize=True
    )
    learning_rate: float = float_field(
        default=0.01, domain=FloatDistribution(1e-4, 1.0, log=True), optimize=True
    )
    hidden_size: int = int_field(default=64, domain=IntDistribution(4, 128), optimize=True)
    n_frequencies: int = int_field(default=10, domain=IntDistribution(2, 16), optimize=True)

    @override
    def create_hyperparameters(self) -> dict[str, BaseDistribution]:
        hyper = super().create_hyperparameters()
        if self.link_kind == LinkKind.perceptron:
            del hyper["n_frequencies"]
        return hyper

    @override
    def populate_from_hyperparameters(self, hyper: dict[str, Any]) -> Self:
        if self.link_kind == LinkKind.perceptron:
            hyper = {**hyper, "n_frequencies": self.n_frequencies}
        return super().populate_from_hyperparameters(hyper)

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
            problem,
            self.n_frequencies,
            self.hidden_size,
            use_spectral_loss=self.use_spectral_loss,
            streams=streams,
        )
