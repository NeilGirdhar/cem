"""Supervised learning solver and perceptron baseline."""

from collections.abc import Mapping
from dataclasses import KW_ONLY
from enum import Enum
from functools import cache
from typing import Self, override

import equinox as eqx
import jax.numpy as jnp
from efax import UnitVarianceNormalNP
from optuna.distributions import CategoricalDistribution, FloatDistribution, IntDistribution
from tjax import JaxRealArray, RngStream, frozendict

from cem.perceptron.mlp import MLP
from cem.perceptron.target_node import PerceptronTargetNode
from cem.phasor.gated_projection import GatedProjection
from cem.phasor.phase_activated_projection import PhaseActivatedProjection
from cem.phasor.target_node import PhasorTargetNode
from cem.structure.graph import Model, ModelResult
from cem.structure.problem import DataSource, Problem
from cem.structure.solver import Solver, float_field, hardware_friendly_ints, int_field
from cem.transforms import encode_observation_phasors

from .problem import (
    SupervisedProblem,
    SupervisedProblemState,
    load_hf_tabular_regression,
    load_iris,
)

_SUPERVISED_HIDDEN_SIZES = tuple(
    sorted({*hardware_friendly_ints(4, 256), 20, 27, 73, 85, 98, 128, 139, 220})
)


# This has to be a cached function to avoid initializing JAX before training.
@cache
def _scalar_prior() -> UnitVarianceNormalNP:
    return UnitVarianceNormalNP(jnp.zeros(()))


def _y_fields(n_targets: int) -> dict[str, UnitVarianceNormalNP]:
    """Return one scalar field per target."""
    if n_targets == 1:
        return {"y": _scalar_prior()}
    return {f"y_{i}": _scalar_prior() for i in range(n_targets)}


def _y_flat_observed(observation_y: JaxRealArray) -> frozendict[str, JaxRealArray]:
    """Split an observed target vector into scalar fields."""
    n = observation_y.shape[0]
    if n == 1:
        return frozendict({"y": observation_y})
    return frozendict({f"y_{i}": observation_y[i : i + 1] for i in range(n)})


class DatasetKind(Enum):
    iris = "iris"
    bike_sharing_demand = "bike_sharing_demand"
    elevators = "elevators"
    cpu_activity = "cpu_activity"


class LinkKind(Enum):
    perceptron = "perceptron"
    phasor = "phasor"
    phase_activated = "phase_activated"


_HF_TABULAR_REGRESSION_CONFIGS: dict[DatasetKind, str] = {
    DatasetKind.bike_sharing_demand: "reg_num_Bike_Sharing_Demand",
    DatasetKind.elevators: "reg_num_elevators",
    DatasetKind.cpu_activity: "reg_num_cpu_act",
}

SUPERVISED_MIN_TRAINING_EXAMPLES = 4


class PerceptronSupervisedModel(Model):
    """Supervised model: flat-encoded features to MLP to target node."""

    link: MLP
    target: PerceptronTargetNode

    @classmethod
    def create(
        cls,
        sup: SupervisedProblem,
        hidden_size: int,
        *,
        streams: Mapping[str, RngStream],
    ) -> Self:
        return cls(
            link=MLP.create(
                sup.n_features,
                sup.n_targets,
                hidden_features=hidden_size,
                streams=streams,
            ),
            target=PerceptronTargetNode.create(_y_fields(sup.n_targets)),
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
        config = self.target.infer(_y_flat_observed(observation.y), y_hat)
        return ModelResult(
            loss=config.total_loss(),
            configurations=frozendict({"target": config}),
            state=state,
        )


class PhasorSupervisedModel(Model):
    """Supervised model with one observation phasor per scalar feature."""

    link: GatedProjection | PhaseActivatedProjection
    target: PhasorTargetNode

    @classmethod
    def create(
        cls,
        sup: SupervisedProblem,
        hidden_size: int,
        *,
        phase_activation: bool = False,
        streams: Mapping[str, RngStream],
    ) -> Self:
        projection = PhaseActivatedProjection if phase_activation else GatedProjection
        return cls(
            link=projection.create(
                sup.n_features,
                sup.n_targets,
                mid_features=hidden_size,
                streams=streams,
            ),
            target=PhasorTargetNode.create(_y_fields(sup.n_targets)),
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
        del state
        assert isinstance(observation, SupervisedProblemState)
        x_phasors = encode_observation_phasors(
            jnp.ones_like(observation.x),
            observation.x,
        )
        prediction = self.link.infer(x_phasors, streams=streams, inference=inference)
        target = self.target.infer(_y_flat_observed(observation.y), prediction)
        return ModelResult(
            loss=target.total_loss(),
            configurations=frozendict({"target": target}),
            state=None,
        )


class SupervisedSolver(Solver[SupervisedProblem]):
    """Solver for perceptron and one-phasor supervised models."""

    _: KW_ONLY
    dataset_kind: DatasetKind = eqx.field(static=True)
    link_kind: LinkKind = eqx.field(static=True)
    inference_examples: int = int_field(
        default=1,
        domain=IntDistribution(1, 1),
        optimize=False,
    )
    training_examples: int = int_field(
        default=200,
        domain=IntDistribution(SUPERVISED_MIN_TRAINING_EXAMPLES, 1024, log=True),
        optimize=True,
    )
    learning_rate: float = float_field(
        default=0.01,
        domain=FloatDistribution(1e-4, 1.0, log=True),
        optimize=True,
    )
    hidden_size: int = int_field(
        default=64,
        domain=CategoricalDistribution(_SUPERVISED_HIDDEN_SIZES),
        optimize=True,
    )

    def compute_proxy(self) -> JaxRealArray:
        """Approximate the local training cost."""
        return jnp.asarray(self.training_examples * self.hidden_size**2)

    @override
    def problem(self) -> SupervisedProblem:
        if self.dataset_kind == DatasetKind.iris:
            return load_iris()
        if self.dataset_kind in _HF_TABULAR_REGRESSION_CONFIGS:
            return load_hf_tabular_regression(_HF_TABULAR_REGRESSION_CONFIGS[self.dataset_kind])
        msg = f"Unsupported supervised dataset kind: {self.dataset_kind}"
        raise ValueError(msg)

    @override
    def create_model(
        self,
        data_source: DataSource,
        problem: Problem,
        *,
        streams: Mapping[str, RngStream],
    ) -> Model:
        del data_source
        assert isinstance(problem, SupervisedProblem)
        if self.link_kind == LinkKind.perceptron:
            return PerceptronSupervisedModel.create(problem, self.hidden_size, streams=streams)
        return PhasorSupervisedModel.create(
            problem,
            self.hidden_size,
            phase_activation=self.link_kind == LinkKind.phase_activated,
            streams=streams,
        )
