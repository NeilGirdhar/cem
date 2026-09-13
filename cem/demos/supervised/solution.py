"""Supervised learning solver and perceptron baseline."""

from collections.abc import Mapping
from dataclasses import KW_ONLY
from enum import Enum
from functools import cache
from typing import Self, override

import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
from efax import NormalNP, UnitVarianceNormalNP
from optuna.distributions import CategoricalDistribution, FloatDistribution, IntDistribution
from tjax import JaxRealArray, RngStream, frozendict
from tjax.gradient import Adam

from cem.npn.gaussian import GaussianNPN
from cem.perceptron.mlp import MLP
from cem.perceptron.target_node import PerceptronTargetNode
from cem.structure.graph import (
    DisGradientTransformation,
    FixedParameter,
    LearnableParameter,
    Model,
    ModelResult,
    ParameterType,
    count_real_learnable_parameters,
)
from cem.structure.graph.node import TargetConfiguration
from cem.structure.problem import DataSource, Problem
from cem.structure.solver import Solver, bool_field, float_field, hardware_friendly_ints, int_field

from .problem import (
    SupervisedProblem,
    SupervisedProblemState,
    load_hf_tabular_regression,
    load_iris,
)

_SUPERVISED_HIDDEN_SIZES = tuple(
    sorted({*hardware_friendly_ints(2, 256), 20, 27, 73, 85, 98, 128, 139, 220})
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
    natural_parameter = "natural_parameter"


_HF_TABULAR_REGRESSION_CONFIGS: dict[DatasetKind, str] = {
    DatasetKind.bike_sharing_demand: "reg_num_Bike_Sharing_Demand",
    DatasetKind.elevators: "reg_num_elevators",
    DatasetKind.cpu_activity: "reg_num_cpu_act",
}

SUPERVISED_MIN_TRAINING_EXAMPLES = 4


def _sample_input_presence(
    values: JaxRealArray,
    missing_probability: float,
    *,
    streams: Mapping[str, RngStream],
) -> JaxRealArray:
    if missing_probability == 0:
        return jnp.ones_like(values)
    return jr.bernoulli(
        streams["inference"].key(),
        1 - missing_probability,
        shape=values.shape,
    ).astype(values.dtype)


class PerceptronSupervisedModel(Model):
    """Supervised model: flat-encoded features to MLP to target node."""

    link: MLP
    target: PerceptronTargetNode
    missing_probability: float = eqx.field(static=True)
    random_missing_values: bool = eqx.field(static=True)
    include_missingness_mask: bool = eqx.field(static=True)

    @classmethod
    def create(
        cls,
        sup: SupervisedProblem,
        hidden_size: int,
        *,
        missing_probability: float = 0.0,
        random_missing_values: bool = True,
        include_missingness_mask: bool = False,
        streams: Mapping[str, RngStream],
    ) -> Self:
        return cls(
            link=MLP.create(
                sup.n_features * (2 if include_missingness_mask else 1),
                sup.n_targets,
                hidden_features=hidden_size,
                streams=streams,
            ),
            target=PerceptronTargetNode.create(_y_fields(sup.n_targets)),
            missing_probability=missing_probability,
            random_missing_values=random_missing_values,
            include_missingness_mask=include_missingness_mask,
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
        presence = _sample_input_presence(
            observation.x,
            self.missing_probability,
            streams=streams,
        )
        if self.random_missing_values and self.missing_probability > 0:
            replacement = jr.normal(
                streams["inference"].key(),
                shape=observation.x.shape,
            )
        else:
            replacement = jnp.zeros_like(observation.x)
        x = jnp.where(presence > 0, observation.x, replacement)
        if self.include_missingness_mask:
            x = jnp.concatenate((x, presence), axis=-1)
        y_hat = self.link.infer(x, streams=streams, inference=inference)
        config = self.target.infer(_y_flat_observed(observation.y), y_hat)
        return ModelResult(
            loss=config.total_loss(),
            configurations=frozendict({"target": config}),
            state=state,
        )


class GaussianNPNTargetConfiguration(TargetConfiguration):
    """Gaussian NPN targets, keyed by scalar field name."""


class GaussianNPNTarget(eqx.Module):
    """Compare Gaussian NPN predictions with unit-presence observations."""

    field_names: tuple[str, ...] = eqx.field(static=True)

    @classmethod
    def create(cls, n_targets: int) -> Self:
        return cls(field_names=tuple(_y_fields(n_targets)))

    def infer(
        self,
        flat_observed: frozendict[str, JaxRealArray],
        prediction: NormalNP,
    ) -> GaussianNPNTargetConfiguration:
        losses = {}
        observed_distributions = {}
        predicted_distributions = {}
        for index, field_name in enumerate(self.field_names):
            observed_mean = flat_observed[field_name]
            observed = NormalNP(
                mean_times_precision=observed_mean,
                negative_half_precision=-0.5 * jnp.ones_like(observed_mean),
            )
            predicted = NormalNP(
                mean_times_precision=prediction.mean_times_precision[..., index : index + 1],
                negative_half_precision=prediction.negative_half_precision[..., index : index + 1],
            )
            observed_exp = observed.to_exp()
            predicted_exp = predicted.to_exp()
            losses[field_name] = observed_exp.kl_divergence(predicted, self_nat=observed)
            observed_distributions[field_name] = observed_exp
            predicted_distributions[field_name] = predicted_exp
        return GaussianNPNTargetConfiguration(
            loss=frozendict(losses),
            observed_distributions=frozendict(observed_distributions),
            predicted_distributions=frozendict(predicted_distributions),
        )


class GaussianNPNSupervisedModel(Model):
    """Supervised Gaussian natural-parameter network."""

    link: GaussianNPN
    target: GaussianNPNTarget
    missing_probability: float = eqx.field(static=True)

    @classmethod
    def create(
        cls,
        sup: SupervisedProblem,
        hidden_size: int,
        *,
        missing_probability: float = 0.0,
        streams: Mapping[str, RngStream],
    ) -> Self:
        return cls(
            link=GaussianNPN.create(
                sup.n_features,
                sup.n_targets,
                hidden_features=hidden_size,
                streams=streams,
            ),
            target=GaussianNPNTarget.create(sup.n_targets),
            missing_probability=missing_probability,
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
        del inference
        assert isinstance(observation, SupervisedProblemState)
        presence = _sample_input_presence(
            observation.x,
            self.missing_probability,
            streams=streams,
        )
        prediction = self.link.infer(observation.x, presence)
        config = self.target.infer(_y_flat_observed(observation.y), prediction)
        return ModelResult(
            loss=config.total_loss(),
            configurations=frozendict({"target": config}),
            state=state,
        )


class SupervisedSolver(Solver[SupervisedProblem]):
    """Solver for perceptron and Gaussian NPN supervised models."""

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
    missing_probability: float = float_field(
        default=0.0,
        domain=FloatDistribution(0.0, 0.95),
        optimize=False,
    )
    random_missing_values: bool = bool_field(default=True, optimize=False)
    include_missingness_mask: bool = bool_field(default=False, optimize=False)

    def gradient_transformations(self) -> DisGradientTransformation:
        return DisGradientTransformation(
            [
                (ParameterType(FixedParameter), None),
                (ParameterType(LearnableParameter), Adam[Model](self.learning_rate)),
            ]
        )

    def compute_proxy(self) -> JaxRealArray:
        """Return the number of trainable real scalars updated during training."""
        return jnp.asarray(self.training_examples * self.parameter_count())

    def parameter_count(self) -> int:
        """Return the model's trainable real scalar degrees of freedom."""
        return _supervised_parameter_count(
            self.dataset_kind,
            self.link_kind,
            self.hidden_size,
            include_missingness_mask=self.include_missingness_mask,
        )

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
            return PerceptronSupervisedModel.create(
                problem,
                self.hidden_size,
                missing_probability=self.missing_probability,
                random_missing_values=self.random_missing_values,
                include_missingness_mask=self.include_missingness_mask,
                streams=streams,
            )
        return GaussianNPNSupervisedModel.create(
            problem,
            self.hidden_size,
            missing_probability=self.missing_probability,
            streams=streams,
        )


@cache
def _supervised_parameter_count(
    dataset_kind: DatasetKind,
    link_kind: LinkKind,
    hidden_size: int,
    *,
    include_missingness_mask: bool,
) -> int:
    solver = SupervisedSolver(
        dataset_kind=dataset_kind,
        link_kind=link_kind,
        hidden_size=hidden_size,
        include_missingness_mask=include_missingness_mask,
    )
    learnable_model = solver.solution().solution_state.dis_learnable_parameters.assembled()
    return count_real_learnable_parameters(learnable_model)
