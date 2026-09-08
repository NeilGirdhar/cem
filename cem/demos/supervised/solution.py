"""Supervised learning solver and perceptron baseline."""

from collections.abc import Mapping
from dataclasses import KW_ONLY
from enum import Enum
from functools import cache
from typing import Self, override

import equinox as eqx
import jax.numpy as jnp
from efax import UnitVarianceNormalNP
from jax.lax import stop_gradient
from optuna.distributions import CategoricalDistribution, FloatDistribution, IntDistribution
from tjax import JaxRealArray, RngStream, copy_cotangent, frozendict
from tjax.gradient import Adam

from cem.experimental.phasor.gated_projection import GatedProjection
from cem.experimental.phasor.mobius_summation import MobiusPresenceRule
from cem.experimental.phasor.phase_activated_projection import PhaseActivatedProjection
from cem.experimental.phasor.target_node import PhasorTargetNode
from cem.perceptron.mlp import MLP
from cem.perceptron.target_node import PerceptronTargetNode
from cem.structure.graph import (
    DisGradientTransformation,
    FixedParameter,
    LearnableParameter,
    MetaParameter,
    Model,
    ModelResult,
    ParameterType,
    count_real_learnable_parameters,
)
from cem.structure.problem import DataSource, Problem
from cem.structure.solver import Solver, bool_field, float_field, hardware_friendly_ints, int_field
from cem.transforms import ArctangentPhaseMap

from .problem import (
    SupervisedProblem,
    SupervisedProblemState,
    load_hf_tabular_regression,
    load_iris,
)

_SUPERVISED_HIDDEN_SIZES = tuple(
    sorted({*hardware_friendly_ints(2, 256), 20, 27, 73, 85, 98, 128, 139, 220})
)
_TWO_LAYER_DEPTH = 2


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
    gated_two_layer = "gated_two_layer"
    phase_activated_two_layer = "phase_activated_two_layer"
    phase_activated_no_parallel = "phase_activated_no_parallel"
    phase_activated_participation_only = "phase_activated_participation_only"


_PHASE_ACTIVATED_LINK_KINDS = frozenset(
    {
        LinkKind.phase_activated,
        LinkKind.phase_activated_two_layer,
        LinkKind.phase_activated_no_parallel,
        LinkKind.phase_activated_participation_only,
    }
)

_TWO_LAYER_LINK_KINDS = frozenset(
    {
        LinkKind.gated_two_layer,
        LinkKind.phase_activated_two_layer,
    }
)

_MOBIUS_PRESENCE_RULES = {
    LinkKind.phasor: MobiusPresenceRule.parallel,
    LinkKind.phase_activated: MobiusPresenceRule.parallel,
    LinkKind.gated_two_layer: MobiusPresenceRule.participation,
    LinkKind.phase_activated_two_layer: MobiusPresenceRule.parallel,
    LinkKind.phase_activated_no_parallel: MobiusPresenceRule.participation,
    LinkKind.phase_activated_participation_only: MobiusPresenceRule.participation_only,
}


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

    input_phase_map: ArctangentPhaseMap
    links: tuple[GatedProjection | PhaseActivatedProjection, ...]
    target: PhasorTargetNode
    phase_map_fisher_weight: float = eqx.field(static=True)
    phase_map_adversarial: bool = eqx.field(static=True)
    phase_map_translation: bool = eqx.field(static=True)

    @classmethod
    def create(  # ruff: ignore[too-many-arguments]
        cls,
        sup: SupervisedProblem,
        hidden_size: int,
        *,
        phase_activation: bool = False,
        depth: int = 1,
        mobius_presence_rule: MobiusPresenceRule = MobiusPresenceRule.parallel,
        phase_map_fisher_weight: float = 0.0,
        phase_map_adversarial: bool = True,
        phase_map_translation: bool = True,
        streams: Mapping[str, RngStream],
    ) -> Self:
        if depth not in {1, _TWO_LAYER_DEPTH}:
            msg = f"depth must be 1 or 2, got {depth}"
            raise ValueError(msg)
        projection = PhaseActivatedProjection if phase_activation else GatedProjection
        if depth == 1:
            layer_shapes = ((sup.n_features, sup.n_targets),)
        else:
            layer_shapes = (
                (sup.n_features, hidden_size),
                (hidden_size, sup.n_targets),
            )
        input_phase_map = ArctangentPhaseMap.create_learned(sup.n_features)
        if not phase_map_translation:
            input_phase_map = eqx.tree_at(
                lambda phase_map: phase_map.centres,
                input_phase_map,
                FixedParameter(jnp.zeros(sup.n_features)),
            )
        return cls(
            input_phase_map=input_phase_map,
            links=tuple(
                projection.create(
                    in_features,
                    out_features,
                    mid_features=hidden_size,
                    mobius_presence_rule=mobius_presence_rule,
                    streams=streams,
                )
                for in_features, out_features in layer_shapes
            ),
            target=PhasorTargetNode.create(_y_fields(sup.n_targets)),
            phase_map_fisher_weight=phase_map_fisher_weight,
            phase_map_adversarial=phase_map_adversarial,
            phase_map_translation=phase_map_translation,
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
        if self.phase_map_adversarial:
            x_phasors = self.input_phase_map.encode_with_reversed_phase_gradient(
                jnp.ones_like(observation.x), observation.x
            )
        else:
            x_phasors = self.input_phase_map.encode(jnp.ones_like(observation.x), observation.x)
        prediction = x_phasors
        mobius_diagnostics = []
        for link in self.links:
            prediction, diagnostics = link.infer_with_diagnostics(
                prediction,
                streams=streams,
                inference=inference,
            )
            mobius_diagnostics.append(diagnostics)
        target = self.target.infer(_y_flat_observed(observation.y), prediction)
        target_loss = target.total_loss()
        fisher_loss = self.input_phase_map.fisher_equalization_loss(observation.x)
        configurations = {"mobius": mobius_diagnostics[-1], "target": target}
        if len(mobius_diagnostics) > 1:
            configurations["mobius_input"] = mobius_diagnostics[0]
        return ModelResult(
            loss=copy_cotangent(
                stop_gradient(target_loss),
                target_loss + self.phase_map_fisher_weight * fisher_loss,
            ),
            configurations=frozendict(configurations),
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
    phase_map_learning_rate_scale: float = float_field(
        default=1.0,
        domain=FloatDistribution(1e-4, 1.0, log=True),
        optimize=False,
    )
    phase_map_fisher_weight: float = float_field(
        default=0.0,
        domain=FloatDistribution(0.0, 1.0),
        optimize=False,
    )
    phase_map_adversarial: bool = bool_field(default=True, optimize=False)
    phase_map_translation: bool = bool_field(default=True, optimize=False)

    def gradient_transformations(self) -> DisGradientTransformation:
        """Use a slower optimizer for adaptive input phase-map scales."""
        return DisGradientTransformation(
            [
                (ParameterType(FixedParameter), None),
                (
                    ParameterType(MetaParameter),
                    Adam[Model](self.phase_map_learning_rate_scale * self.learning_rate),
                ),
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
            phase_map_translation=self.phase_map_translation,
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
            return PerceptronSupervisedModel.create(problem, self.hidden_size, streams=streams)
        return PhasorSupervisedModel.create(
            problem,
            self.hidden_size,
            phase_activation=self.link_kind in _PHASE_ACTIVATED_LINK_KINDS,
            depth=_TWO_LAYER_DEPTH if self.link_kind in _TWO_LAYER_LINK_KINDS else 1,
            mobius_presence_rule=_MOBIUS_PRESENCE_RULES[self.link_kind],
            phase_map_fisher_weight=self.phase_map_fisher_weight,
            phase_map_adversarial=self.phase_map_adversarial,
            phase_map_translation=self.phase_map_translation,
            streams=streams,
        )


@cache
def _supervised_parameter_count(
    dataset_kind: DatasetKind,
    link_kind: LinkKind,
    hidden_size: int,
    *,
    phase_map_translation: bool,
) -> int:
    solver = SupervisedSolver(
        dataset_kind=dataset_kind,
        link_kind=link_kind,
        hidden_size=hidden_size,
        phase_map_translation=phase_map_translation,
    )
    learnable_model = solver.solution().solution_state.dis_learnable_parameters.assembled()
    return count_real_learnable_parameters(learnable_model)
