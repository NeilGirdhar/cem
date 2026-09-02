"""Supervised learning solver: perceptron and phasor variants."""

from collections.abc import Mapping
from dataclasses import KW_ONLY
from enum import Enum
from functools import cache
from typing import Any, Self, override

import equinox as eqx
import jax.numpy as jnp
from efax import Flattener, UnitVarianceNormalNP
from optuna.distributions import CategoricalDistribution, FloatDistribution, IntDistribution
from tjax import JaxRealArray, RngStream, frozendict

from cem.perceptron.mlp import MLP
from cem.perceptron.target_node import PerceptronTargetNode
from cem.phasor.frequency import frequency_base_for_domain_width, geometric_frequencies
from cem.phasor.gated_projection import GatedProjection
from cem.phasor.particle import ObservationParticleState
from cem.phasor.target_node import PhasorTargetNode
from cem.structure.graph import FixedParameter, Model, ModelResult
from cem.structure.problem import DataSource, Problem
from cem.structure.solver import Solver, float_field, hardware_friendly_ints, int_field
from cem.transforms import encode_phasor

from .problem import (
    SupervisedProblem,
    SupervisedProblemState,
    load_hf_tabular_regression,
    load_iris,
)

_PHASOR_DROPOUT_RATE = 0.1


# This has to be a cached function to avoid initializing JAX before training.
@cache
def _scalar_prior() -> UnitVarianceNormalNP:
    return UnitVarianceNormalNP(jnp.zeros(()))


def _y_fields(n_targets: int) -> dict[str, UnitVarianceNormalNP]:
    """One scalar field per target, named 'y' for single-target or 'y_i' for multi."""
    if n_targets == 1:
        return {"y": _scalar_prior()}
    return {f"y_{i}": _scalar_prior() for i in range(n_targets)}


def _y_flat_observed(observation_y: JaxRealArray) -> frozendict[str, JaxRealArray]:
    """Split observation.y (shape (n_targets,)) into per-field flat encodings."""
    n = observation_y.shape[0]
    if n == 1:
        return frozendict({"y": observation_y})
    return frozendict({f"y_{i}": observation_y[i : i + 1] for i in range(n)})


def _y_particle_bounds(y_flat: JaxRealArray) -> dict[str, tuple[JaxRealArray, JaxRealArray]]:
    """Return padded scalar bounds for each standardized target field."""
    lower = jnp.min(y_flat, axis=0)
    upper = jnp.max(y_flat, axis=0)
    width = upper - lower
    padding = jnp.where(width > 0.0, 0.05 * width, jnp.ones_like(width))
    names = ["y"] if y_flat.shape[1] == 1 else [f"y_{i}" for i in range(y_flat.shape[1])]
    return {
        field_name: (
            (lower[i] - padding[i])[jnp.newaxis],
            (upper[i] + padding[i])[jnp.newaxis],
        )
        for i, field_name in enumerate(names)
    }


class DatasetKind(Enum):
    iris = "iris"
    bike_sharing_demand = "bike_sharing_demand"
    elevators = "elevators"
    cpu_activity = "cpu_activity"


class LinkKind(Enum):
    perceptron = "perceptron"
    phasor = "phasor"


_HF_TABULAR_REGRESSION_CONFIGS: dict[DatasetKind, str] = {
    DatasetKind.bike_sharing_demand: "reg_num_Bike_Sharing_Demand",
    DatasetKind.elevators: "reg_num_elevators",
    DatasetKind.cpu_activity: "reg_num_cpu_act",
}

SUPERVISED_MIN_TRAINING_EXAMPLES = 4


class PerceptronSupervisedModel(Model):
    """Supervised model: flat-encoded features → MLP → PerceptronTargetNode."""

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
        in_size = sup.n_features
        out_size = sup.n_targets
        return cls(
            link=MLP.create(in_size, out_size, hidden_features=hidden_size, streams=streams),
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
    """Supervised model: phasor-encoded features → GatedProjection → PhasorTargetNode."""

    link: GatedProjection
    target: PhasorTargetNode
    _x_flattener: FixedParameter[Flattener[Any]]
    _frequencies: FixedParameter[JaxRealArray]

    @override
    def initial_state(self) -> ObservationParticleState:
        """Initialize persistent target-observation particles."""
        return self.target.initial_particle_state()

    @classmethod
    def create(
        cls,
        sup: SupervisedProblem,
        n_frequencies: int,
        hidden_size: int,
        *,
        streams: Mapping[str, RngStream],
    ) -> Self:
        target_domain_width = jnp.max(sup.y_flat) - jnp.min(sup.y_flat)
        freqs = geometric_frequencies(
            n_frequencies, base=frequency_base_for_domain_width(target_domain_width)
        )
        in_size = sup.n_features * n_frequencies
        out_size = sup.n_targets * n_frequencies
        x_flattener, _ = Flattener.flatten(sup.x_prior, mapped_to_plane=True)
        return cls(
            link=GatedProjection.create(
                in_size,
                out_size,
                mid_features=hidden_size,
                dropout_rate=_PHASOR_DROPOUT_RATE,
                streams=streams,
            ),
            target=PhasorTargetNode.create(
                _y_fields(sup.n_targets),
                freqs,
                particle_bounds=_y_particle_bounds(sup.y_flat),
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
        # observation.x: (n_features,), flat UnitVarianceNormalNP encodings.
        x_phasor = encode_phasor(observation.x, self._x_flattener.value, self._frequencies.value)
        # z_hat: (n_targets * n_frequencies,), concatenated target phasors.
        z_hat = self.link.infer(x_phasor, streams=streams, inference=inference)
        # observation.y: (n_targets,), split into scalar fields; z_hat stays concatenated so
        # PhasorTargetNode can split it by field_sizes along the last axis.
        particle_state = state if isinstance(state, ObservationParticleState) else None
        config = self.target.infer(
            _y_flat_observed(observation.y),
            z_hat,
            particle_state,
        )
        return ModelResult(
            loss=config.total_loss(),
            configurations=frozendict({"target": config}),
            state=config.particle_state,
        )


class SupervisedSolver(Solver[SupervisedProblem]):
    """Solver for supervised learning using one perceptron or phasor link.

    Attributes:
        dataset_kind: Which dataset to use (set by the demo, not optimised).
        link_kind: Whether to use a perceptron or phasor link (set by the demo, not optimised).
        hidden_size: Hidden dimension of the perceptron MLP or phasor gated projection.
        n_frequencies: Number of phasor frequencies (only used when ``link_kind == phasor``).
    """

    _: KW_ONLY
    dataset_kind: DatasetKind = eqx.field(static=True)
    link_kind: LinkKind = eqx.field(static=True)
    training_examples: int = int_field(
        default=200,
        domain=IntDistribution(SUPERVISED_MIN_TRAINING_EXAMPLES, 1024, log=True),
        optimize=True,
    )
    learning_rate: float = float_field(
        default=0.01, domain=FloatDistribution(1e-4, 1.0, log=True), optimize=True
    )
    hidden_size: int = int_field(
        default=64,
        domain=CategoricalDistribution(hardware_friendly_ints(4, 128)),
        optimize=True,
    )
    n_frequencies: int = int_field(
        default=10,
        domain=CategoricalDistribution(hardware_friendly_ints(2, 16)),
        optimize=True,
        condition=lambda solver: solver.link_kind == LinkKind.phasor,  # type: ignore
    )

    def compute_proxy(self) -> JaxRealArray:
        """Proxy for supervised demo runtime cost.

        Uses the tunable dimensions that dominate local training cost:
        ``training_examples * training_batch_size * hidden_size**2``.
        """
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
        assert isinstance(problem, SupervisedProblem)
        if self.link_kind == LinkKind.perceptron:
            return PerceptronSupervisedModel.create(problem, self.hidden_size, streams=streams)
        return PhasorSupervisedModel.create(
            problem,
            self.n_frequencies,
            self.hidden_size,
            streams=streams,
        )
