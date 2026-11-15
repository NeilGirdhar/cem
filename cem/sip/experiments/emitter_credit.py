"""Emitter and credit experiments: whether credit moves observations along their causal effects.

See `sip.md`, tests 9–13.
"""

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import KW_ONLY, asdict, dataclass, replace
from typing import Any, Self, override

import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from efax import NormalNP
from jax import tree
from optuna.distributions import IntDistribution
from tjax import JaxRealArray, RngStream, create_streams
from tjax.dataclasses import field

from cem.framework.graph import Model, ModelResult, NodeConfiguration
from cem.framework.plotter import Run, Variant
from cem.framework.plotter.plotter import PlottedSeries, Plotter
from cem.framework.problem import DataSource, Problem
from cem.framework.solution import (
    ExecutionPacket,
    Inference,
    InferenceResult,
    Telemetries,
    Telemetry,
    TrainingResults,
)
from cem.framework.solver import Solver, int_field
from cem.npn.gaussian import GaussianNeuralProjection
from cem.sip.credit import Credit
from cem.sip.emitter import Emitter, ObservationCoordinates
from cem.sip.settings import CreditSettings

from .common import SEED_COUNT, SeededExperiment, bar_chart, seed_runs
from .datasets import DirectCreditProblem, DirectCreditState
from .scenarios import DirectCreditScenario

_FEATURE_NODE = "feature"


class DirectCreditConfiguration(NodeConfiguration):
    """One emission's gain, value, and goal, and parameter vectors for the observation map.

    Each vector flattens the map's trainable parameters in leaf order. The mean credit is the
    goal times the mean eligibility trace, the update direction that the goal contributes. The
    shrinkage credit is the update that mean shrinkage contributes. The observational credit pairs
    the goal with the emitted value in place of the instrument. The Jacobian is the derivative of
    the raw mean with respect to the parameters.
    """

    gain: JaxRealArray  # scalar
    value: JaxRealArray  # scalar
    goal: JaxRealArray  # scalar
    mean_credit: JaxRealArray  # (p,)
    shrinkage_credit: JaxRealArray  # (p,)
    observational_credit: JaxRealArray  # (p,)
    mean_jacobian: JaxRealArray  # (p,)


def _flatten(parameters: eqx.Module) -> JaxRealArray:
    return jnp.concatenate([jnp.ravel(leaf) for leaf in tree.leaves(parameters)])


def _with_output_weight(
    projection: GaussianNeuralProjection, weight: float
) -> GaussianNeuralProjection:
    """Return the one-layer projection with its weight set, keeping its other initial values."""
    (layer,) = projection.layers
    return eqx.tree_at(
        lambda p: p.layers[0].weight.value, projection, jnp.full_like(layer.weight.value, weight)
    )


class DirectCreditModel(Model):
    """A controllable emitter whose observation reads the confounder, at a fixed operating point.

    The observation map is affine in the observation content, with the given weight and a zero
    bias. The feature has no parents, so the instrument map reads an absent instrument. Its weight
    is zero, so the inherited instrument is the map's initial bias, a standard normal. Each
    emission therefore injects unit-variance noise with a centred, unit-precision instrument.

    The model stores the emitter's maps and assembles the emitter for each example, so every
    example starts with empty traces. The mean trace then holds only the current emission's
    contribution, and the goal credits that emission alone. The model measures credit at fixed
    parameters and never trains, so it reports no loss.
    """

    observation_map: GaussianNeuralProjection
    instrument_map: GaussianNeuralProjection
    goal_baseline_map: GaussianNeuralProjection
    scenario: DirectCreditScenario = eqx.field(static=True)
    credit_settings: CreditSettings = eqx.field(static=True)

    @classmethod
    def create(
        cls,
        scenario: DirectCreditScenario,
        *,
        observation_weight: float,
        credit_settings: CreditSettings,
        streams: Mapping[str, RngStream],
    ) -> Self:
        def projection() -> GaussianNeuralProjection:
            return GaussianNeuralProjection.create(in_features=1, out_features=1, streams=streams)

        return cls(
            observation_map=_with_output_weight(projection(), observation_weight),
            instrument_map=_with_output_weight(projection(), 0.0),
            goal_baseline_map=projection(),
            scenario=scenario,
            credit_settings=credit_settings,
        )

    def emitter(self) -> Emitter:
        """Assemble a controllable emitter from the maps, with empty traces."""
        credit = Credit.create(1, self.goal_baseline_map, credit_settings=self.credit_settings)
        return Emitter.create(
            observation_map=self.observation_map,
            instrument_map=self.instrument_map,
            credit=credit,
            controllable=True,
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
        """Emit, evaluate the oracle goal at the emitted value, and form each credit vector.

        The mean trace does not depend on the goal, so the model emits before the goal arrives.
        """
        assert isinstance(observation, DirectCreditState)
        emitter = self.emitter()
        absent = NormalNP(jnp.zeros(1), jnp.zeros(1))
        emitter_state, outputs = emitter.step(
            eqx.nn.State(emitter),
            streams["inference"].key(),
            observation_content=observation.content,
            predictor_instrument=absent,
            gain_observation=observation.gain,
            goal=jnp.zeros(()),
        )
        value = outputs.message.observation.mean()[0]
        goal = self.scenario.goal(value, observation.sample)
        gain = observation.gain.mean()[0]
        _, pullback = emitter.observation_projection.infer(observation.content)

        def pull_back_mean(cotangent: JaxRealArray) -> JaxRealArray:
            (parameters,) = pullback(
                ObservationCoordinates(mean=cotangent, precision=jnp.zeros_like(cotangent))
            )
            return _flatten(parameters)

        instrument_precision = outputs.noised_instrument.to_precision_parametrization().precision
        mean_trace = emitter_state.get(emitter.trace_index).mean
        configuration = DirectCreditConfiguration(
            gain=gain,
            value=value,
            goal=goal,
            mean_credit=goal * _flatten(mean_trace),
            shrinkage_credit=pull_back_mean(outputs.credit_outputs.shrinkage_cotangent),
            observational_credit=goal * pull_back_mean(gain * instrument_precision * value),
            mean_jacobian=pull_back_mean(jnp.ones(1)),
        )
        return ModelResult(
            loss=jnp.zeros(()),
            configurations=frozendict({_FEATURE_NODE: configuration}),
            state=state,
        )

    def parameter_names(self) -> list[str]:
        """Return the observation map's trainable parameter names in flattened order."""
        emitter = self.emitter()
        trace = eqx.nn.State(emitter).get(emitter.trace_index).mean
        names: list[str] = []
        for path, leaf in tree.leaves_with_path(trace):
            name = "".join(str(key) for key in path).removeprefix(".").removesuffix(".value")
            names.extend([name] if leaf.size == 1 else [f"{name}{i}" for i in range(leaf.size)])
        return names


class DirectCreditTelemetry(Telemetry):
    """Record every held-out emission's configuration."""

    @override
    def inference_snapshot(
        self,
        inference: Inference,
        inference_result: InferenceResult,
        snapshots: Mapping[Telemetry, Any],
    ) -> DirectCreditConfiguration:
        configuration = inference_result.model_configuration[_FEATURE_NODE]
        assert isinstance(configuration, DirectCreditConfiguration)
        return configuration


@dataclass(frozen=True)
class Alignment:
    """The inner product of an expected update with a gradient, and the angle between them.

    The angle is in degrees. An update ascends the gradient's objective when the inner product is
    positive, that is, when the angle is below 90°.
    """

    inner_product: float
    angle: float

    @classmethod
    def between(cls, update: np.ndarray, gradient: np.ndarray) -> Self:
        inner_product = float(update @ gradient)
        cosine = inner_product / float(np.linalg.norm(update) * np.linalg.norm(gradient))
        return cls(inner_product, math.degrees(math.acos(max(-1.0, min(1.0, cosine)))))


@dataclass(frozen=True)
class DirectCreditScores:
    """One run's expected credit vectors, the oracle gradients, and their alignments.

    Each expectation is a mean over the run's held-out emissions. The causal gradient is the
    derivative of the expected goal with respect to the observation map's parameters, β E[∂μ/∂θ].
    The gain-weighted gradient is the derivative of the expected gain-weighted goal, β E[γ ∂μ/∂θ].
    The confounded slope is the regression slope of the goal on the emitted value.
    """

    confounded_slope: float
    mean_credit: list[float]
    shrinkage_credit: list[float]
    observational_credit: list[float]
    causal_gradient: list[float]
    gain_weighted_gradient: list[float]
    causal: Alignment
    gain_weighted: Alignment
    with_shrinkage: Alignment
    observational: Alignment


def score_direct_credit_run(run: Run, scenario: DirectCreditScenario) -> DirectCreditScores:
    samples = run.inference_results.telemetries[DirectCreditTelemetry()]
    assert isinstance(samples, DirectCreditConfiguration)

    def vectors(values: JaxRealArray) -> np.ndarray:
        array = np.asarray(values)
        return array.reshape(-1, array.shape[-1])

    gain = np.asarray(samples.gain).ravel()
    value = np.asarray(samples.value).ravel()
    goal = np.asarray(samples.goal).ravel()
    jacobian = vectors(samples.mean_jacobian)
    mean_credit = vectors(samples.mean_credit).mean(axis=0)
    shrinkage_credit = vectors(samples.shrinkage_credit).mean(axis=0)
    observational_credit = vectors(samples.observational_credit).mean(axis=0)
    causal_gradient = scenario.causal_effect * jacobian.mean(axis=0)
    gain_weighted_gradient = scenario.causal_effect * (gain[:, np.newaxis] * jacobian).mean(axis=0)
    return DirectCreditScores(
        confounded_slope=float(np.cov(goal, value)[0, 1] / np.var(value, ddof=1)),
        mean_credit=mean_credit.tolist(),
        shrinkage_credit=shrinkage_credit.tolist(),
        observational_credit=observational_credit.tolist(),
        causal_gradient=causal_gradient.tolist(),
        gain_weighted_gradient=gain_weighted_gradient.tolist(),
        causal=Alignment.between(mean_credit, causal_gradient),
        gain_weighted=Alignment.between(mean_credit, gain_weighted_gradient),
        with_shrinkage=Alignment.between(mean_credit + shrinkage_credit, causal_gradient),
        observational=Alignment.between(observational_credit, causal_gradient),
    )


class DirectCreditSolver(Solver[DirectCreditProblem]):
    """Measure an emitter's credit at its operating point on held-out samples, without training."""

    _: KW_ONLY
    scenario: DirectCreditScenario = eqx.field(default_factory=DirectCreditScenario)
    observation_weight: float = eqx.field(static=True, default=1.0)
    credit_settings: CreditSettings = eqx.field(
        static=True, default_factory=lambda: CreditSettings(trace_decay_rate=1.0)
    )
    inference_examples: int = int_field(default=256, domain=IntDistribution(1, 1 << 12, log=True))
    inference_batch_size: int = int_field(default=256, domain=IntDistribution(1, 1 << 12, log=True))

    @override
    def problem(self) -> DirectCreditProblem:
        return DirectCreditProblem(self.scenario)

    @override
    def create_model(
        self,
        data_source: DataSource,
        problem: Problem,
        *,
        streams: Mapping[str, RngStream],
    ) -> Model:
        return DirectCreditModel.create(
            self.scenario,
            observation_weight=self.observation_weight,
            credit_settings=self.credit_settings,
            streams=streams,
        )

    @override
    def training_results(self, *, packet: ExecutionPacket) -> TrainingResults:
        """Return the initial state, since the test measures credit at fixed parameters."""
        return TrainingResults(count=0, telemetries={}, final_state=self.solution().solution_state)


class DirectCreditPlotter(Plotter):
    """Compare each condition's expected credit with the oracle gradients.

    The bars show the angle between each expected update and a gradient. The seed scores record
    the inner products and the vectors, labelled by the parameter names.
    """

    _: KW_ONLY
    conditions: Mapping[str, DirectCreditScenario] = field(static=True)

    @override
    def telemetries(self) -> Telemetries:
        return Telemetries((DirectCreditTelemetry(),))

    @override
    def plotted_series(self, runs: Sequence[Run]) -> PlottedSeries:
        scores = {
            condition: [
                score_direct_credit_run(run, scenario) for run in seed_runs(runs, condition)
            ]
            for condition, scenario in self.conditions.items()
        }

        def angles(alignment: Callable[[DirectCreditScores], Alignment]) -> list[list[float]]:
            return [
                [alignment(score).angle for score in seed_scores] for seed_scores in scores.values()
            ]

        names = DirectCreditModel.create(
            next(iter(self.conditions.values())),
            observation_weight=1.0,
            credit_settings=CreditSettings(trace_decay_rate=1.0),
            streams=create_streams({"parameters": jr.key(0)}),
        ).parameter_names()
        return bar_chart(
            list(self.conditions),
            {
                "causal": ("Goal credit and causal gradient", angles(lambda s: s.causal)),
                "gain weighted": (
                    "Goal credit and gain-weighted causal gradient",
                    angles(lambda s: s.gain_weighted),
                ),
                "with shrinkage": (
                    "Goal and shrinkage credit and causal gradient",
                    angles(lambda s: s.with_shrinkage),
                ),
                "observational": (
                    "Observational credit and causal gradient",
                    angles(lambda s: s.observational),
                ),
            },
        ) | {
            "parameters": names,
            "seed scores": {
                condition: [asdict(score) for score in seed_scores]
                for condition, seed_scores in scores.items()
            },
        }


UNIT_GAIN = "Unit gain"
ENDOGENOUS_GAIN = "Endogenous gain"
DIRECT_CREDIT_SCENARIOS = {
    UNIT_GAIN: DirectCreditScenario(),
    ENDOGENOUS_GAIN: DirectCreditScenario(
        base_gain=0.5, gain_confounding=0.2, gain_noise_width=0.2
    ),
}


class DirectCreditExperiment(SeededExperiment):
    """Test 9: whether a controllable feature's goal credit follows the goal's causal gradient.

    The goal's confounded correlation with the feature opposes its causal effect. Both conditions
    measure credit at the same operating point on held-out emissions. The unit-gain condition
    weights every emission equally. Under endogenous gain, the gain is
    0.5 + 0.2 tanh(U) + γ̃, so credit weights emissions by a gain that depends on the confounder.
    """

    def __init__(
        self, solver: DirectCreditSolver | None = None, *, seeds: int = SEED_COUNT
    ) -> None:
        solver = DirectCreditSolver() if solver is None else solver
        super().__init__(
            name="sip-direct-credit",
            variants=[
                Variant(label=condition, solver=replace(solver, scenario=scenario))
                for condition, scenario in DIRECT_CREDIT_SCENARIOS.items()
            ],
            plotters=[
                DirectCreditPlotter(
                    name="direct-credit-angles",
                    title="Angles between expected credit and the causal gradients",
                    conditions=DIRECT_CREDIT_SCENARIOS,
                )
            ],
            sweep=[{"seed": seed} for seed in range(seeds)],
        )


sip_direct_credit_experiment = DirectCreditExperiment()
