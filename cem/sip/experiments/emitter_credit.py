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
from efax import NormalNP, NormalPP
from jax import tree
from optuna.distributions import FloatDistribution, IntDistribution
from tjax import JaxRealArray, KeyArray, RngStream, create_streams
from tjax.dataclasses import field

from cem.framework.graph import Model, ModelResult, NodeConfiguration
from cem.framework.plotter import Run, Variant
from cem.framework.plotter.plotter import PlottedSeries, Plotter
from cem.framework.problem import DataSource, Problem
from cem.framework.solution import Inference, InferenceResult, Telemetries, Telemetry
from cem.framework.solver import Solver, float_field, int_field
from cem.npn.gaussian import GaussianNeuralProjection
from cem.sip.credit import Credit
from cem.sip.emitter import Emitter, ObservationCoordinates
from cem.sip.messages import InstrumentedMessage

from .common import SEED_COUNT, SeededExperiment, bar_chart, seed_runs
from .datasets import (
    FEATURE_OBSERVATION_VARIANCE,
    DirectCreditProblem,
    DirectCreditState,
    InheritedCreditProblem,
    InheritedCreditState,
)
from .scenarios import DirectCreditScenario, InheritedCreditScenario

_FEATURE_NODE = "feature"


class DirectCreditConfiguration(NodeConfiguration):
    """One emission's gain, value, goal, and goal residual, and parameter vectors for the map.

    The goal residual is the goal minus the fitted goal baseline. Each vector flattens the
    observation map's trainable parameters in leaf order. The mean credit is the goal residual
    times the recorded mean cotangent, the update direction that the goal contributes. The
    shrinkage credit is the update that mean shrinkage contributes. The observational credit pairs
    the same residual with the emitted value in place of the instrument. The Jacobian is the
    derivative of the raw mean with respect to the parameters.
    """

    gain: JaxRealArray  # scalar
    value: JaxRealArray  # scalar
    goal: JaxRealArray  # scalar
    goal_residual: JaxRealArray  # scalar
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


class _Measurement(eqx.Module):
    """One emission's credit measurements and the losses that fit its maps.

    The goal residual is the goal minus the fitted goal baseline. The instrument is the noised
    instrument's mean before gain, and the substrate is the predictor substrate's mean. Each vector
    flattens the observation map's trainable parameters in leaf order:
    - mean credit is the goal residual times the recorded mean cotangent;
    - the mean drift is the recorded mean cotangent itself;
    - raw-goal credit is the goal itself times that cotangent;
    - shrinkage credit is the update that mean shrinkage contributes;
    - observational credit pairs the goal residual with the emitted value in place of the
      instrument;
    - the Jacobian is the derivative of the raw mean with respect to the parameters.
    """

    gain: JaxRealArray  # scalar
    value: JaxRealArray  # scalar
    goal: JaxRealArray  # scalar
    goal_residual: JaxRealArray  # scalar
    instrument: JaxRealArray  # scalar
    substrate: JaxRealArray  # scalar
    mean_credit: JaxRealArray  # (p,)
    mean_drift: JaxRealArray  # (p,)
    raw_goal_credit: JaxRealArray  # (p,)
    shrinkage_credit: JaxRealArray  # (p,)
    observational_credit: JaxRealArray  # (p,)
    mean_jacobian: JaxRealArray  # (p,)
    baseline_loss: JaxRealArray  # scalar, the goal and quadratic cost baseline losses
    instrument_loss: JaxRealArray  # scalar


def _measure_credit(
    emitter: Emitter,
    key: KeyArray,
    *,
    content: NormalNP,
    predictor: InstrumentedMessage,
    gain_observation: NormalNP,
    goal_of: Callable[[JaxRealArray], JaxRealArray],
) -> _Measurement:
    """Emit, score the oracle goal at the emitted value, and form each credit vector.

    The emission's drift does not depend on the goal, so the emitter emits before the goal arrives,
    retains the emission, and then credits the goal to that emission alone, as the lag filter does
    when the goal arrives at the next step.
    """
    credit_state, emission = emitter.emit(
        eqx.nn.State(emitter),
        key,
        observation_content=content,
        predictor=predictor,
        gain_observation=gain_observation,
    )
    value = emission.message.observation.mean()[0]
    goal = goal_of(value)
    gain = gain_observation.mean()[0]
    credit_state, _ = emitter.credit.credit(credit_state, emission.credit_emission, jnp.zeros(()))
    _, credit_outputs = emitter.credit.credit(credit_state, emission.credit_emission, goal)
    goal_residual = credit_outputs.baseline_goal_residual
    _, pullback = emitter.observation_projection.infer(content)

    def pull_back_mean(cotangent: JaxRealArray) -> JaxRealArray:
        (parameters,) = pullback(
            ObservationCoordinates(mean=cotangent, precision=jnp.zeros_like(cotangent))
        )
        return _flatten(parameters)

    instrument = emission.noised_instrument.to_precision_parametrization()
    mean_drift = _flatten(emission.mean_drift)
    return _Measurement(
        gain=gain,
        value=value,
        goal=goal,
        goal_residual=goal_residual,
        instrument=instrument.mean[0],
        substrate=emission.credit_emission.predictor_substrate.mean()[0],
        mean_credit=goal_residual * mean_drift,
        mean_drift=mean_drift,
        raw_goal_credit=goal * mean_drift,
        shrinkage_credit=pull_back_mean(emission.credit_emission.shrinkage_cotangent),
        observational_credit=goal_residual * pull_back_mean(gain * instrument.precision * value),
        mean_jacobian=pull_back_mean(jnp.ones(1)),
        baseline_loss=credit_outputs.goal_baseline_loss + credit_outputs.quadratic_cost_loss,
        instrument_loss=emission.loss,
    )


def _parameter_names(emitter: Emitter) -> list[str]:
    """Return the observation map's trainable parameter names in flattened order."""
    trace = emitter.observation_projection.zero_trace()
    names: list[str] = []
    for path, leaf in tree.leaves_with_path(trace):
        name = "".join(str(key) for key in path).removeprefix(".").removesuffix(".value")
        names.extend([name] if leaf.size == 1 else [f"{name}{i}" for i in range(leaf.size)])
    return names


class DirectCreditModel(Model):
    """A controllable emitter whose observation reads the confounder, at a fixed operating point.

    The observation map is affine in the observation content, with the given weight and a zero
    bias. The feature has no parents, so its predictor message is absent. The instrument map's
    weight is zero, so the inherited instrument is the map's initial bias, a standard normal. Each
    emission therefore injects unit-variance noise with a centred, unit-precision instrument. The
    absent predictor leaves the substrate empty, so the fitted goal baseline is the mean goal.

    The model stores the emitter's maps and assembles the emitter for each example, then credits
    the goal to that example's emission alone, as the lag filter does when the goal arrives at the
    next step. Its loss holds the observation and instrument maps fixed, so training fits only the
    goal and quadratic cost baselines.
    """

    observation_map: GaussianNeuralProjection
    instrument_map: GaussianNeuralProjection
    goal_baseline_map: GaussianNeuralProjection
    quadratic_cost_baseline_map: GaussianNeuralProjection
    scenario: DirectCreditScenario = eqx.field(static=True)

    @classmethod
    def create(
        cls,
        scenario: DirectCreditScenario,
        *,
        observation_weight: float,
        streams: Mapping[str, RngStream],
    ) -> Self:
        def projection() -> GaussianNeuralProjection:
            return GaussianNeuralProjection.create(in_features=1, out_features=1, streams=streams)

        return cls(
            observation_map=_with_output_weight(projection(), observation_weight),
            instrument_map=_with_output_weight(projection(), 0.0),
            goal_baseline_map=projection(),
            quadratic_cost_baseline_map=projection(),
            scenario=scenario,
        )

    def emitter(self) -> Emitter:
        """Assemble a controllable emitter from the maps, with empty traces."""
        credit = Credit.create(self.goal_baseline_map, self.quadratic_cost_baseline_map)
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
        """Measure the emission's credit and fit the baselines to its goal and cost.

        See `_measure_credit`. The loss fits the goal and quadratic cost baselines.
        """
        assert isinstance(observation, DirectCreditState)
        absent = NormalNP(jnp.zeros(1), jnp.zeros(1))
        measurement = _measure_credit(
            self.emitter(),
            streams["inference"].key(),
            content=observation.content,
            predictor=InstrumentedMessage(absent, absent),
            gain_observation=observation.gain,
            goal_of=lambda value: self.scenario.goal(value, observation.sample),
        )
        configuration = DirectCreditConfiguration(
            gain=measurement.gain,
            value=measurement.value,
            goal=measurement.goal,
            goal_residual=measurement.goal_residual,
            mean_credit=measurement.mean_credit,
            shrinkage_credit=measurement.shrinkage_credit,
            observational_credit=measurement.observational_credit,
            mean_jacobian=measurement.mean_jacobian,
        )
        return ModelResult(
            loss=measurement.baseline_loss,
            configurations=frozendict({_FEATURE_NODE: configuration}),
            state=state,
        )

    def parameter_names(self) -> list[str]:
        """Return the observation map's trainable parameter names in flattened order."""
        return _parameter_names(self.emitter())


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
    The confounded slope is the regression slope of the goal on the emitted value. The goal
    baseline is the fitted baseline's mean on the held-out emissions, for comparison with their
    mean goal.
    """

    confounded_slope: float
    goal_baseline: float
    mean_goal: float
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
    goal_baseline = goal - np.asarray(samples.goal_residual).ravel()
    jacobian = vectors(samples.mean_jacobian)
    mean_credit = vectors(samples.mean_credit).mean(axis=0)
    shrinkage_credit = vectors(samples.shrinkage_credit).mean(axis=0)
    observational_credit = vectors(samples.observational_credit).mean(axis=0)
    causal_gradient = scenario.causal_effect * jacobian.mean(axis=0)
    gain_weighted_gradient = scenario.causal_effect * (gain[:, np.newaxis] * jacobian).mean(axis=0)
    return DirectCreditScores(
        confounded_slope=float(np.cov(goal, value)[0, 1] / np.var(value, ddof=1)),
        goal_baseline=float(goal_baseline.mean()),
        mean_goal=float(goal.mean()),
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
    """Fit an emitter's credit baselines, then measure its credit on held-out samples.

    Training batches fit only the goal and quadratic cost baselines, at the fixed operating point.
    Measurement then holds the fitted baselines fixed.
    """

    _: KW_ONLY
    scenario: DirectCreditScenario = eqx.field(default_factory=DirectCreditScenario)
    observation_weight: float = eqx.field(static=True, default=1.0)
    training_examples: int = int_field(default=512, domain=IntDistribution(1, 1 << 12, log=True))
    learning_rate: float = float_field(
        default=0.01, domain=FloatDistribution(1e-4, 1.0, log=True), optimize=True
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
            streams=streams,
        )


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
            streams=create_streams({"parameters": jr.key(0)}),
        ).parameter_names()
        return bar_chart(
            list(self.conditions),
            {
                "causal": ("Mean credit and causal gradient", angles(lambda s: s.causal)),
                "gain weighted": (
                    "Mean credit and gain-weighted causal gradient",
                    angles(lambda s: s.gain_weighted),
                ),
                "with shrinkage": (
                    "Mean and shrinkage credit and causal gradient",
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
    """Test 9: whether a controllable feature's mean credit follows the goal's causal gradient.

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


class InheritedCreditConfiguration(DirectCreditConfiguration):
    """An inherited emission's direct-credit measurements, with its instrument and raw-goal credit.

    The instrument is the inherited instrument's mean, and the substrate is the predictor
    substrate's mean. The mean drift is the recorded mean cotangent, and raw-goal credit is the
    goal itself times that cotangent.
    """

    instrument: JaxRealArray  # scalar
    substrate: JaxRealArray  # scalar
    mean_drift: JaxRealArray  # (p,)
    raw_goal_credit: JaxRealArray  # (p,)


class InheritedCreditModel(Model):
    """A non-controllable emitter whose only experiments come from its parent, at a fixed point.

    The observation map is affine in the observation content with weight 1, so the raw mean
    tracks the parent's value plus the confounder. Its bias makes the gain-weighted raw mean zero,
    where mean shrinkage would hold it. The gain-weighted instrument loss then centres the fitted
    instrument too, as the mean-credit argument assumes. The emitter injects no noise; the
    instrument map predicts the raw observation from the parent's instrument. Training fits that
    first stage and the goal and quadratic cost baselines, and holds the observation map fixed.
    """

    observation_map: GaussianNeuralProjection
    instrument_map: GaussianNeuralProjection
    goal_baseline_map: GaussianNeuralProjection
    quadratic_cost_baseline_map: GaussianNeuralProjection
    scenario: InheritedCreditScenario = eqx.field(static=True)

    @classmethod
    def create(cls, scenario: InheritedCreditScenario, *, streams: Mapping[str, RngStream]) -> Self:
        def projection() -> GaussianNeuralProjection:
            return GaussianNeuralProjection.create(in_features=1, out_features=1, streams=streams)

        observation_map = _with_output_weight(projection(), 1.0)
        centre = NormalPP(
            jnp.full(1, scenario.gain_weighted_content_mean()),
            jnp.full(1, 1.0 / FEATURE_OBSERVATION_VARIANCE),
        ).to_nat()
        # The raw mean is affine in the content's mean, so this bias centres it under gain
        # weighting.
        observation_map = eqx.tree_at(
            lambda p: p.layers[0].bias.value,
            observation_map,
            observation_map.layers[0].bias.value - observation_map.infer(centre).mean(),
        )
        return cls(
            observation_map=observation_map,
            instrument_map=projection(),
            goal_baseline_map=projection(),
            quadratic_cost_baseline_map=projection(),
            scenario=scenario,
        )

    def emitter(self) -> Emitter:
        """Assemble a non-controllable emitter from the maps, with empty traces."""
        credit = Credit.create(self.goal_baseline_map, self.quadratic_cost_baseline_map)
        return Emitter.create(
            observation_map=self.observation_map,
            instrument_map=self.instrument_map,
            credit=credit,
            controllable=False,
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
        """Measure the emission's credit and fit the first stage and baselines.

        See `_measure_credit`. The loss sums the instrument loss and the goal and quadratic cost
        baseline losses.
        """
        assert isinstance(observation, InheritedCreditState)
        measurement = _measure_credit(
            self.emitter(),
            streams["inference"].key(),
            content=observation.content,
            predictor=observation.predictor,
            gain_observation=observation.gain,
            goal_of=lambda value: self.scenario.goal(value, observation.sample),
        )
        configuration = InheritedCreditConfiguration(
            gain=measurement.gain,
            value=measurement.value,
            goal=measurement.goal,
            goal_residual=measurement.goal_residual,
            mean_credit=measurement.mean_credit,
            shrinkage_credit=measurement.shrinkage_credit,
            observational_credit=measurement.observational_credit,
            mean_jacobian=measurement.mean_jacobian,
            instrument=measurement.instrument,
            substrate=measurement.substrate,
            mean_drift=measurement.mean_drift,
            raw_goal_credit=measurement.raw_goal_credit,
        )
        return ModelResult(
            loss=measurement.baseline_loss + measurement.instrument_loss,
            configurations=frozendict({_FEATURE_NODE: configuration}),
            state=state,
        )

    def parameter_names(self) -> list[str]:
        """Return the observation map's trainable parameter names in flattened order."""
        return _parameter_names(self.emitter())


class InheritedCreditTelemetry(Telemetry):
    """Record every held-out inherited emission's configuration."""

    @override
    def inference_snapshot(
        self,
        inference: Inference,
        inference_result: InferenceResult,
        snapshots: Mapping[Telemetry, Any],
    ) -> InheritedCreditConfiguration:
        configuration = inference_result.model_configuration[_FEATURE_NODE]
        assert isinstance(configuration, InheritedCreditConfiguration)
        return configuration


def _slope(response: np.ndarray, regressor: np.ndarray) -> float:
    """Return the least-squares slope of the response on the regressor, with an intercept."""
    return float(np.cov(response, regressor)[0, 1] / np.var(regressor, ddof=1))


@dataclass(frozen=True)
class InheritedCreditScores:
    """One run's fitted maps, expected credit vectors, oracle gradients, and their alignments.

    Each expectation is a mean over the run's held-out emissions. The target gradient is the
    gain-weighted causal gradient β E[γ ∂μ/∂θ], which equals the causal gradient β E[∂μ/∂θ] under
    unit gain. Three forms of credit scale the same recorded mean cotangent: raw-goal credit uses
    the goal itself, constant-baseline credit subtracts the held-out mean goal, and mean credit
    subtracts the fitted goal baseline, which reads the substrate.

    Three diagnostics check the fitted maps. The first-stage slope regresses the emitted value on
    the instrument, which an exact first stage puts at one. The instrument mean should be near
    zero, as should the baseline slope, which regresses the goal residual on the substrate. The
    confounded slope regresses the goal on the value.
    """

    confounded_slope: float
    first_stage_slope: float
    instrument_mean: float
    baseline_slope: float
    mean_residual: float
    mean_credit: list[float]
    raw_goal_credit: list[float]
    constant_baseline_credit: list[float]
    shrinkage_credit: list[float]
    observational_credit: list[float]
    causal_gradient: list[float]
    gain_weighted_gradient: list[float]
    target: Alignment  # mean credit and the gain-weighted gradient
    raw_goal: Alignment  # raw-goal credit and the gain-weighted gradient
    constant_baseline: Alignment  # constant-baseline credit and the gain-weighted gradient
    causal: Alignment  # mean credit and the causal gradient
    with_shrinkage: Alignment  # mean and shrinkage credit and the gain-weighted gradient
    observational: Alignment  # observational credit and the causal gradient


def score_inherited_credit_run(
    run: Run, scenario: InheritedCreditScenario
) -> InheritedCreditScores:
    samples = run.inference_results.telemetries[InheritedCreditTelemetry()]
    assert isinstance(samples, InheritedCreditConfiguration)

    def scalars(values: JaxRealArray) -> np.ndarray:
        return np.asarray(values).ravel()

    def mean_vector(values: JaxRealArray) -> np.ndarray:
        array = np.asarray(values)
        return array.reshape(-1, array.shape[-1]).mean(axis=0)

    gain, value, goal = scalars(samples.gain), scalars(samples.value), scalars(samples.goal)
    instrument, residual = scalars(samples.instrument), scalars(samples.goal_residual)
    jacobian = np.asarray(samples.mean_jacobian).reshape(gain.size, -1)
    mean_credit = mean_vector(samples.mean_credit)
    raw_goal_credit = mean_vector(samples.raw_goal_credit)
    constant_baseline_credit = raw_goal_credit - goal.mean() * mean_vector(samples.mean_drift)
    shrinkage_credit = mean_vector(samples.shrinkage_credit)
    observational_credit = mean_vector(samples.observational_credit)
    causal_gradient = scenario.causal_effect * jacobian.mean(axis=0)
    gain_weighted_gradient = scenario.causal_effect * (gain[:, np.newaxis] * jacobian).mean(axis=0)
    return InheritedCreditScores(
        confounded_slope=_slope(goal, value),
        first_stage_slope=_slope(value, instrument),
        instrument_mean=float(instrument.mean()),
        baseline_slope=_slope(residual, scalars(samples.substrate)),
        mean_residual=float(residual.mean()),
        mean_credit=mean_credit.tolist(),
        raw_goal_credit=raw_goal_credit.tolist(),
        constant_baseline_credit=constant_baseline_credit.tolist(),
        shrinkage_credit=shrinkage_credit.tolist(),
        observational_credit=observational_credit.tolist(),
        causal_gradient=causal_gradient.tolist(),
        gain_weighted_gradient=gain_weighted_gradient.tolist(),
        target=Alignment.between(mean_credit, gain_weighted_gradient),
        raw_goal=Alignment.between(raw_goal_credit, gain_weighted_gradient),
        constant_baseline=Alignment.between(constant_baseline_credit, gain_weighted_gradient),
        causal=Alignment.between(mean_credit, causal_gradient),
        with_shrinkage=Alignment.between(mean_credit + shrinkage_credit, gain_weighted_gradient),
        observational=Alignment.between(observational_credit, causal_gradient),
    )


class InheritedCreditSolver(Solver[InheritedCreditProblem]):
    """Fit an inherited emitter's first stage and credit baselines, then measure its credit.

    Training batches fit the instrument map and the goal and quadratic cost baselines at the fixed
    operating point. Measurement then holds them fixed.
    """

    _: KW_ONLY
    scenario: InheritedCreditScenario = eqx.field(default_factory=InheritedCreditScenario)
    training_examples: int = int_field(default=8192, domain=IntDistribution(1, 1 << 14, log=True))
    learning_rate: float = float_field(
        default=0.001, domain=FloatDistribution(1e-4, 1.0, log=True), optimize=True
    )
    inference_examples: int = int_field(default=256, domain=IntDistribution(1, 1 << 12, log=True))
    inference_batch_size: int = int_field(default=256, domain=IntDistribution(1, 1 << 12, log=True))

    @override
    def problem(self) -> InheritedCreditProblem:
        return InheritedCreditProblem(self.scenario)

    @override
    def create_model(
        self,
        data_source: DataSource,
        problem: Problem,
        *,
        streams: Mapping[str, RngStream],
    ) -> Model:
        return InheritedCreditModel.create(self.scenario, streams=streams)


class InheritedCreditPlotter(Plotter):
    """Compare each condition's expected credit with its target gradient.

    The bars show the angle between each expected update and a gradient. The seed scores record
    the fitted maps' diagnostics, the inner products, and the vectors, labelled by the parameter
    names.
    """

    _: KW_ONLY
    conditions: Mapping[str, InheritedCreditScenario] = field(static=True)

    @override
    def telemetries(self) -> Telemetries:
        return Telemetries((InheritedCreditTelemetry(),))

    @override
    def plotted_series(self, runs: Sequence[Run]) -> PlottedSeries:
        scores = {
            condition: [
                score_inherited_credit_run(run, scenario) for run in seed_runs(runs, condition)
            ]
            for condition, scenario in self.conditions.items()
        }

        def angles(alignment: Callable[[InheritedCreditScores], Alignment]) -> list[list[float]]:
            return [
                [alignment(score).angle for score in seed_scores] for seed_scores in scores.values()
            ]

        names = InheritedCreditModel.create(
            next(iter(self.conditions.values())),
            streams=create_streams({"parameters": jr.key(0)}),
        ).parameter_names()
        return bar_chart(
            list(self.conditions),
            {
                "target": ("Mean credit and target gradient", angles(lambda s: s.target)),
                "raw goal": ("Raw-goal credit and target gradient", angles(lambda s: s.raw_goal)),
                "constant baseline": (
                    "Constant-baseline credit and target gradient",
                    angles(lambda s: s.constant_baseline),
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


CONFOUNDED_GAIN = "Confounded gain"
CONTEXT_GAIN = "Context gain"
EXOGENOUS_GAIN = "Exogenous gain"
_GOAL_OFFSET = 2.0
INHERITED_CREDIT_SCENARIOS = {
    UNIT_GAIN: InheritedCreditScenario(goal_offset=_GOAL_OFFSET),
    CONFOUNDED_GAIN: InheritedCreditScenario(
        goal_offset=_GOAL_OFFSET, base_gain=0.5, gain_confounding=0.2, gain_noise_width=0.2
    ),
    CONTEXT_GAIN: InheritedCreditScenario(
        goal_offset=_GOAL_OFFSET,
        base_gain=0.5,
        gain_confounding=0.2,
        gain_noise_width=0.2,
        gain_reads_context=True,
    ),
    EXOGENOUS_GAIN: InheritedCreditScenario(
        goal_offset=_GOAL_OFFSET, base_gain=0.5, gain_noise_width=0.4
    ),
}


class InheritedCreditExperiment(SeededExperiment):
    """Test 10: whether an inherited feature's mean credit follows the goal's causal gradient.

    The feature's only experiments come from its parent, so the draw changes both the instrument
    and the observation map's parameter derivative. Every condition adds 2 to the goal and
    measures credit on held-out emissions, with the confounder hidden from the substrate:
    - unit gain;
    - gain 0.5 + 0.2 tanh(V) + γ̃, which depends on the hidden confounder V;
    - gain 0.5 + 0.2 tanh(C) + γ̃, which depends on the context C that the substrate reveals;
    - gain 0.5 + γ̃ with γ̃ uniform on [-0.4, 0.4], which depends on neither.
    """

    def __init__(
        self, solver: InheritedCreditSolver | None = None, *, seeds: int = SEED_COUNT
    ) -> None:
        solver = InheritedCreditSolver() if solver is None else solver
        super().__init__(
            name="sip-inherited-credit",
            variants=[
                Variant(label=condition, solver=replace(solver, scenario=scenario))
                for condition, scenario in INHERITED_CREDIT_SCENARIOS.items()
            ],
            plotters=[
                InheritedCreditPlotter(
                    name="inherited-credit-angles",
                    title="Angles between expected inherited credit and the causal gradients",
                    conditions=INHERITED_CREDIT_SCENARIOS,
                )
            ],
            sweep=[{"seed": seed} for seed in range(seeds)],
        )


sip_inherited_credit_experiment = InheritedCreditExperiment()
