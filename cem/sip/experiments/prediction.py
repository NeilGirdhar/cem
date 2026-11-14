"""Prediction experiments: whether purification makes prediction maps learn causal effects.

See `sip.md`, tests 4–7.
"""

from collections.abc import Mapping, Sequence
from dataclasses import KW_ONLY, dataclass, replace
from typing import Any, Self, override

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from optuna.distributions import FloatDistribution, IntDistribution
from tjax import JaxRealArray, RngStream
from tjax.dataclasses import field
from tjax.gradient import SGD, ChainedGradientTransformation, KeepParamsNonnegative

from cem.framework.graph import (
    DisGradientTransformation,
    FixedParameter,
    LearnableParameter,
    Model,
    ModelResult,
    NodeConfiguration,
    ParameterType,
)
from cem.framework.plotter import Run, Variant
from cem.framework.plotter.plotter import PlottedSeries, Plotter
from cem.framework.problem import DataSource, Problem
from cem.framework.solution import Inference, InferenceResult, Telemetries, Telemetry
from cem.framework.solver import Solver, bool_field, float_field, int_field
from cem.sip.mismatch import Mismatch
from cem.sip.parameters import WitnessMultiplierParameter
from cem.sip.settings import MismatchSettings

from .baselines import ordinary_least_squares_coefficients, two_stage_least_squares_coefficients
from .common import SEED_COUNT, SeededExperiment, bar_chart
from .datasets import (
    FEATURE_OBSERVATION_VARIANCE,
    ConfoundedPredictionObservation,
    ConfoundedPredictionProblem,
    ConfoundedPredictionState,
)
from .scenarios import ConfoundedPredictionScenario

_TARGET_NODE = "target"


class PredictionConfiguration(NodeConfiguration):
    """The mean and variance of the target's prediction."""

    predicted_mean: JaxRealArray  # scalar
    predicted_variance: JaxRealArray  # scalar


class PurificationModel(Model):
    """A target's mismatch circuit, trained with or without purification.

    With purification, the circuit's full loss trains the prediction map, including the
    confounding cost from the witnesses over the parents' instruments. Without it, only the
    reconstruction cost trains the prediction map.
    """

    mismatch: Mismatch
    purify: bool = eqx.field(static=True)

    @classmethod
    def create(cls, n_parents: int, *, purify: bool, streams: Mapping[str, RngStream]) -> Self:
        mismatch = Mismatch.create(
            1,
            predictor_features=n_parents,
            gain_features=0,
            mismatch_settings=MismatchSettings(),
            streams=streams,
        )
        return cls(mismatch, purify=purify)

    @override
    def infer(
        self,
        observation: object,
        state: object,
        *,
        streams: Mapping[str, RngStream],
        inference: bool,
    ) -> ModelResult:
        assert isinstance(observation, ConfoundedPredictionObservation)
        outputs = self.mismatch.step(
            raw_observation=observation.target_observation,
            local_variance=jnp.zeros(1),
            predictor=observation.predictor,
            gain=None,
            streams=streams,
            inference=inference,
        )
        prediction = self.mismatch.prediction_map.infer(
            observation.predictor.observation
        ).to_variance_parametrization()
        return ModelResult(
            loss=outputs.loss if self.purify else outputs.reconstruction_cost,
            configurations=frozendict(
                {
                    _TARGET_NODE: PredictionConfiguration(
                        predicted_mean=prediction.mean[0], predicted_variance=prediction.variance[0]
                    )
                }
            ),
            state=state,
        )


class PredictionSamples(eqx.Module):
    """Held-out examples: the prediction's moments and the structural variables."""

    predicted_mean: JaxRealArray  # (..., batch)
    predicted_variance: JaxRealArray  # (..., batch)
    injected_noise: JaxRealArray  # (..., batch, k)
    parents: JaxRealArray  # (..., batch, k)
    target: JaxRealArray  # (..., batch)


class PredictionTelemetry(Telemetry):
    """Record every held-out example so the estimators can run on the pooled samples."""

    @override
    def inference_snapshot(
        self,
        inference: Inference,
        inference_result: InferenceResult,
        snapshots: Mapping[Telemetry, Any],
    ) -> PredictionSamples:
        configuration = inference_result.model_configuration[_TARGET_NODE]
        assert isinstance(configuration, PredictionConfiguration)
        state = inference_result.initial_problem_state
        assert isinstance(state, ConfoundedPredictionState)
        sample = state.sample
        return PredictionSamples(
            predicted_mean=configuration.predicted_mean,
            predicted_variance=configuration.predicted_variance,
            injected_noise=sample.injected_noise,
            parents=sample.parents,
            target=sample.target,
        )


@dataclass(frozen=True)
class PredictionEstimates:
    """Effects of the parents on the target from one run's held-out samples.

    The learned effects are the coefficients of the prediction's mean on the parents. Ordinary
    least squares on the samples gives the confounded effects that reconstruction alone should
    reach. Two-stage least squares on the injected noise gives the reference causal effects.
    """

    learned: tuple[float, ...]
    ordinary_least_squares: tuple[float, ...]
    injected_noise: tuple[float, ...]


def estimate_prediction(samples: PredictionSamples) -> PredictionEstimates:
    """Estimate the parents' effects learned by the prediction map and by the baselines."""
    predicted = np.asarray(samples.predicted_mean).ravel()
    target = np.asarray(samples.target).ravel()
    parents = np.asarray(samples.parents)
    parents = parents.reshape(-1, parents.shape[-1])
    noise = np.asarray(samples.injected_noise)
    noise = noise.reshape(-1, noise.shape[-1])

    def coefficients(values: np.ndarray) -> tuple[float, ...]:
        return tuple(float(value) for value in values)

    return PredictionEstimates(
        learned=coefficients(ordinary_least_squares_coefficients(parents, predicted)),
        ordinary_least_squares=coefficients(ordinary_least_squares_coefficients(parents, target)),
        injected_noise=coefficients(two_stage_least_squares_coefficients(noise, parents, target)),
    )


def run_prediction_estimates(run: Run) -> PredictionEstimates:
    """Return the estimates from one run's held-out telemetry."""
    samples = run.inference_results.telemetries[PredictionTelemetry()]
    assert isinstance(samples, PredictionSamples)
    return estimate_prediction(samples)


def run_predicted_variance(run: Run) -> float:
    """Return the mean predicted variance over one run's held-out samples."""
    samples = run.inference_results.telemetries[PredictionTelemetry()]
    assert isinstance(samples, PredictionSamples)
    return float(np.mean(np.asarray(samples.predicted_variance)))


def variant_label(condition: str, *, purify: bool) -> str:
    """Return the label of a condition's variant with or without purification."""
    return f"{condition}, {'purification' if purify else 'reconstruction'}"


def _condition_runs(runs: Sequence[Run], condition: str, *, purify: bool) -> list[Run]:
    label = variant_label(condition, purify=purify)
    return [run for run in runs if run.variant_label == label]


class CausalMeanPlotter(Plotter):
    """Compare each estimator's error in a condition's effects.

    Ordinary least squares and two-stage least squares on the injected noise use the held-out
    samples of the purification runs. Every variant with the same seed evaluates the same samples.
    """

    _: KW_ONLY
    condition: str = field(static=True)
    effects: tuple[float, ...] = field(static=True)

    @override
    def telemetries(self) -> Telemetries:
        return Telemetries((PredictionTelemetry(),))

    @override
    def plotted_series(self, runs: Sequence[Run]) -> PlottedSeries:
        def estimates(*, purify: bool) -> list[PredictionEstimates]:
            return [
                run_prediction_estimates(run)
                for run in _condition_runs(runs, self.condition, purify=purify)
            ]

        reconstruction = estimates(purify=False)
        purification = estimates(purify=True)
        by_estimator = {
            "OLS": [e.ordinary_least_squares for e in purification],
            "Reconstruction": [e.learned for e in reconstruction],
            "Purification": [e.learned for e in purification],
            "2SLS, noise": [e.injected_noise for e in purification],
        }

        def errors(parent: int) -> list[list[float]]:
            return [
                [abs(values[parent] - self.effects[parent]) for values in seed_values]
                for seed_values in by_estimator.values()
            ]

        n_parents = len(self.effects)
        return bar_chart(
            list(by_estimator),
            {
                f"parent {parent + 1}": (
                    "Effect" if n_parents == 1 else f"Parent {parent + 1}",
                    errors(parent),
                )
                for parent in range(n_parents)
            },
        ) | {
            "true effects": list(self.effects),
            "seed estimates": {
                estimator: [list(values) for values in seed_values]
                for estimator, seed_values in by_estimator.items()
            },
        }


class CausalVariancePlotter(Plotter):
    """Compare the learned predicted variances with the observational and interventional ones.

    The observational variance is the target observation's variance about its best linear
    prediction from the parents. The interventional variance is its variance with the parents set
    by intervention. Both are population values, so they have no spread across seeds.
    """

    _: KW_ONLY
    condition: str = field(static=True)
    scenario: ConfoundedPredictionScenario = field(static=True)

    @override
    def telemetries(self) -> Telemetries:
        return Telemetries((PredictionTelemetry(),))

    @override
    def plotted_series(self, runs: Sequence[Run]) -> PlottedSeries:
        def variances(*, purify: bool) -> list[float]:
            return [
                run_predicted_variance(run)
                for run in _condition_runs(runs, self.condition, purify=purify)
            ]

        reconstruction = variances(purify=False)
        purification = variances(purify=True)
        observational = self.scenario.observational_variance() + FEATURE_OBSERVATION_VARIANCE
        interventional = self.scenario.interventional_variance() + FEATURE_OBSERVATION_VARIANCE
        return bar_chart(
            ["Observational", "Reconstruction", "Purification", "Interventional"],
            {
                "variance": (
                    "Variance",
                    [
                        [observational for _ in purification],
                        reconstruction,
                        purification,
                        [interventional for _ in purification],
                    ],
                )
            },
        ) | {"observational variance": observational, "interventional variance": interventional}


def sgd_gradient_transformations(learning_rate: float) -> DisGradientTransformation:
    """Apply SGD to every learnable parameter, keeping witness multipliers nonnegative.

    Under Adam, the prediction and witnesses keep cycling around the causal prediction even with
    the proportional witness; its per-parameter normalization and momentum defeat the damping.
    """
    return DisGradientTransformation(
        [
            (ParameterType(FixedParameter), None),
            (
                ParameterType(WitnessMultiplierParameter),
                ChainedGradientTransformation[Model](
                    [SGD[Model](learning_rate), KeepParamsNonnegative[Model]()]
                ),
            ),
            (ParameterType(LearnableParameter), SGD[Model](learning_rate)),
        ]
    )


class PurificationSolver(Solver[ConfoundedPredictionProblem]):
    """Train a target's mismatch circuit on fresh samples and evaluate it on held-out ones."""

    _: KW_ONLY
    scenario: ConfoundedPredictionScenario = eqx.field(default_factory=ConfoundedPredictionScenario)
    purify: bool = bool_field(default=True)
    training_examples: int = int_field(default=40_000, domain=IntDistribution(1, 1 << 20, log=True))
    training_batch_size: int = int_field(default=64, domain=IntDistribution(1, 1 << 12, log=True))
    inference_examples: int = int_field(default=16, domain=IntDistribution(1, 1 << 12, log=True))
    inference_batch_size: int = int_field(default=256, domain=IntDistribution(1, 1 << 12, log=True))
    learning_rate: float = float_field(
        default=0.003, domain=FloatDistribution(1e-4, 1.0, log=True), optimize=True
    )

    @override
    def problem(self) -> ConfoundedPredictionProblem:
        return ConfoundedPredictionProblem(self.scenario)

    @override
    def create_model(
        self,
        data_source: DataSource,
        problem: Problem,
        *,
        streams: Mapping[str, RngStream],
    ) -> Model:
        return PurificationModel.create(
            self.scenario.n_parents, purify=self.purify, streams=streams
        )

    @override
    def gradient_transformations(self) -> DisGradientTransformation:
        return sgd_gradient_transformations(self.learning_rate)


ONE_PARENT = "One parent"
TWO_PARENTS = "Two parents"
PURIFICATION_SCENARIOS = {
    ONE_PARENT: ConfoundedPredictionScenario(),
    TWO_PARENTS: ConfoundedPredictionScenario(effects=(1.5, -1.0)),
}
_CONDITION_SLUGS = {ONE_PARENT: "one-parent", TWO_PARENTS: "two-parents"}


def _purification_variants(solver: PurificationSolver) -> list[Variant]:
    """Return each condition's variants with reconstruction alone and with purification."""
    return [
        Variant(
            label=variant_label(condition, purify=purify),
            solver=replace(solver, scenario=scenario, purify=purify),
        )
        for condition, scenario in PURIFICATION_SCENARIOS.items()
        for purify in (False, True)
    ]


class CausalMeanExperiment(SeededExperiment):
    """Test 4: reconstruction reaches OLS, and purification reaches the causal effect.

    Each condition trains a target's mismatch circuit with reconstruction alone and with
    purification. One confounder drives the parents and the target, and each parent's injected
    noise is its instrument. The two-parent condition needs both instruments to separate the
    effects.
    """

    def __init__(
        self, solver: PurificationSolver | None = None, *, seeds: int = SEED_COUNT
    ) -> None:
        super().__init__(
            name="sip-causal-mean",
            variants=_purification_variants(PurificationSolver() if solver is None else solver),
            plotters=[
                CausalMeanPlotter(
                    name=f"causal-mean-{_CONDITION_SLUGS[condition]}",
                    title=f"Effect errors on held-out samples: {condition.lower()}",
                    condition=condition,
                    effects=scenario.effects,
                )
                for condition, scenario in PURIFICATION_SCENARIOS.items()
            ],
            sweep=[{"seed": seed} for seed in range(seeds)],
        )


# The variance converges more slowly than the mean.
_VARIANCE_TRAINING_EXAMPLES = 80_000


class CausalVarianceExperiment(SeededExperiment):
    """Test 5: purification makes the predicted variance match the interventional variance.

    The confounder enters the target additively, so the target's deviation from its causal mean is
    independent of the instruments. The interventional variance then satisfies the circuit's
    second-moment restriction. Reconstruction alone learns the smaller observational variance,
    because the parents carry information about the confounder.
    """

    def __init__(
        self, solver: PurificationSolver | None = None, *, seeds: int = SEED_COUNT
    ) -> None:
        solver = (
            PurificationSolver(training_examples=_VARIANCE_TRAINING_EXAMPLES)
            if solver is None
            else solver
        )
        super().__init__(
            name="sip-causal-variance",
            variants=_purification_variants(solver),
            plotters=[
                CausalVariancePlotter(
                    name=f"causal-variance-{_CONDITION_SLUGS[condition]}",
                    title=f"Predicted variance on held-out samples: {condition.lower()}",
                    condition=condition,
                    scenario=scenario,
                )
                for condition, scenario in PURIFICATION_SCENARIOS.items()
            ],
            sweep=[{"seed": seed} for seed in range(seeds)],
        )


sip_causal_mean_experiment = CausalMeanExperiment()
sip_causal_variance_experiment = CausalVarianceExperiment()
