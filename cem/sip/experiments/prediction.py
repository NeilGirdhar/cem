"""Prediction experiments: whether purification makes prediction maps learn causal effects.

See `sip.md`, tests 4–8.
"""

from collections.abc import Callable, Mapping, Sequence
from dataclasses import KW_ONLY, asdict, dataclass, replace
from typing import Any, Self, override

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from efax import NormalNP, NormalPP, NormalVP, parameter_concat
from jax.nn import softplus
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
from cem.framework.solution import (
    Inference,
    InferenceResult,
    Telemetries,
    Telemetry,
    TrainingResult,
    TrainingSolution,
)
from cem.framework.solver import Solver, bool_field, float_field, int_field
from cem.npn.gaussian import MINIMUM_VARIANCE, UNIT_SOFTPLUS_RAW
from cem.sip.mismatch import Mismatch
from cem.sip.parameters import WitnessMultiplierParameter
from cem.sip.settings import MismatchSettings
from cem.sip.witness import (
    IntegralWitnessMap,
    ProportionalWitnessMap,
    integral_witness_objective,
    witness_objectives,
    witness_residuals,
)

from .baselines import (
    ordinary_least_squares_coefficients,
    polynomial_basis,
    projected_root_mean_square,
    sieve_two_stage_least_squares_coefficients,
    two_stage_least_squares_coefficients,
)
from .common import SEED_COUNT, SeededExperiment, bar_chart, reference_seed, seed_runs
from .datasets import (
    FEATURE_OBSERVATION_VARIANCE,
    ConfoundedPredictionObservation,
    ConfoundedPredictionProblem,
    ConfoundedPredictionState,
    EndogenousGainObservation,
    EndogenousGainProblem,
    EndogenousGainState,
)
from .scenarios import (
    ConfoundedPredictionSample,
    ConfoundedPredictionScenario,
    EndogenousGainScenario,
)

_TARGET_NODE = "target"


class PredictionConfiguration(NodeConfiguration):
    """The mean and variance of the target's prediction."""

    predicted_mean: JaxRealArray  # scalar
    predicted_variance: JaxRealArray  # scalar


class PurificationModel(Model):
    """A target's mismatch circuit, trained with or without purification.

    With purification, the circuit's full loss trains the prediction map, including the
    confounding objective from the witnesses over the parents' instruments. Without it, only the
    reconstruction loss trains the prediction map.

    Turning off the integral term removes the integral witness's correction from the prediction's
    update, which leaves proportional-only purification. The integral witness and its multiplier
    still train on their own objectives, so every variant builds and updates the same modules.
    """

    mismatch: Mismatch
    purify: bool = eqx.field(static=True)
    integral: bool = eqx.field(static=True, default=True)

    @classmethod
    def create(
        cls,
        n_parents: int,
        *,
        purify: bool,
        mismatch_settings: MismatchSettings,
        streams: Mapping[str, RngStream],
        integral: bool = True,
    ) -> Self:
        mismatch = Mismatch.create(
            1,
            predictor_features=n_parents,
            gain_features=0,
            mismatch_settings=mismatch_settings,
            streams=streams,
        )
        return cls(mismatch, purify=purify, integral=integral)

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
            jax.lax.stop_gradient(observation.predictor.observation)
        ).to_variance_parametrization()
        loss = outputs.loss if self.purify else outputs.reconstruction_loss
        if self.purify and not self.integral:
            loss -= self._integral_correction(
                observation, prediction, streams=streams, inference=inference
            )
        return ModelResult(
            loss=loss,
            configurations=frozendict(
                {
                    _TARGET_NODE: PredictionConfiguration(
                        predicted_mean=prediction.mean[0], predicted_variance=prediction.variance[0]
                    )
                }
            ),
            state=state,
        )

    def _integral_correction(
        self,
        observation: ConfoundedPredictionObservation,
        prediction: NormalVP,
        *,
        streams: Mapping[str, RngStream],
        inference: bool,
    ) -> JaxRealArray:
        """Return the confounding objective's integral term, through which the prediction learns.

        The witness is held fixed, so subtracting this term changes only the prediction's update.
        """
        target = jax.lax.stop_gradient(observation.target_observation.to_variance_parametrization())
        witness = self.mismatch.integral_witness_map.infer(
            observation.predictor.instrument, streams=streams, inference=inference
        )
        return integral_witness_objective(
            witness_residuals(target, prediction), jax.lax.stop_gradient(witness), None
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


def _run_samples(run: Run) -> PredictionSamples:
    samples = run.inference_results.telemetries[PredictionTelemetry()]
    assert isinstance(samples, PredictionSamples)
    return samples


def run_prediction_estimates(run: Run) -> PredictionEstimates:
    """Return the estimates from one run's held-out telemetry."""
    return estimate_prediction(_run_samples(run))


def run_predicted_variance(run: Run) -> float:
    """Return the mean predicted variance over one run's held-out samples."""
    return float(np.mean(np.asarray(_run_samples(run).predicted_variance)))


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


# Polynomial degrees for the sieve reference and the projected error. The instrument basis has more
# functions than the regressor basis so that the sieve's first stage is overidentified.
_INSTRUMENT_DEGREE = 7
_REGRESSOR_DEGREE = 5
# The sieve's fit sample has as many examples as the default held-out sample.
SIEVE_FIT_EXAMPLES = 4096


def sieve_fit_sample(
    scenario: ConfoundedPredictionScenario, seed: int
) -> ConfoundedPredictionSample:
    """Return the sample that fits sieve 2SLS for one experiment seed.

    It is drawn from the scenario independently of the seed's training and held-out samples.
    """
    return scenario.samples(jr.key(reference_seed(seed)), SIEVE_FIT_EXAMPLES)


def fit_sieve(sample: ConfoundedPredictionSample) -> np.ndarray:
    """Return the sieve 2SLS coefficients of the structural function on the parent's powers.

    The structural function is a polynomial in the parent, instrumented by a polynomial in the
    parent's noise. The sample must have one parent.
    """
    parent = np.asarray(sample.parents)
    assert parent.shape[-1] == 1
    return sieve_two_stage_least_squares_coefficients(
        polynomial_basis(np.asarray(sample.injected_noise).ravel(), _INSTRUMENT_DEGREE),
        polynomial_basis(parent.ravel(), _REGRESSOR_DEGREE),
        np.asarray(sample.target).ravel(),
    )


@dataclass(frozen=True)
class NonlinearEstimates:
    """Errors of fitted structural functions on one run's held-out samples.

    Each error compares a function of the parent with the structural function. The root-mean-square
    error measures the whole difference. The projected error measures only its conditional mean
    given the injected noise, which is the part the instrument can detect.
    """

    learned: float
    learned_projected: float
    observational: float
    observational_projected: float
    sieve: float
    sieve_projected: float


def estimate_nonlinear(
    samples: PredictionSamples,
    scenario: ConfoundedPredictionScenario,
    sieve_coefficients: np.ndarray,
) -> NonlinearEstimates:
    """Score the learned, observational, and sieve 2SLS functions against the structural function.

    Every function is scored on the held-out samples, and the projected error uses the same
    instrument basis for each. The observational function is the target's population mean given
    the parent. The sieve coefficients come from `fit_sieve` on a separate sample, so the held-out
    targets play no part in the scores. The scenario must have one parent.
    """
    assert scenario.n_parents == 1
    parents = np.asarray(samples.parents).reshape(-1, 1)
    noise = np.asarray(samples.injected_noise).ravel()
    structural = np.asarray(scenario.structural_mean(jnp.asarray(parents)))
    instrument_basis = polynomial_basis(noise, _INSTRUMENT_DEGREE)

    def errors(fitted: np.ndarray) -> tuple[float, float]:
        difference = fitted - structural
        return (
            float(np.sqrt(np.mean(np.square(difference)))),
            projected_root_mean_square(difference, instrument_basis),
        )

    learned = errors(np.asarray(samples.predicted_mean).ravel())
    observational = errors(np.asarray(scenario.observational_mean(jnp.asarray(parents))))
    sieve = errors(polynomial_basis(parents[:, 0], _REGRESSOR_DEGREE) @ sieve_coefficients)
    return NonlinearEstimates(*learned, *observational, *sieve)


def learned_curve(samples: PredictionSamples, grid: np.ndarray) -> np.ndarray:
    """Return the predicted mean at each grid value of the parent.

    The prediction's mean is a function of the parent alone, so interpolating the held-out samples
    sorted by the parent traces it. Grid values beyond the samples take the nearest sample's value.
    """
    parent = np.asarray(samples.parents).ravel()
    predicted = np.asarray(samples.predicted_mean).ravel()
    order = np.argsort(parent)
    return np.interp(grid, parent[order], predicted[order])


class NonlinearErrorsPlotter(Plotter):
    """Compare each function's error and projected error against the structural function.

    Every function is scored on the held-out samples of the purification runs, which the
    reconstruction run with the same seed shares. Sieve 2SLS is fitted on each seed's
    `sieve_fit_sample`.
    """

    _: KW_ONLY
    condition: str = field(static=True)
    scenario: ConfoundedPredictionScenario = field(static=True)

    @override
    def telemetries(self) -> Telemetries:
        return Telemetries((PredictionTelemetry(),))

    @override
    def plotted_series(self, runs: Sequence[Run]) -> PlottedSeries:
        def estimates(*, purify: bool) -> list[NonlinearEstimates]:
            return [
                estimate_nonlinear(
                    _run_samples(run),
                    self.scenario,
                    fit_sieve(sieve_fit_sample(self.scenario, run.sweep_entry["seed"])),
                )
                for run in _condition_runs(runs, self.condition, purify=purify)
            ]

        reconstruction = estimates(purify=False)
        purification = estimates(purify=True)
        (evaluation_examples,) = {
            np.asarray(_run_samples(run).target).size
            for run in _condition_runs(runs, self.condition, purify=True)
        }
        return bar_chart(
            ["Observational", "Reconstruction", "Purification", "Sieve 2SLS"],
            {
                "error": (
                    "RMSE",
                    [
                        [e.observational for e in purification],
                        [e.learned for e in reconstruction],
                        [e.learned for e in purification],
                        [e.sieve for e in purification],
                    ],
                ),
                "projected": (
                    "Projected RMSE",
                    [
                        [e.observational_projected for e in purification],
                        [e.learned_projected for e in reconstruction],
                        [e.learned_projected for e in purification],
                        [e.sieve_projected for e in purification],
                    ],
                ),
            },
        ) | {
            "evaluation examples": evaluation_examples,
            "sieve fit examples": SIEVE_FIT_EXAMPLES,
        }


class NonlinearCurvesPlotter(Plotter):
    """Plot the learned predicted means against the structural and observational functions.

    Each learned curve is the mean across seeds of the curves traced by the held-out samples.
    """

    _: KW_ONLY
    condition: str = field(static=True)
    scenario: ConfoundedPredictionScenario = field(static=True)
    grid: tuple[float, ...] = field(
        static=True, default=tuple(float(x) for x in np.linspace(-3.0, 3.0, 61))
    )

    @override
    def telemetries(self) -> Telemetries:
        return Telemetries((PredictionTelemetry(),))

    @override
    def plotted_series(self, runs: Sequence[Run]) -> PlottedSeries:
        grid = np.asarray(self.grid)
        parents = grid[:, np.newaxis]

        def mean_curve(*, purify: bool) -> list[float]:
            curves = [
                learned_curve(_run_samples(run), grid)
                for run in _condition_runs(runs, self.condition, purify=purify)
            ]
            return [float(value) for value in np.mean(curves, axis=0)]

        return {
            "iteration": [float(value) for value in grid],
            "line plots": {
                "structural": "Structural",
                "observational": "Observational",
                "reconstruction": "Reconstruction",
                "purification": "Purification",
            },
            "line styles": {"structural": "dashed", "observational": "dashed"},
            "structural": [
                float(v) for v in np.asarray(self.scenario.structural_mean(jnp.asarray(parents)))
            ],
            "observational": [
                float(v) for v in np.asarray(self.scenario.observational_mean(jnp.asarray(parents)))
            ],
            "reconstruction": mean_curve(purify=False),
            "purification": mean_curve(purify=True),
        }


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
    integral: bool = bool_field(default=True)
    mismatch_settings: MismatchSettings = eqx.field(static=True, default_factory=MismatchSettings)
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
            self.scenario.n_parents,
            purify=self.purify,
            mismatch_settings=self.mismatch_settings,
            streams=streams,
            integral=self.integral,
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


def _probe(parent: float) -> NormalNP:
    """Return a one-parent observation at `parent`, as precise as the training observations."""
    value = jnp.asarray([parent])
    return NormalPP(value, jnp.full_like(value, 1.0 / FEATURE_OBSERVATION_VARIANCE)).to_nat()


class EffectTelemetry(Telemetry):
    """Record a one-parent purification model's effect after every training update.

    The effect is the difference between the predicted means at the parent values one and zero.
    For an affine prediction map, it is the slope. Other models record nothing.
    """

    @override
    def training_snapshot(
        self,
        training_solution: TrainingSolution,
        training_result: TrainingResult,
        snapshots: Mapping[Telemetry, Any],
    ) -> JaxRealArray | None:
        model = training_solution.inference.assemble_model(
            training_result.solution_state.dis_learnable_parameters.assembled()
        )
        if not isinstance(model, PurificationModel):
            return None
        one, zero = (
            model.mismatch.prediction_map.infer(_probe(parent))
            .to_variance_parametrization()
            .mean[0]
            for parent in (1.0, 0.0)
        )
        return one - zero


def fixed_mean_variance(
    offset: JaxRealArray, slope: JaxRealArray, squared_parent: JaxRealArray
) -> JaxRealArray:
    """Return the variance softplus(b + c y²), floored like every predicted variance.

    Args:
        offset: The offset b.
        slope: The coefficient c of the parent's square.
        squared_parent: The parent's square y².
    """
    return jnp.maximum(softplus(offset + slope * squared_parent), MINIMUM_VARIANCE)


# Selects the variance coordinate of the one channel's (mean, variance) witness pair.
_VARIANCE_ONLY = jnp.asarray([[0.0, 1.0]])


class FixedMeanVarianceModel(Model):
    """Predict a one-parent target with its mean fixed at the causal mean and a learned variance.

    With the mean fixed, the variance update is the only difference between reconstruction alone
    and purification. The variance, softplus(b + c y²), can be constant or grow with the parent's
    square. Reconstruction should learn the target's observational squared deviation from the
    causal mean, which grows with y² because the parent carries information about the confounder.
    Purification should learn the constant interventional variance.

    The witnesses see only the variance residual, so the mean residual cannot use the integral
    witness's norm budget. Each witness needs a hidden layer to detect the residual's even
    dependence on the instrument.
    """

    variance_offset: LearnableParameter[JaxRealArray]  # scalar b
    variance_slope: LearnableParameter[JaxRealArray]  # scalar c
    integral_witness_map: IntegralWitnessMap
    proportional_witness_map: ProportionalWitnessMap
    witness_multiplier: WitnessMultiplierParameter  # scalar
    damping_weight: FixedParameter[JaxRealArray]  # scalar
    effect: float = eqx.field(static=True)
    purify: bool = eqx.field(static=True)

    @classmethod
    def create(
        cls,
        *,
        effect: float,
        purify: bool,
        mismatch_settings: MismatchSettings,
        streams: Mapping[str, RngStream],
    ) -> Self:
        """Start from a unit constant variance, the same in every run."""
        return cls(
            variance_offset=LearnableParameter(jnp.asarray(UNIT_SOFTPLUS_RAW)),
            variance_slope=LearnableParameter(jnp.zeros(())),
            integral_witness_map=IntegralWitnessMap.create(
                1,
                instrument_features=1,
                hidden_features=mismatch_settings.integral_witness_hidden_features,
                streams=streams,
            ),
            proportional_witness_map=ProportionalWitnessMap.create(
                1,
                instrument_features=1,
                hidden_features=mismatch_settings.proportional_witness_hidden_features,
                streams=streams,
            ),
            witness_multiplier=WitnessMultiplierParameter(jnp.zeros(())),
            damping_weight=FixedParameter(jnp.asarray(mismatch_settings.damping_weight)),
            effect=effect,
            purify=purify,
        )

    def variance(self, squared_parent: JaxRealArray) -> JaxRealArray:
        """Return the predicted variance at the parent's square."""
        return fixed_mean_variance(
            self.variance_offset.value, self.variance_slope.value, squared_parent
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
        assert isinstance(observation, ConfoundedPredictionObservation)
        parent = observation.predictor.observation.to_variance_parametrization().mean
        prediction = NormalVP(self.effect * parent, self.variance(jnp.square(parent)))
        target = jax.lax.stop_gradient(observation.target_observation.to_variance_parametrization())
        loss = jnp.sum(target.cross_entropy(prediction))
        if self.purify:
            instrument = observation.predictor.instrument
            objectives = witness_objectives(
                _VARIANCE_ONLY * witness_residuals(target, prediction),
                integral_witness=_VARIANCE_ONLY
                * self.integral_witness_map.infer(instrument, streams=streams, inference=inference),
                proportional_witness=_VARIANCE_ONLY
                * self.proportional_witness_map.infer(
                    instrument, streams=streams, inference=inference
                ),
                gain=None,
                multiplier=self.witness_multiplier.value,
                damping_weight=self.damping_weight.value,
            )
            loss += objectives.loss
        return ModelResult(
            loss=loss,
            configurations=frozendict(
                {
                    _TARGET_NODE: PredictionConfiguration(
                        predicted_mean=prediction.mean[0], predicted_variance=prediction.variance[0]
                    )
                }
            ),
            state=state,
        )


class VarianceTelemetry(Telemetry):
    """Record the fixed-mean variance model's offset and slope after every training update.

    Other models record nothing.
    """

    @override
    def training_snapshot(
        self,
        training_solution: TrainingSolution,
        training_result: TrainingResult,
        snapshots: Mapping[Telemetry, Any],
    ) -> JaxRealArray | None:
        model = training_solution.inference.assemble_model(
            training_result.solution_state.dis_learnable_parameters.assembled()
        )
        if not isinstance(model, FixedMeanVarianceModel):
            return None
        return jnp.stack([model.variance_offset.value, model.variance_slope.value])


class FixedMeanVarianceSolver(PurificationSolver):
    """Train the fixed-mean variance model on a linear one-parent scenario."""

    @override
    def create_model(
        self,
        data_source: DataSource,
        problem: Problem,
        *,
        streams: Mapping[str, RngStream],
    ) -> Model:
        assert self.scenario.n_parents == 1
        assert not self.scenario.nonlinear
        return FixedMeanVarianceModel.create(
            effect=self.scenario.effects[0],
            purify=self.purify,
            mismatch_settings=self.mismatch_settings,
            streams=streams,
        )


def _late_window(trajectory: np.ndarray) -> np.ndarray:
    """Return the last quarter of a trajectory over training."""
    return trajectory[-max(1, len(trajectory) // 4) :]


class EffectTrajectoriesPlotter(Plotter):
    """Plot each seed's effect estimate over training for each purification arm.

    Each seed has its own line, so averaging cannot hide oscillations that are out of phase across
    seeds. The late-training metrics use every update in the last quarter of training: the bias,
    the mean estimate minus the true effect, measures where an arm settles, and the standard
    deviation measures how much it still moves.
    """

    _: KW_ONLY
    arms: tuple[str, ...] = field(static=True)
    effect: float = field(static=True)
    points: int = field(static=True, default=200)

    @override
    def telemetries(self) -> Telemetries:
        return Telemetries((EffectTelemetry(),))

    @override
    def plotted_series(self, runs: Sequence[Run]) -> PlottedSeries:
        trajectories = {
            arm: [
                np.asarray(run.training_results.telemetries[EffectTelemetry()]).ravel()
                for run in seed_runs(runs, arm)
            ]
            for arm in self.arms
        }
        updates = len(next(iter(trajectories.values()))[0])
        shown = np.unique(np.linspace(0, updates - 1, self.points).round().astype(int))
        lines = {
            f"{arm} {seed}": [float(value) for value in trajectory[shown]]
            for arm, arm_trajectories in trajectories.items()
            for seed, trajectory in enumerate(arm_trajectories)
        }
        return {
            "iteration": [int(index) + 1 for index in shown],
            "line plots": {key: key for key in lines} | {"true effect": "True effect"},
            "line styles": {"true effect": "dashed"},
            **lines,
            "true effect": [self.effect for _ in shown],
            "late bias": {
                arm: [float(np.mean(_late_window(t)) - self.effect) for t in arm_trajectories]
                for arm, arm_trajectories in trajectories.items()
            },
            "late variability": {
                arm: [float(np.std(_late_window(t))) for t in arm_trajectories]
                for arm, arm_trajectories in trajectories.items()
            },
            "updates": updates,
        }


class VarianceCurvesPlotter(Plotter):
    """Plot the fixed-mean variance curves under intervention against both references.

    Each curve is the final variance model evaluated at interventions on the parent, averaged
    across seeds. The interventional reference is the observation variance with the parent set by
    intervention. The observational reference is the observation's mean squared deviation from the
    causal mean given the observed parent. The errors use a uniform grid because the observational
    reference also averages to the interventional variance over the parent's distribution.
    """

    _: KW_ONLY
    reconstruction: str = field(static=True)
    purification: str = field(static=True)
    scenario: ConfoundedPredictionScenario = field(static=True)
    grid: tuple[float, ...] = field(
        static=True, default=tuple(float(x) for x in np.linspace(-3.0, 3.0, 61))
    )

    @override
    def telemetries(self) -> Telemetries:
        return Telemetries((VarianceTelemetry(),))

    @override
    def plotted_series(self, runs: Sequence[Run]) -> PlottedSeries:
        grid = np.asarray(self.grid)
        interventional = self.scenario.interventional_variance() + FEATURE_OBSERVATION_VARIANCE
        observational = (
            np.asarray(self.scenario.causal_residual_second_moment(jnp.asarray(grid[:, None])))
            + FEATURE_OBSERVATION_VARIANCE
        )

        def final_parameters(label: str) -> list[np.ndarray]:
            return [
                np.asarray(run.training_results.telemetries[VarianceTelemetry()])[-1]
                for run in seed_runs(runs, label)
            ]

        def curves(label: str) -> list[np.ndarray]:
            return [
                np.asarray(fixed_mean_variance(offset, slope, jnp.asarray(np.square(grid))))
                for offset, slope in final_parameters(label)
            ]

        arms = {"reconstruction": self.reconstruction, "purification": self.purification}
        arm_curves = {key: curves(label) for key, label in arms.items()}
        return {
            "iteration": [float(value) for value in grid],
            "line plots": {
                "interventional": "Interventional",
                "observational": "Observational",
                "reconstruction": "Reconstruction",
                "purification": "Purification",
            },
            "line styles": {"interventional": "dashed", "observational": "dashed"},
            "interventional": [interventional for _ in grid],
            "observational": [float(value) for value in observational],
            **{
                key: [float(value) for value in np.mean(seed_curves, axis=0)]
                for key, seed_curves in arm_curves.items()
            },
            "grid error": {
                key: [
                    float(np.sqrt(np.mean(np.square(curve - interventional))))
                    for curve in seed_curves
                ]
                for key, seed_curves in arm_curves.items()
            },
            "slope": {
                key: [float(slope) for _, slope in final_parameters(label)]
                for key, label in arms.items()
            },
            "interventional variance": interventional,
        }


INTEGRAL_ONLY = "Integral only"
PROPORTIONAL_ONLY = "Proportional only"
INTEGRAL_AND_PROPORTIONAL = "Integral and proportional"
DAMPING_ARMS = (INTEGRAL_ONLY, PROPORTIONAL_ONLY, INTEGRAL_AND_PROPORTIONAL)
FIXED_MEAN_RECONSTRUCTION = "Fixed mean, reconstruction"
FIXED_MEAN_PURIFICATION = "Fixed mean, purification"
# The witnesses need a hidden layer to detect the variance residual's even dependence on the
# instrument.
VARIANCE_MISMATCH_SETTINGS = MismatchSettings(
    integral_witness_hidden_features=8, proportional_witness_hidden_features=8
)


def _damping_variants(solver: PurificationSolver) -> list[Variant]:
    """Return the three purification arms, which differ only in the prediction's update."""
    settings = solver.mismatch_settings
    return [
        Variant(
            label=INTEGRAL_ONLY,
            solver=replace(solver, mismatch_settings=replace(settings, damping_weight=0.0)),
        ),
        Variant(label=PROPORTIONAL_ONLY, solver=replace(solver, integral=False)),
        Variant(label=INTEGRAL_AND_PROPORTIONAL, solver=solver),
    ]


class WitnessDiagnosticsExperiment(SeededExperiment):
    """Test 6: what the proportional witness and the variance residual each contribute.

    The damping comparison trains three purification arms on the one-parent scenario of test 4:
    integral only, proportional only, and both. Integral only can keep oscillating about the
    causal effect, and proportional only acts as a finite quadratic penalty that settles short of
    it. The variance comparison fixes the mean at the causal mean, so reconstruction alone and
    purification differ only in the variance update. Within each comparison, runs with the same
    seed share initialization and training samples.
    """

    def __init__(
        self,
        damping_solver: PurificationSolver | None = None,
        variance_solver: FixedMeanVarianceSolver | None = None,
        *,
        seeds: int = SEED_COUNT,
    ) -> None:
        damping_solver = PurificationSolver() if damping_solver is None else damping_solver
        variance_solver = (
            FixedMeanVarianceSolver(
                mismatch_settings=VARIANCE_MISMATCH_SETTINGS,
                training_examples=_VARIANCE_TRAINING_EXAMPLES,
            )
            if variance_solver is None
            else variance_solver
        )
        super().__init__(
            name="sip-witness-diagnostics",
            variants=[
                *_damping_variants(damping_solver),
                Variant(
                    label=FIXED_MEAN_RECONSTRUCTION,
                    solver=replace(variance_solver, purify=False),
                ),
                Variant(
                    label=FIXED_MEAN_PURIFICATION, solver=replace(variance_solver, purify=True)
                ),
            ],
            plotters=[
                EffectTrajectoriesPlotter(
                    name="witness-diagnostics-damping",
                    title="Effect estimates over training",
                    arms=DAMPING_ARMS,
                    effect=damping_solver.scenario.effects[0],
                ),
                VarianceCurvesPlotter(
                    name="witness-diagnostics-variance",
                    title="Predicted variance under intervention with the mean fixed",
                    reconstruction=FIXED_MEAN_RECONSTRUCTION,
                    purification=FIXED_MEAN_PURIFICATION,
                    scenario=variance_solver.scenario,
                ),
            ],
            sweep=[{"seed": seed} for seed in range(seeds)],
        )


NONLINEAR_CONDITION = "Tanh effect"
NONLINEAR_SCENARIO = ConfoundedPredictionScenario(effects=(2.0,), nonlinear=True)
# A nonlinear function needs hidden layers in the prediction map, and the witnesses need them to
# detect nonlinear violations of the restriction.
NONLINEAR_MISMATCH_SETTINGS = MismatchSettings(
    prediction_hidden_features=8,
    integral_witness_hidden_features=8,
    proportional_witness_hidden_features=8,
)
# With larger steps, the fitted function keeps steepening in the tails, where the instrument
# barely constrains it.
_NONLINEAR_LEARNING_RATE = 0.001
_NONLINEAR_TRAINING_EXAMPLES = 160_000


class NonlinearEffectExperiment(SeededExperiment):
    """Test 7: purification matches a nonlinear structural function that the instrument identifies.

    The target is 2 tanh of its confounded parent. The instrument sees the structural function
    only through its conditional mean given the parent's noise, which smooths it, so the errors
    are reported both in full and projected onto the instrument.
    """

    def __init__(
        self, solver: PurificationSolver | None = None, *, seeds: int = SEED_COUNT
    ) -> None:
        solver = (
            PurificationSolver(
                scenario=NONLINEAR_SCENARIO,
                mismatch_settings=NONLINEAR_MISMATCH_SETTINGS,
                training_examples=_NONLINEAR_TRAINING_EXAMPLES,
                learning_rate=_NONLINEAR_LEARNING_RATE,
            )
            if solver is None
            else solver
        )
        super().__init__(
            name="sip-nonlinear-effect",
            variants=[
                Variant(
                    label=variant_label(NONLINEAR_CONDITION, purify=purify),
                    solver=replace(solver, purify=purify),
                )
                for purify in (False, True)
            ],
            plotters=[
                NonlinearErrorsPlotter(
                    name="nonlinear-effect-errors",
                    title="Structural function errors on held-out samples",
                    condition=NONLINEAR_CONDITION,
                    scenario=solver.scenario,
                ),
                NonlinearCurvesPlotter(
                    name="nonlinear-effect-curves",
                    title="Learned and structural functions",
                    condition=NONLINEAR_CONDITION,
                    scenario=solver.scenario,
                ),
            ],
            sweep=[{"seed": seed} for seed in range(seeds)],
        )


class GainPredictionConfiguration(NodeConfiguration):
    """The target's predicted moments and both witnesses' outputs for one example."""

    predicted_mean: JaxRealArray  # scalar
    predicted_variance: JaxRealArray  # scalar
    integral_witness: JaxRealArray  # (2,), mean and variance coordinates
    proportional_witness: JaxRealArray  # (2,), mean and variance coordinates


class EndogenousGainModel(Model):
    """A target's full mismatch circuit with a gain parent.

    The prediction map reads the parent's observation and fits one slope, one intercept, and one
    variance. The observed gain weights learning, and both witnesses read the parent's instrument
    and the gain instrument.
    """

    mismatch: Mismatch

    @classmethod
    def create(
        cls, *, mismatch_settings: MismatchSettings, streams: Mapping[str, RngStream]
    ) -> Self:
        return cls(
            Mismatch.create(
                1,
                predictor_features=1,
                gain_features=1,
                mismatch_settings=mismatch_settings,
                streams=streams,
            )
        )

    def prediction(self, parent: NormalNP) -> NormalVP:
        """Return the prediction from the parent's observation."""
        return self.mismatch.prediction_map.infer(parent).to_variance_parametrization()

    @override
    def infer(
        self,
        observation: object,
        state: object,
        *,
        streams: Mapping[str, RngStream],
        inference: bool,
    ) -> ModelResult:
        assert isinstance(observation, EndogenousGainObservation)
        outputs = self.mismatch.step(
            raw_observation=observation.target_observation,
            local_variance=jnp.zeros(1),
            predictor=observation.predictor,
            gain=observation.gain,
            streams=streams,
            inference=inference,
        )
        instruments = parameter_concat(
            [observation.predictor.instrument, observation.gain.instrument]
        )
        prediction = self.prediction(observation.predictor.observation)
        return ModelResult(
            loss=outputs.loss,
            configurations=frozendict(
                {
                    _TARGET_NODE: GainPredictionConfiguration(
                        predicted_mean=prediction.mean[0],
                        predicted_variance=prediction.variance[0],
                        integral_witness=self.mismatch.integral_witness_map.infer(
                            instruments, streams=streams, inference=inference
                        )[0],
                        proportional_witness=self.mismatch.proportional_witness_map.infer(
                            instruments, streams=streams, inference=inference
                        )[0],
                    )
                }
            ),
            state=state,
        )


class GainFitTelemetry(Telemetry):
    """Record the endogenous-gain model's slope, intercept, and variance after every update.

    The snapshot holds the predicted means at the parent values zero and one, and the predicted
    variance, which is constant. Other models record nothing.
    """

    @override
    def training_snapshot(
        self,
        training_solution: TrainingSolution,
        training_result: TrainingResult,
        snapshots: Mapping[Telemetry, Any],
    ) -> JaxRealArray | None:
        model = training_solution.inference.assemble_model(
            training_result.solution_state.dis_learnable_parameters.assembled()
        )
        if not isinstance(model, EndogenousGainModel):
            return None
        at_zero, at_one = (model.prediction(_probe(parent)) for parent in (0.0, 1.0))
        return jnp.stack([at_zero.mean[0], at_one.mean[0], at_zero.variance[0]])


class GainWitnessSamples(eqx.Module):
    """Held-out instrument draws, gains, and both witnesses' outputs."""

    parent_noise: JaxRealArray  # (..., batch)
    gain_noise: JaxRealArray  # (..., batch)
    integral_witness: JaxRealArray  # (..., batch, 2)
    proportional_witness: JaxRealArray  # (..., batch, 2)


class GainWitnessTelemetry(Telemetry):
    """Record every held-out example's instruments and witness outputs."""

    @override
    def inference_snapshot(
        self,
        inference: Inference,
        inference_result: InferenceResult,
        snapshots: Mapping[Telemetry, Any],
    ) -> GainWitnessSamples | None:
        configuration = inference_result.model_configuration[_TARGET_NODE]
        if not isinstance(configuration, GainPredictionConfiguration):
            return None
        state = inference_result.initial_problem_state
        assert isinstance(state, EndogenousGainState)
        return GainWitnessSamples(
            parent_noise=state.sample.prediction.injected_noise[..., 0],
            gain_noise=state.sample.gain_noise,
            integral_witness=configuration.integral_witness,
            proportional_witness=configuration.proportional_witness,
        )


@dataclass(frozen=True)
class GainFit:
    """A run's final slope, intercept, and variance."""

    slope: float
    intercept: float
    variance: float

    @classmethod
    def from_run(cls, run: Run) -> Self:
        at_zero, at_one, variance = np.asarray(
            run.training_results.telemetries[GainFitTelemetry()]
        )[-1]
        return cls(
            slope=float(at_one - at_zero), intercept=float(at_zero), variance=float(variance)
        )


@dataclass(frozen=True)
class GainScores:
    """A run's fit and the population moments at it, over its held-out instrument draws.

    Each moment is a root mean square over the draws. The proportional witness estimates each
    gain-weighted moment divided by the mean gain, so each correlation compares the learned
    proportional witness with that population target. The integral score is the population value
    of the integral witness objective, the mean over draws of each moment times the integral
    witness.
    """

    fit: GainFit
    mean_moment: float
    variance_moment: float
    mean_correlation: float
    variance_correlation: float
    mean_integral_score: float
    variance_integral_score: float


def score_gain_run(run: Run, scenario: EndogenousGainScenario) -> GainScores:
    """Evaluate the population moments at a run's fit on its held-out instrument draws."""
    fit = GainFit.from_run(run)
    samples = run.inference_results.telemetries[GainWitnessTelemetry()]
    assert isinstance(samples, GainWitnessSamples)
    moments = scenario.population_moments(
        slope=fit.slope,
        intercept=fit.intercept,
        variance=fit.variance,
        observation_variance=FEATURE_OBSERVATION_VARIANCE,
        parent_noise=jnp.asarray(samples.parent_noise).ravel(),
        gain_noise=jnp.asarray(samples.gain_noise).ravel(),
    )
    integral = np.asarray(samples.integral_witness).reshape(-1, 2)
    proportional = np.asarray(samples.proportional_witness).reshape(-1, 2)
    gain = np.asarray(moments.gain)

    def correlation(witness: np.ndarray, moment: np.ndarray) -> float:
        target = moment / gain
        if np.std(target) == 0 or np.std(witness) == 0:
            return float("nan")
        return float(np.corrcoef(witness, target)[0, 1])

    by_coordinate = [np.asarray(moments.mean), np.asarray(moments.variance)]
    return GainScores(
        fit=fit,
        mean_moment=float(np.sqrt(np.mean(np.square(by_coordinate[0])))),
        variance_moment=float(np.sqrt(np.mean(np.square(by_coordinate[1])))),
        mean_correlation=correlation(proportional[:, 0], by_coordinate[0]),
        variance_correlation=correlation(proportional[:, 1], by_coordinate[1]),
        mean_integral_score=float(np.mean(by_coordinate[0] * integral[:, 0])),
        variance_integral_score=float(np.mean(by_coordinate[1] * integral[:, 1])),
    )


class EndogenousGainSolver(Solver[EndogenousGainProblem]):
    """Train the endogenous-gain model on fresh samples and evaluate it on held-out ones."""

    _: KW_ONLY
    scenario: EndogenousGainScenario = eqx.field(default_factory=EndogenousGainScenario)
    mismatch_settings: MismatchSettings = eqx.field(static=True, default_factory=MismatchSettings)
    training_examples: int = int_field(default=40_000, domain=IntDistribution(1, 1 << 20, log=True))
    training_batch_size: int = int_field(default=64, domain=IntDistribution(1, 1 << 12, log=True))
    inference_examples: int = int_field(default=16, domain=IntDistribution(1, 1 << 12, log=True))
    inference_batch_size: int = int_field(default=256, domain=IntDistribution(1, 1 << 12, log=True))
    learning_rate: float = float_field(
        default=0.003, domain=FloatDistribution(1e-4, 1.0, log=True), optimize=True
    )

    @override
    def problem(self) -> EndogenousGainProblem:
        return EndogenousGainProblem(self.scenario)

    @override
    def create_model(
        self,
        data_source: DataSource,
        problem: Problem,
        *,
        streams: Mapping[str, RngStream],
    ) -> Model:
        return EndogenousGainModel.create(mismatch_settings=self.mismatch_settings, streams=streams)

    @override
    def gradient_transformations(self) -> DisGradientTransformation:
        return sgd_gradient_transformations(self.learning_rate)


class EndogenousGainScoresPlotter(Plotter):
    """Compare each gain rule's slope bias and remaining population moments.

    The moments are evaluated at each run's fitted parameters on its held-out instrument draws,
    which both conditions share. The fits and witness checks are recorded for each seed.
    """

    _: KW_ONLY
    conditions: Mapping[str, EndogenousGainScenario] = field(static=True)

    @override
    def telemetries(self) -> Telemetries:
        return Telemetries((GainFitTelemetry(), GainWitnessTelemetry()))

    @override
    def plotted_series(self, runs: Sequence[Run]) -> PlottedSeries:
        scores = {
            condition: [score_gain_run(run, scenario) for run in seed_runs(runs, condition)]
            for condition, scenario in self.conditions.items()
        }
        effects = {
            condition: scenario.prediction.effects[0]
            for condition, scenario in self.conditions.items()
        }

        def series(value: Callable[[GainScores], float]) -> list[list[float]]:
            return [[value(score) for score in seed_scores] for seed_scores in scores.values()]

        return bar_chart(
            list(self.conditions),
            {
                "slope bias": (
                    "Slope bias",
                    [
                        [score.fit.slope - effects[condition] for score in seed_scores]
                        for condition, seed_scores in scores.items()
                    ],
                ),
                "mean moment": ("Mean moment", series(lambda score: score.mean_moment)),
                "variance moment": (
                    "Variance moment",
                    series(lambda score: score.variance_moment),
                ),
            },
        ) | {
            "seed scores": {
                condition: [asdict(score) for score in seed_scores]
                for condition, seed_scores in scores.items()
            }
        }


class EndogenousGainConflictPlotter(Plotter):
    """Plot the intercept and variance that each gain instrument value calls for.

    At the causal slope, the mean moment at gain instrument value γ̃ vanishes for the intercept
    λ m(γ̃), where λ is the target's confounding coefficient and m(γ̃) is the confounder's
    gain-weighted mean. At that intercept, the variance moment vanishes for the observation
    variance minus λ² m(γ̃)². A constant prediction can meet both only where m is constant.
    """

    _: KW_ONLY
    conditions: Mapping[str, EndogenousGainScenario] = field(static=True)
    grid: tuple[float, ...] = field(
        static=True, default=tuple(float(x) for x in np.linspace(-0.2, 0.2, 41))
    )

    @override
    def plotted_series(self, runs: Sequence[Run]) -> PlottedSeries:
        grid = jnp.asarray(self.grid)
        lines: dict[str, list[float]] = {}
        for condition, scenario in self.conditions.items():
            prediction = scenario.prediction
            weighted_mean = np.asarray(scenario.gain_weighted_confounder_mean(grid))
            confounding = prediction.target_confounding
            lines[f"{condition} intercept"] = [float(v) for v in confounding * weighted_mean]
            lines[f"{condition} variance"] = [
                float(v)
                for v in prediction.interventional_variance()
                + FEATURE_OBSERVATION_VARIANCE
                - confounding**2 * np.square(weighted_mean)
            ]
        return {
            "iteration": [float(value) for value in self.grid],
            "line plots": {key: key for key in lines},
            **lines,
        }


# The variance settles more slowly than the mean, as in test 5.
_GAIN_TRAINING_EXAMPLES = 160_000
GAIN_CONTROL = "Exogenous gain"
ENDOGENOUS_GAIN = "Endogenous gain"
ENDOGENOUS_GAIN_SCENARIOS = {
    GAIN_CONTROL: EndogenousGainScenario(),
    ENDOGENOUS_GAIN: EndogenousGainScenario(gain_confounding=0.2),
}


class EndogenousGainExperiment(SeededExperiment):
    """Test 8: the full mismatch circuit under an additive gain that depends on the confounder.

    Both conditions draw the gain instrument uniformly from [-0.2, 0.2] and add it to f(U). The
    control's f is constant, so the causal slope, a zero intercept, and the interventional
    observation variance satisfy both witness moments. With f(U) = 0.5 + 0.2 tanh(U), the mean
    moment calls for an intercept that varies with the gain instrument, so no single intercept
    satisfies it. The conditions share seeds, initialization, and sample draws.
    """

    def __init__(
        self, solver: EndogenousGainSolver | None = None, *, seeds: int = SEED_COUNT
    ) -> None:
        solver = (
            EndogenousGainSolver(training_examples=_GAIN_TRAINING_EXAMPLES)
            if solver is None
            else solver
        )
        super().__init__(
            name="sip-endogenous-gain",
            variants=[
                Variant(label=condition, solver=replace(solver, scenario=scenario))
                for condition, scenario in ENDOGENOUS_GAIN_SCENARIOS.items()
            ],
            plotters=[
                EndogenousGainScoresPlotter(
                    name="endogenous-gain-scores",
                    title="Slope bias and remaining population moments",
                    conditions=ENDOGENOUS_GAIN_SCENARIOS,
                ),
                EndogenousGainConflictPlotter(
                    name="endogenous-gain-conflict",
                    title="Intercept and variance that each gain instrument value calls for",
                    conditions=ENDOGENOUS_GAIN_SCENARIOS,
                ),
            ],
            sweep=[{"seed": seed} for seed in range(seeds)],
        )


sip_causal_mean_experiment = CausalMeanExperiment()
sip_causal_variance_experiment = CausalVarianceExperiment()
sip_witness_diagnostics_experiment = WitnessDiagnosticsExperiment()
sip_nonlinear_effect_experiment = NonlinearEffectExperiment()
sip_endogenous_gain_experiment = EndogenousGainExperiment()
