"""Instrument experiments: whether inherited instruments identify downstream causal effects.

See `sip.md`, tests 1–3.
"""

from collections.abc import Mapping, Sequence
from dataclasses import KW_ONLY, dataclass, replace
from statistics import fmean, pstdev
from typing import Any, Self, override

import equinox as eqx
import numpy as np
from efax import NormalNP
from optuna.distributions import FloatDistribution, IntDistribution
from tjax import JaxRealArray, RngStream
from tjax.dataclasses import field

from cem.framework.graph import Model, ModelResult, NodeConfiguration
from cem.framework.plotter import Experiment, Run, Variant
from cem.framework.plotter.plotter import PlottedSeries, Plotter
from cem.framework.problem import DataSource, Problem
from cem.framework.solution import Inference, InferenceResult, Telemetries, Telemetry
from cem.framework.solver import Solver, float_field, int_field
from cem.npn.gaussian import GaussianNeuralProjection
from cem.sip.instrument import inherit_instrument, instrument_loss

from .baselines import ordinary_least_squares, two_stage_least_squares
from .datasets import (
    InheritedInstrumentObservation,
    InheritedInstrumentProblem,
    InheritedInstrumentState,
)
from .scenarios import InheritedInstrumentScenario

_CHILD_NODE = "child"
_SEED_COUNT = 5


class InheritedInstrumentConfiguration(NodeConfiguration):
    """The child's inherited instrument."""

    inherited_instrument: NormalNP  # (1,)


class InheritedInstrumentModel(Model):
    """A non-controllable child's instrument map, trained by the instrument loss alone.

    The map has the same construction as a SIP feature's instrument map. Isolating it tests the
    first stage without the circuits that learn the child's observation or prediction.
    """

    instrument_map: GaussianNeuralProjection

    @classmethod
    def create(cls, n_parents: int, *, streams: Mapping[str, RngStream]) -> Self:
        return cls(
            GaussianNeuralProjection.create(in_features=n_parents, out_features=1, streams=streams)
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
        assert isinstance(observation, InheritedInstrumentObservation)
        inherited = inherit_instrument(self.instrument_map, observation.parent_instrument)
        return ModelResult(
            loss=instrument_loss(inherited, observation.child_observation, None),
            configurations=frozendict(
                {_CHILD_NODE: InheritedInstrumentConfiguration(inherited_instrument=inherited)}
            ),
            state=state,
        )


class InheritedInstrumentSamples(eqx.Module):
    """Held-out examples: the inherited instrument's mean and the structural variables."""

    inherited_instrument: JaxRealArray  # (..., batch)
    oracle_first_stage: JaxRealArray  # (..., batch)
    injected_noise: JaxRealArray  # (..., batch, k)
    child: JaxRealArray  # (..., batch)
    downstream: JaxRealArray  # (..., batch)


class InheritedInstrumentTelemetry(Telemetry):
    """Record every held-out example so the estimators can run on the pooled samples."""

    @override
    def inference_snapshot(
        self,
        inference: Inference,
        inference_result: InferenceResult,
        snapshots: Mapping[Telemetry, Any],
    ) -> InheritedInstrumentSamples:
        configuration = inference_result.model_configuration[_CHILD_NODE]
        assert isinstance(configuration, InheritedInstrumentConfiguration)
        state = inference_result.initial_problem_state
        assert isinstance(state, InheritedInstrumentState)
        sample = state.sample
        return InheritedInstrumentSamples(
            inherited_instrument=configuration.inherited_instrument.mean()[..., 0],
            oracle_first_stage=sample.oracle_first_stage,
            injected_noise=sample.injected_noise,
            child=sample.child,
            downstream=sample.downstream,
        )


@dataclass(frozen=True)
class InheritedInstrumentEstimates:
    """First-stage quality and downstream effect estimates from one run's held-out samples.

    Strength is the fraction of the child's variance that an instrument explains. The
    approximation error is the root-mean-square difference between the inherited instrument's
    mean and the oracle first stage.
    """

    approximation_error: float
    oracle_strength: float
    inherited_strength: float
    ordinary_least_squares: float
    injected_noise: float
    inherited_instrument: float


def _strength(instrument: np.ndarray, child: np.ndarray) -> float:
    return float(np.square(np.corrcoef(instrument, child)[0, 1]))


def estimate_inherited_instrument(
    samples: InheritedInstrumentSamples,
) -> InheritedInstrumentEstimates:
    """Compare the inherited instrument with the oracle and estimate the downstream effect.

    Two-stage least squares on the parents' injected noise is the reference estimate. Ordinary
    least squares shows the confounded estimate that the instruments must avoid.
    """
    inherited = np.asarray(samples.inherited_instrument).ravel()
    oracle = np.asarray(samples.oracle_first_stage).ravel()
    child = np.asarray(samples.child).ravel()
    downstream = np.asarray(samples.downstream).ravel()
    noise = np.asarray(samples.injected_noise)
    noise = noise.reshape(-1, noise.shape[-1])
    return InheritedInstrumentEstimates(
        approximation_error=float(np.sqrt(np.mean(np.square(inherited - oracle)))),
        oracle_strength=_strength(oracle, child),
        inherited_strength=_strength(inherited, child),
        ordinary_least_squares=ordinary_least_squares(child, downstream),
        injected_noise=two_stage_least_squares(noise, child, downstream),
        inherited_instrument=two_stage_least_squares(inherited, child, downstream),
    )


def _run_estimates(run: Run) -> InheritedInstrumentEstimates:
    samples = run.inference_results.telemetries[InheritedInstrumentTelemetry()]
    assert isinstance(samples, InheritedInstrumentSamples)
    return estimate_inherited_instrument(samples)


def _bar_series(
    key: str, label: str, x_labels: Sequence[str], seed_values: Sequence[Sequence[float]]
) -> PlottedSeries:
    """Plot each group's mean with one population standard deviation across seeds."""
    return {
        "iteration": list(range(len(x_labels))),
        "x labels": list(x_labels),
        "bar plots": {key: label},
        "bar errors": {key: [pstdev(values) for values in seed_values]},
        key: [fmean(values) for values in seed_values],
        "seed values": [list(values) for values in seed_values],
        "error bars": "one population standard deviation across seeds",
    }


class FirstStagePlotter(Plotter):
    """Compare the inherited instrument's strength with the oracle first stage's."""

    _: KW_ONLY
    name: str = field(static=True, default="inherited-instrument-first-stage")
    title: str = field(static=True, default="First-stage strength on held-out samples")

    @override
    def telemetries(self) -> Telemetries:
        return Telemetries((InheritedInstrumentTelemetry(),))

    @override
    def plotted_series(self, runs: Sequence[Run]) -> PlottedSeries:
        estimates = [_run_estimates(run) for run in runs]
        approximation_errors = [e.approximation_error for e in estimates]
        return _bar_series(
            "strength",
            "Variance explained",
            ["Oracle", "Inherited"],
            [[e.oracle_strength for e in estimates], [e.inherited_strength for e in estimates]],
        ) | {
            "approximation error": fmean(approximation_errors),
            "approximation error deviation": pstdev(approximation_errors),
            "seed approximation errors": approximation_errors,
        }


class DownstreamEffectPlotter(Plotter):
    """Compare each estimator's error in the child's effect on the downstream feature."""

    _: KW_ONLY
    name: str = field(static=True, default="inherited-instrument-effect")
    title: str = field(static=True, default="Downstream effect error on held-out samples")
    effect: float = field(static=True)

    @override
    def telemetries(self) -> Telemetries:
        return Telemetries((InheritedInstrumentTelemetry(),))

    @override
    def plotted_series(self, runs: Sequence[Run]) -> PlottedSeries:
        estimates = [_run_estimates(run) for run in runs]
        by_estimator = {
            "OLS": [e.ordinary_least_squares for e in estimates],
            "2SLS, noise": [e.injected_noise for e in estimates],
            "2SLS, inherited": [e.inherited_instrument for e in estimates],
        }
        return _bar_series(
            "error",
            "Absolute error",
            list(by_estimator),
            [[abs(value - self.effect) for value in values] for values in by_estimator.values()],
        ) | {"true effect": self.effect, "seed estimates": by_estimator}


class InheritedInstrumentSolver(Solver[InheritedInstrumentProblem]):
    """Train a child's instrument map on fresh scenario samples and evaluate it on held-out ones."""

    _: KW_ONLY
    scenario: InheritedInstrumentScenario = eqx.field(default_factory=InheritedInstrumentScenario)
    training_examples: int = int_field(default=10_000, domain=IntDistribution(1, 1 << 20, log=True))
    training_batch_size: int = int_field(default=64, domain=IntDistribution(1, 1 << 12, log=True))
    inference_examples: int = int_field(default=16, domain=IntDistribution(1, 1 << 12, log=True))
    inference_batch_size: int = int_field(default=256, domain=IntDistribution(1, 1 << 12, log=True))
    learning_rate: float = float_field(
        default=0.005, domain=FloatDistribution(1e-4, 1.0, log=True), optimize=True
    )

    @override
    def problem(self) -> InheritedInstrumentProblem:
        return InheritedInstrumentProblem(self.scenario)

    @override
    def create_model(
        self,
        data_source: DataSource,
        problem: Problem,
        *,
        streams: Mapping[str, RngStream],
    ) -> Model:
        return InheritedInstrumentModel.create(self.scenario.n_parents, streams=streams)


class InheritedInstrumentExperiment(Experiment):
    """Test 1: an inherited instrument identifies a downstream effect despite confounding.

    Each seed trains the child's instrument map from fresh initial parameters and evaluates it on
    held-out samples, where every estimator sees the same samples.
    """

    def __init__(
        self, solver: InheritedInstrumentSolver | None = None, *, seeds: int = _SEED_COUNT
    ) -> None:
        solver = InheritedInstrumentSolver() if solver is None else solver
        super().__init__(
            name="sip-inherited-instrument",
            variants=[Variant(label="", solver=solver)],
            plotters=[FirstStagePlotter(), DownstreamEffectPlotter(effect=solver.scenario.effect)],
            sweep=[{"seed": seed} for seed in range(seeds)],
        )

    @override
    def resolve_solver(
        self,
        variant: Variant,
        sweep_entry: Mapping[str, Any],
        hyperparameters: Mapping[str, Any],
    ) -> Solver[Any]:
        """Give each seed its own initial parameters and its own training and held-out samples."""
        solver = super().resolve_solver(variant, sweep_entry, hyperparameters)
        seed = sweep_entry["seed"]
        return replace(
            solver, parameters_seed=seed, training_seed=100 + seed, inference_seed=200 + seed
        )


sip_inherited_instrument_experiment = InheritedInstrumentExperiment()
