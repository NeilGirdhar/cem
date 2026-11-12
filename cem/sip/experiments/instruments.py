"""Instrument experiments: whether inherited instruments identify downstream causal effects.

See `sip.md`, tests 1–3.
"""

from collections.abc import Mapping, Sequence
from dataclasses import KW_ONLY, dataclass, replace
from statistics import fmean, pstdev
from typing import Any, Self, override

import equinox as eqx
import jax.numpy as jnp
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
from .datasets import InstrumentChainObservation, InstrumentChainProblem, InstrumentChainState
from .scenarios import InstrumentChainScenario

_CHAIN_NODE = "chain"
_SEED_COUNT = 5


class InstrumentChainConfiguration(NodeConfiguration):
    """The mean of each feature's inherited instrument."""

    inherited_instruments: JaxRealArray  # (depth,)


class InstrumentChainModel(Model):
    """The instrument maps of a chain of non-controllable features.

    The first map receives the parents' instruments, and each later map receives its predecessor's
    inherited instrument, which a non-controllable feature emits unchanged. Each map has the same
    construction as a SIP feature's instrument map, and only its own instrument loss trains it.
    Isolating the maps tests the first stages without the circuits that learn each feature's
    observation or prediction.
    """

    instrument_maps: tuple[GaussianNeuralProjection, ...]

    @classmethod
    def create(cls, n_parents: int, depth: int, *, streams: Mapping[str, RngStream]) -> Self:
        return cls(
            tuple(
                GaussianNeuralProjection.create(
                    in_features=n_parents if hop == 0 else 1, out_features=1, streams=streams
                )
                for hop in range(depth)
            )
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
        assert isinstance(observation, InstrumentChainObservation)
        observations = observation.feature_observations
        instrument = observation.parent_instrument
        loss = jnp.zeros(())
        means = []
        for hop, instrument_map in enumerate(self.instrument_maps):
            instrument = inherit_instrument(instrument_map, instrument)
            feature_observation = NormalNP(
                observations.mean_times_precision[hop : hop + 1],
                observations.negative_half_precision[hop : hop + 1],
            )
            loss += instrument_loss(instrument, feature_observation, None)
            means.append(instrument.mean()[0])
        return ModelResult(
            loss=loss,
            configurations=frozendict(
                {_CHAIN_NODE: InstrumentChainConfiguration(inherited_instruments=jnp.stack(means))}
            ),
            state=state,
        )


class InstrumentChainSamples(eqx.Module):
    """Held-out examples: the inherited instruments' means and the structural variables."""

    inherited_instruments: JaxRealArray  # (..., batch, depth)
    oracle_first_stages: JaxRealArray  # (..., batch, depth)
    injected_noise: JaxRealArray  # (..., batch, k)
    features: JaxRealArray  # (..., batch, depth + 1)


class InstrumentChainTelemetry(Telemetry):
    """Record every held-out example so the estimators can run on the pooled samples."""

    @override
    def inference_snapshot(
        self,
        inference: Inference,
        inference_result: InferenceResult,
        snapshots: Mapping[Telemetry, Any],
    ) -> InstrumentChainSamples:
        configuration = inference_result.model_configuration[_CHAIN_NODE]
        assert isinstance(configuration, InstrumentChainConfiguration)
        state = inference_result.initial_problem_state
        assert isinstance(state, InstrumentChainState)
        sample = state.sample
        return InstrumentChainSamples(
            inherited_instruments=configuration.inherited_instruments,
            oracle_first_stages=sample.oracle_first_stages,
            injected_noise=sample.injected_noise,
            features=sample.features,
        )


@dataclass(frozen=True)
class HopEstimates:
    """First-stage quality and effect estimates for one hop from one run's held-out samples.

    Strength is the fraction of the feature's variance that an instrument explains. The
    approximation error is the root-mean-square difference between the inherited instrument's
    mean and the oracle first stage. The estimates are of the feature's effect on its successor.
    """

    approximation_error: float
    oracle_strength: float
    inherited_strength: float
    ordinary_least_squares: float
    injected_noise: float
    inherited_instrument: float


def _strength(instrument: np.ndarray, feature: np.ndarray) -> float:
    return float(np.square(np.corrcoef(instrument, feature)[0, 1]))


def estimate_hops(samples: InstrumentChainSamples) -> tuple[HopEstimates, ...]:
    """Compare each inherited instrument with the oracle and estimate each hop's effect.

    Two-stage least squares on the parents' injected noise is the reference estimate. Ordinary
    least squares shows the confounded estimate that the instruments must avoid.
    """
    inherited = np.asarray(samples.inherited_instruments)
    inherited = inherited.reshape(-1, inherited.shape[-1])
    oracle = np.asarray(samples.oracle_first_stages).reshape(inherited.shape)
    features = np.asarray(samples.features)
    features = features.reshape(-1, features.shape[-1])
    noise = np.asarray(samples.injected_noise)
    noise = noise.reshape(-1, noise.shape[-1])
    return tuple(
        HopEstimates(
            approximation_error=float(
                np.sqrt(np.mean(np.square(inherited[:, hop] - oracle[:, hop])))
            ),
            oracle_strength=_strength(oracle[:, hop], features[:, hop]),
            inherited_strength=_strength(inherited[:, hop], features[:, hop]),
            ordinary_least_squares=ordinary_least_squares(features[:, hop], features[:, hop + 1]),
            injected_noise=two_stage_least_squares(noise, features[:, hop], features[:, hop + 1]),
            inherited_instrument=two_stage_least_squares(
                inherited[:, hop], features[:, hop], features[:, hop + 1]
            ),
        )
        for hop in range(inherited.shape[-1])
    )


def _run_hops(run: Run) -> tuple[HopEstimates, ...]:
    samples = run.inference_results.telemetries[InstrumentChainTelemetry()]
    assert isinstance(samples, InstrumentChainSamples)
    return estimate_hops(samples)


def _bar_chart(
    x_labels: Sequence[str], series: Mapping[str, tuple[str, Sequence[Sequence[float]]]]
) -> PlottedSeries:
    """Plot the mean of each series at each x position with one standard deviation across seeds.

    Each series maps its key to its legend label and to its seed values at each x position.
    """
    return {
        "iteration": list(range(len(x_labels))),
        "x labels": list(x_labels),
        "bar plots": {key: label for key, (label, _) in series.items()},
        "bar errors": {
            key: [pstdev(values) for values in seed_values]
            for key, (_, seed_values) in series.items()
        },
        **{
            key: [fmean(values) for values in seed_values]
            for key, (_, seed_values) in series.items()
        },
        "seed values": {
            key: [list(values) for values in seed_values]
            for key, (_, seed_values) in series.items()
        },
        "error bars": "one population standard deviation across seeds",
    }


def _approximation_errors(hops_by_run: Sequence[tuple[HopEstimates, ...]]) -> PlottedSeries:
    """Report each hop's approximation error across seeds."""
    by_hop = [
        [hops[hop].approximation_error for hops in hops_by_run]
        for hop in range(len(hops_by_run[0]))
    ]
    return {
        "approximation error": [fmean(errors) for errors in by_hop],
        "approximation error deviation": [pstdev(errors) for errors in by_hop],
        "seed approximation errors": by_hop,
    }


def _estimator_errors(hops: Sequence[HopEstimates], effect: float) -> dict[str, list[float]]:
    """Return each estimator's absolute error across seeds for one hop."""
    return {
        "OLS": [abs(h.ordinary_least_squares - effect) for h in hops],
        "2SLS, noise": [abs(h.injected_noise - effect) for h in hops],
        "2SLS, inherited": [abs(h.inherited_instrument - effect) for h in hops],
    }


class FirstStagePlotter(Plotter):
    """Compare the first feature's inherited instrument strength with the oracle first stage's."""

    _: KW_ONLY
    name: str = field(static=True, default="inherited-instrument-first-stage")
    title: str = field(static=True, default="First-stage strength on held-out samples")

    @override
    def telemetries(self) -> Telemetries:
        return Telemetries((InstrumentChainTelemetry(),))

    @override
    def plotted_series(self, runs: Sequence[Run]) -> PlottedSeries:
        hops_by_run = [_run_hops(run) for run in runs]
        first = [hops[0] for hops in hops_by_run]
        approximation = _approximation_errors(hops_by_run)
        return _bar_chart(
            ["Oracle", "Inherited"],
            {
                "strength": (
                    "Variance explained",
                    [[h.oracle_strength for h in first], [h.inherited_strength for h in first]],
                )
            },
        ) | {
            "approximation error": approximation["approximation error"][0],
            "approximation error deviation": approximation["approximation error deviation"][0],
            "seed approximation errors": approximation["seed approximation errors"][0],
        }


class DownstreamEffectPlotter(Plotter):
    """Compare each estimator's error in the first feature's effect on its successor."""

    _: KW_ONLY
    name: str = field(static=True, default="inherited-instrument-effect")
    title: str = field(static=True, default="Downstream effect error on held-out samples")
    effect: float = field(static=True)

    @override
    def telemetries(self) -> Telemetries:
        return Telemetries((InstrumentChainTelemetry(),))

    @override
    def plotted_series(self, runs: Sequence[Run]) -> PlottedSeries:
        first = [_run_hops(run)[0] for run in runs]
        errors = _estimator_errors(first, self.effect)
        return _bar_chart(list(errors), {"error": ("Absolute error", list(errors.values()))}) | {
            "true effect": self.effect
        }


class DepthFirstStagePlotter(Plotter):
    """Compare each feature's inherited instrument strength with its oracle first stage's."""

    _: KW_ONLY
    name: str = field(static=True, default="instrument-depth-first-stage")
    title: str = field(static=True, default="First-stage strength by depth")

    @override
    def telemetries(self) -> Telemetries:
        return Telemetries((InstrumentChainTelemetry(),))

    @override
    def plotted_series(self, runs: Sequence[Run]) -> PlottedSeries:
        hops_by_run = [_run_hops(run) for run in runs]
        depth = len(hops_by_run[0])
        return _bar_chart(
            [f"Feature {hop + 1}" for hop in range(depth)],
            {
                "oracle": (
                    "Oracle",
                    [[hops[hop].oracle_strength for hops in hops_by_run] for hop in range(depth)],
                ),
                "inherited": (
                    "Inherited",
                    [
                        [hops[hop].inherited_strength for hops in hops_by_run]
                        for hop in range(depth)
                    ],
                ),
            },
        ) | _approximation_errors(hops_by_run)


class DepthEffectPlotter(Plotter):
    """Compare each estimator's error in every hop's effect."""

    _: KW_ONLY
    name: str = field(static=True, default="instrument-depth-effect")
    title: str = field(static=True, default="Effect error by depth")
    effect: float = field(static=True)

    @override
    def telemetries(self) -> Telemetries:
        return Telemetries((InstrumentChainTelemetry(),))

    @override
    def plotted_series(self, runs: Sequence[Run]) -> PlottedSeries:
        hops_by_run = [_run_hops(run) for run in runs]
        depth = len(hops_by_run[0])
        by_hop = [
            _estimator_errors([hops[hop] for hops in hops_by_run], self.effect)
            for hop in range(depth)
        ]
        return _bar_chart(
            [f"Hop {hop + 1}" for hop in range(depth)],
            {
                key: (key, [errors[key] for errors in by_hop])
                for key in ("OLS", "2SLS, noise", "2SLS, inherited")
            },
        ) | {"true effect": self.effect}


class InstrumentChainSolver(Solver[InstrumentChainProblem]):
    """Train a chain's instrument maps on fresh samples and evaluate them on held-out ones."""

    _: KW_ONLY
    scenario: InstrumentChainScenario = eqx.field(default_factory=InstrumentChainScenario)
    training_examples: int = int_field(default=10_000, domain=IntDistribution(1, 1 << 20, log=True))
    training_batch_size: int = int_field(default=64, domain=IntDistribution(1, 1 << 12, log=True))
    inference_examples: int = int_field(default=16, domain=IntDistribution(1, 1 << 12, log=True))
    inference_batch_size: int = int_field(default=256, domain=IntDistribution(1, 1 << 12, log=True))
    learning_rate: float = float_field(
        default=0.005, domain=FloatDistribution(1e-4, 1.0, log=True), optimize=True
    )

    @override
    def problem(self) -> InstrumentChainProblem:
        return InstrumentChainProblem(self.scenario)

    @override
    def create_model(
        self,
        data_source: DataSource,
        problem: Problem,
        *,
        streams: Mapping[str, RngStream],
    ) -> Model:
        return InstrumentChainModel.create(
            self.scenario.n_parents, self.scenario.depth, streams=streams
        )


class _SeededExperiment(Experiment):
    """Run one solver over several seeds.

    Each seed trains from fresh initial parameters and evaluates on its own held-out samples,
    where every estimator sees the same samples.
    """

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


class InheritedInstrumentExperiment(_SeededExperiment):
    """Test 1: an inherited instrument identifies a downstream effect despite confounding."""

    def __init__(
        self, solver: InstrumentChainSolver | None = None, *, seeds: int = _SEED_COUNT
    ) -> None:
        solver = InstrumentChainSolver() if solver is None else solver
        super().__init__(
            name="sip-inherited-instrument",
            variants=[Variant(label="", solver=solver)],
            plotters=[FirstStagePlotter(), DownstreamEffectPlotter(effect=solver.scenario.effect)],
            sweep=[{"seed": seed} for seed in range(seeds)],
        )


_DEPTH_SCENARIO = InstrumentChainScenario(depth=4, effect=1.0)
# Each predecessor's instrument is less precise, so the input pool shrinks it more and deeper maps
# need larger weights. They take longer to reach the oracle first stage.
_DEPTH_TRAINING_EXAMPLES = 40_000


class InstrumentDepthExperiment(_SeededExperiment):
    """Test 2: inherited instruments identify every hop's effect along a confounded chain.

    With a unit effect, each hop adds confounded variation without amplifying the injected noise,
    so the instruments explain less of each successive feature.
    """

    def __init__(
        self, solver: InstrumentChainSolver | None = None, *, seeds: int = _SEED_COUNT
    ) -> None:
        solver = (
            InstrumentChainSolver(
                scenario=_DEPTH_SCENARIO, training_examples=_DEPTH_TRAINING_EXAMPLES
            )
            if solver is None
            else solver
        )
        super().__init__(
            name="sip-instrument-depth",
            variants=[Variant(label="", solver=solver)],
            plotters=[DepthFirstStagePlotter(), DepthEffectPlotter(effect=solver.scenario.effect)],
            sweep=[{"seed": seed} for seed in range(seeds)],
        )


sip_inherited_instrument_experiment = InheritedInstrumentExperiment()
sip_instrument_depth_experiment = InstrumentDepthExperiment()
