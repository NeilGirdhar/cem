"""Check tests 1–3: inherited instruments identify effects despite confounding and bypasses."""

from collections.abc import Sequence

import pytest

from cem.commands.registry import experiments
from cem.framework.plotter import Run
from cem.sip.experiments.instruments import (
    DepthEffectPlotter,
    DepthFirstStagePlotter,
    DownstreamEffectPlotter,
    ExclusionDriftPlotter,
    ExclusionRepairExperiment,
    FirstStagePlotter,
    InheritedInstrumentExperiment,
    InstrumentChainSamples,
    InstrumentChainScenario,
    InstrumentChainSolver,
    InstrumentChainTelemetry,
    InstrumentDepthExperiment,
    bypass_drift,
    bypass_parents,
    estimate_exclusion,
    estimate_hops,
    sip_exclusion_repair_experiment,
    sip_inherited_instrument_experiment,
    sip_instrument_depth_experiment,
)

_SEEDS = 2
_DEPTH = 3
# The oracle first stage has a standard deviation of about 1.1 at every depth.
_APPROXIMATION_TOLERANCE = 0.25
_STRENGTH_TOLERANCE = 0.01
_REFERENCE_TOLERANCE = 0.01
# Two-stage least squares has a standard error of at most about 0.05 on the held-out samples.
_EFFECT_TOLERANCE = 0.15
# Ordinary least squares misses the effect by at least 0.4 at every hop.
_CONFOUNDING_BIAS = 0.3
# Omitting the bypass shifts the estimate by 0.5.
_DRIFT_TOLERANCE = 0.1


def _check_hops(runs: Sequence[Run], effect: float, depth: int) -> None:
    """Every hop's first stage matches the oracle, and 2SLS with it matches the reference.

    Ordinary least squares stays far from the effect, so every hop is confounded.
    """
    for run in runs:
        samples = run.inference_results.telemetries[InstrumentChainTelemetry()]
        assert isinstance(samples, InstrumentChainSamples)
        hops = estimate_hops(samples)
        assert len(hops) == depth
        for hop in hops:
            assert hop.approximation_error < _APPROXIMATION_TOLERANCE
            assert abs(hop.inherited_strength - hop.oracle_strength) < _STRENGTH_TOLERANCE
            assert abs(hop.inherited_instrument - hop.injected_noise) < _REFERENCE_TOLERANCE
            assert abs(hop.inherited_instrument - effect) < _EFFECT_TOLERANCE
            assert abs(hop.ordinary_least_squares - effect) > _CONFOUNDING_BIAS


def test_instrument_experiments_are_registered() -> None:
    assert experiments["sip-inherited-instrument"] is sip_inherited_instrument_experiment
    assert experiments["sip-instrument-depth"] is sip_instrument_depth_experiment
    assert experiments["sip-exclusion-repair"] is sip_exclusion_repair_experiment


@pytest.mark.usefixtures("log")
def test_inherited_instrument_identifies_the_downstream_effect() -> None:
    solver = InstrumentChainSolver(training_examples=1000, learning_rate=0.02)
    effect = solver.scenario.effect
    runs = InheritedInstrumentExperiment(solver, seeds=_SEEDS).runs({})
    _check_hops(runs, effect, 1)

    first_stage = FirstStagePlotter().plotted_series(runs)
    assert first_stage["x labels"] == ["Oracle", "Inherited"]
    assert len(first_stage["seed approximation errors"]) == _SEEDS
    errors = DownstreamEffectPlotter(effect=effect).plotted_series(runs)["error"]
    assert errors[0] > errors[2]


@pytest.mark.usefixtures("log")
def test_inherited_instruments_survive_depth() -> None:
    scenario = InstrumentChainScenario(depth=_DEPTH, effect=1.0)
    solver = InstrumentChainSolver(scenario=scenario, training_examples=3000, learning_rate=0.02)
    runs = InstrumentDepthExperiment(solver, seeds=_SEEDS).runs({})
    _check_hops(runs, scenario.effect, _DEPTH)

    first_stage = DepthFirstStagePlotter().plotted_series(runs)
    assert first_stage["x labels"] == [f"Feature {hop + 1}" for hop in range(_DEPTH)]
    assert len(first_stage["approximation error"]) == _DEPTH
    effect_series = DepthEffectPlotter(effect=scenario.effect).plotted_series(runs)
    assert all(
        ols > inherited
        for ols, inherited in zip(
            effect_series["OLS"], effect_series["2SLS, inherited"], strict=True
        )
    )


@pytest.mark.usefixtures("log")
def test_bypass_regressor_repairs_the_exclusion_violation() -> None:
    """Both paths are recovered with the bypass regressor and drift as predicted without it."""
    scenario = InstrumentChainScenario(first_stage=(1.0, 1.0), bypass=(1.0, 0.0))
    solver = InstrumentChainSolver(scenario=scenario, training_examples=1000, learning_rate=0.02)
    runs = ExclusionRepairExperiment(solver, seeds=_SEEDS).runs({})
    truth = (scenario.effect, 1.0)
    drift = bypass_drift(scenario)

    for run in runs:
        samples = run.inference_results.telemetries[InstrumentChainTelemetry()]
        assert isinstance(samples, InstrumentChainSamples)
        estimates = estimate_exclusion(samples, bypass_parents(scenario))
        for inherited, reference, ordinary, true in zip(
            estimates.inherited_instrument,
            estimates.injected_noise,
            estimates.ordinary_least_squares,
            truth,
            strict=True,
        ):
            assert abs(inherited - true) < _EFFECT_TOLERANCE
            assert abs(inherited - reference) < _REFERENCE_TOLERANCE
            assert abs(ordinary - true) > _CONFOUNDING_BIAS
        assert abs(estimates.bypass_omitted - drift) < _DRIFT_TOLERANCE
        assert abs(estimates.bypass_omitted - scenario.effect) > _CONFOUNDING_BIAS

    drift_series = ExclusionDriftPlotter(scenario=scenario).plotted_series(runs)
    assert drift_series["x labels"] == ["True", "Repaired", "Omitted", "Predicted drift"]
    assert drift_series["predicted drift"] == drift
