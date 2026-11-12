"""Check test 1: an inherited instrument identifies a downstream effect despite confounding."""

import pytest

from cem.commands.registry import experiments
from cem.sip.experiments.instruments import (
    DownstreamEffectPlotter,
    FirstStagePlotter,
    InheritedInstrumentExperiment,
    InheritedInstrumentSamples,
    InheritedInstrumentSolver,
    InheritedInstrumentTelemetry,
    estimate_inherited_instrument,
    sip_inherited_instrument_experiment,
)

_SEEDS = 2
# The oracle first stage has a standard deviation of about 1.1.
_APPROXIMATION_TOLERANCE = 0.15
_STRENGTH_TOLERANCE = 0.01
_REFERENCE_TOLERANCE = 0.01
# Two-stage least squares has a standard error of about 0.03 on the held-out samples.
_EFFECT_TOLERANCE = 0.1
_CONFOUNDING_BIAS = 0.5


def test_inherited_instrument_experiment_is_registered() -> None:
    assert experiments["sip-inherited-instrument"] is sip_inherited_instrument_experiment


@pytest.mark.usefixtures("log")
def test_inherited_instrument_identifies_the_downstream_effect() -> None:
    """The learned first stage matches the oracle, and 2SLS with it matches the reference.

    Ordinary least squares stays far from the effect, so the scenario is confounded.
    """
    solver = InheritedInstrumentSolver(training_examples=1000, learning_rate=0.02)
    experiment = InheritedInstrumentExperiment(solver, seeds=_SEEDS)
    effect = solver.scenario.effect
    runs = experiment.runs({})

    for run in runs:
        samples = run.inference_results.telemetries[InheritedInstrumentTelemetry()]
        assert isinstance(samples, InheritedInstrumentSamples)
        estimates = estimate_inherited_instrument(samples)
        assert estimates.approximation_error < _APPROXIMATION_TOLERANCE
        assert abs(estimates.inherited_strength - estimates.oracle_strength) < _STRENGTH_TOLERANCE
        assert abs(estimates.inherited_instrument - estimates.injected_noise) < _REFERENCE_TOLERANCE
        assert abs(estimates.inherited_instrument - effect) < _EFFECT_TOLERANCE
        assert abs(estimates.ordinary_least_squares - effect) > _CONFOUNDING_BIAS

    first_stage = FirstStagePlotter().plotted_series(runs)
    assert first_stage["x labels"] == ["Oracle", "Inherited"]
    assert len(first_stage["seed approximation errors"]) == _SEEDS
    effect_series = DownstreamEffectPlotter(effect=effect).plotted_series(runs)
    assert effect_series["true effect"] == effect
    errors = effect_series["error"]
    assert errors[0] > errors[2]
