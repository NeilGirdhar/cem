"""Check tests 4 and 5: purification reaches the causal mean and interventional variance."""

import pytest

from cem.commands.registry import experiments
from cem.sip.experiments.datasets import FEATURE_OBSERVATION_VARIANCE
from cem.sip.experiments.prediction import (
    PURIFICATION_SCENARIOS,
    CausalMeanExperiment,
    CausalMeanPlotter,
    CausalVarianceExperiment,
    CausalVariancePlotter,
    PurificationSolver,
    run_predicted_variance,
    run_prediction_estimates,
    sip_causal_mean_experiment,
    sip_causal_variance_experiment,
    variant_label,
)

_SEEDS = 2
_TRAINING_EXAMPLES = 20_000
# Reconstruction fits the held-out OLS coefficients up to their sampling error of about 0.03.
_OLS_TOLERANCE = 0.05
# Purification lands within about 0.1 of the effect after 20,000 episodes.
_EFFECT_TOLERANCE = 0.15
# Two-stage least squares on the injected noise has a standard error of about 0.05.
_REFERENCE_TOLERANCE = 0.1
# Ordinary least squares misses every effect by at least 0.5.
_CONFOUNDING_BIAS = 0.3
_VARIANCE_TRAINING_EXAMPLES = 40_000
# Reconstruction's variance matches the observational variance to within about 0.02.
_OBSERVATIONAL_TOLERANCE = 0.05
# Purification's variance lands within about 0.2 of the interventional variance after 40,000
# episodes; the observational variance is at least 1.7 below it.
_INTERVENTIONAL_TOLERANCE = 0.4


def test_purification_experiments_are_registered() -> None:
    assert experiments["sip-causal-mean"] is sip_causal_mean_experiment
    assert experiments["sip-causal-variance"] is sip_causal_variance_experiment


def test_scenario_variances_match_their_closed_forms() -> None:
    one_parent, two_parents = PURIFICATION_SCENARIOS.values()
    # The confounder contributes 2² and the disturbance 0.5².
    assert one_parent.interventional_variance() == pytest.approx(4.25)
    # Var(Z) = 3.3125 and Cov(Y, Z) = 1.375 with Var(Y) = 2.25.
    assert one_parent.observational_variance() == pytest.approx(3.3125 - 1.375**2 / 2.25)
    assert two_parents.observational_variance() < one_parent.observational_variance()


@pytest.mark.usefixtures("log")
def test_purification_reaches_the_causal_effect() -> None:
    solver = PurificationSolver(training_examples=_TRAINING_EXAMPLES)
    runs = CausalMeanExperiment(solver, seeds=_SEEDS).runs({})

    for condition, scenario in PURIFICATION_SCENARIOS.items():
        for purify in (False, True):
            label = variant_label(condition, purify=purify)
            condition_runs = [run for run in runs if run.variant_label == label]
            assert len(condition_runs) == _SEEDS
            for run in condition_runs:
                estimates = run_prediction_estimates(run)
                for learned, ordinary, reference, effect in zip(
                    estimates.learned,
                    estimates.ordinary_least_squares,
                    estimates.injected_noise,
                    scenario.effects,
                    strict=True,
                ):
                    assert abs(reference - effect) < _REFERENCE_TOLERANCE
                    assert abs(ordinary - effect) > _CONFOUNDING_BIAS
                    if purify:
                        assert abs(learned - effect) < _EFFECT_TOLERANCE
                    else:
                        assert abs(learned - ordinary) < _OLS_TOLERANCE

    plotters = [p for p in sip_causal_mean_experiment.plotters if isinstance(p, CausalMeanPlotter)]
    assert [p.condition for p in plotters] == list(PURIFICATION_SCENARIOS)
    series = plotters[1].plotted_series(runs)
    assert series["x labels"] == ["OLS", "Reconstruction", "Purification", "2SLS, noise"]
    assert series["bar plots"] == {"parent 1": "Parent 1", "parent 2": "Parent 2"}
    for parent in ("parent 1", "parent 2"):
        ordinary, reconstruction, purification, _ = series[parent]
        assert purification < reconstruction
        assert purification < ordinary


@pytest.mark.usefixtures("log")
def test_purification_reaches_the_interventional_variance() -> None:
    solver = PurificationSolver(training_examples=_VARIANCE_TRAINING_EXAMPLES)
    runs = CausalVarianceExperiment(solver, seeds=_SEEDS).runs({})

    for condition, scenario in PURIFICATION_SCENARIOS.items():
        observational = scenario.observational_variance() + FEATURE_OBSERVATION_VARIANCE
        interventional = scenario.interventional_variance() + FEATURE_OBSERVATION_VARIANCE
        for purify, expected, tolerance in (
            (False, observational, _OBSERVATIONAL_TOLERANCE),
            (True, interventional, _INTERVENTIONAL_TOLERANCE),
        ):
            label = variant_label(condition, purify=purify)
            condition_runs = [run for run in runs if run.variant_label == label]
            assert len(condition_runs) == _SEEDS
            for run in condition_runs:
                assert abs(run_predicted_variance(run) - expected) < tolerance

    plotters = [
        p for p in sip_causal_variance_experiment.plotters if isinstance(p, CausalVariancePlotter)
    ]
    assert [p.condition for p in plotters] == list(PURIFICATION_SCENARIOS)
    series = plotters[0].plotted_series(runs)
    assert series["x labels"] == [
        "Observational",
        "Reconstruction",
        "Purification",
        "Interventional",
    ]
    observational, _, purification, interventional = series["variance"]
    assert abs(purification - interventional) < abs(purification - observational)
