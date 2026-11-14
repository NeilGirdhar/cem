"""Check tests 4–6: purification reaches causal means, variances, and nonlinear effects."""

import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

from cem.commands.registry import experiments
from cem.sip.experiments.datasets import FEATURE_OBSERVATION_VARIANCE
from cem.sip.experiments.prediction import (
    NONLINEAR_MISMATCH_SETTINGS,
    NONLINEAR_SCENARIO,
    PURIFICATION_SCENARIOS,
    SIEVE_FIT_EXAMPLES,
    CausalMeanExperiment,
    CausalMeanPlotter,
    CausalVarianceExperiment,
    CausalVariancePlotter,
    NonlinearCurvesPlotter,
    NonlinearEffectExperiment,
    PredictionSamples,
    PredictionTelemetry,
    PurificationSolver,
    estimate_nonlinear,
    fit_sieve,
    run_predicted_variance,
    run_prediction_estimates,
    sieve_fit_sample,
    sip_causal_mean_experiment,
    sip_causal_variance_experiment,
    sip_nonlinear_effect_experiment,
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
# Larger steps reach the structural function sooner, before the tails drift.
_NONLINEAR_TRAINING_EXAMPLES = 40_000
_NONLINEAR_LEARNING_RATE = 0.003
# Purification's projected error is 0.1 to 0.2 on these seeds; reconstruction's is about 0.86.
_PROJECTED_TOLERANCE = 0.3


def test_purification_experiments_are_registered() -> None:
    assert experiments["sip-causal-mean"] is sip_causal_mean_experiment
    assert experiments["sip-causal-variance"] is sip_causal_variance_experiment
    assert experiments["sip-nonlinear-effect"] is sip_nonlinear_effect_experiment


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


def test_sieve_is_scored_on_samples_it_was_not_fitted_on() -> None:
    scenario = NONLINEAR_SCENARIO
    fit = sieve_fit_sample(scenario, 0)
    assert fit.target.shape == (SIEVE_FIT_EXAMPLES,)
    assert not np.array_equal(fit.parents, sieve_fit_sample(scenario, 1).parents)
    draws = scenario.samples(jr.key(0), 1024)
    structural = scenario.structural_mean(draws.parents)
    evaluation = PredictionSamples(
        predicted_mean=structural,
        predicted_variance=jnp.ones_like(structural),
        injected_noise=draws.injected_noise,
        parents=draws.parents,
        target=draws.target,
    )
    coefficients = fit_sieve(fit)
    estimates = estimate_nonlinear(evaluation, scenario, coefficients)
    assert estimates.learned == pytest.approx(0.0, abs=1e-6)
    # The evaluation targets play no part in any score.
    blanked = eqx.tree_at(lambda s: s.target, evaluation, jnp.zeros_like(evaluation.target))
    assert estimate_nonlinear(blanked, scenario, coefficients) == estimates
    # The sieve's score depends on its fit sample.
    refitted = estimate_nonlinear(evaluation, scenario, fit_sieve(sieve_fit_sample(scenario, 1)))
    assert refitted.sieve != estimates.sieve


@pytest.mark.usefixtures("log")
def test_purification_matches_a_nonlinear_effect() -> None:
    solver = PurificationSolver(
        scenario=NONLINEAR_SCENARIO,
        mismatch_settings=NONLINEAR_MISMATCH_SETTINGS,
        training_examples=_NONLINEAR_TRAINING_EXAMPLES,
        learning_rate=_NONLINEAR_LEARNING_RATE,
    )
    experiment = NonlinearEffectExperiment(solver, seeds=_SEEDS)
    runs = experiment.runs({})

    errors_plotter, curves_plotter = experiment.plotters
    errors = errors_plotter.plotted_series(runs)
    assert errors["x labels"] == ["Observational", "Reconstruction", "Purification", "Sieve 2SLS"]
    _, reconstruction, purification, _ = errors["projected"]
    assert purification < _PROJECTED_TOLERANCE
    assert purification < reconstruction / 2
    _, reconstruction_error, purification_error, _ = errors["error"]
    assert purification_error < reconstruction_error / 2
    assert errors["sieve fit examples"] == SIEVE_FIT_EXAMPLES
    for run in runs:
        held_out = np.asarray(run.inference_results.telemetries[PredictionTelemetry()].parents)
        assert errors["evaluation examples"] == held_out.size
        fit = np.asarray(sieve_fit_sample(solver.scenario, run.sweep_entry["seed"]).parents)
        assert not np.isin(fit, held_out).any()

    assert isinstance(curves_plotter, NonlinearCurvesPlotter)
    curves = curves_plotter.plotted_series(runs)
    assert set(curves["line plots"]) == {
        "structural",
        "observational",
        "reconstruction",
        "purification",
    }
    assert len(curves["purification"]) == len(curves["iteration"])
