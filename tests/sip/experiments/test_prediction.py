"""Check test 4: reconstruction reaches OLS, and purification reaches the causal effect."""

import pytest

from cem.commands.registry import experiments
from cem.sip.experiments.prediction import (
    CAUSAL_MEAN_SCENARIOS,
    CausalMeanExperiment,
    CausalMeanPlotter,
    PurificationSolver,
    run_prediction_estimates,
    sip_causal_mean_experiment,
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


def test_causal_mean_experiment_is_registered() -> None:
    assert experiments["sip-causal-mean"] is sip_causal_mean_experiment


@pytest.mark.usefixtures("log")
def test_purification_reaches_the_causal_effect() -> None:
    solver = PurificationSolver(training_examples=_TRAINING_EXAMPLES)
    runs = CausalMeanExperiment(solver, seeds=_SEEDS).runs({})

    for condition, scenario in CAUSAL_MEAN_SCENARIOS.items():
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
    assert [p.condition for p in plotters] == list(CAUSAL_MEAN_SCENARIOS)
    series = plotters[1].plotted_series(runs)
    assert series["x labels"] == ["OLS", "Reconstruction", "Purification", "2SLS, noise"]
    assert series["bar plots"] == {"parent 1": "Parent 1", "parent 2": "Parent 2"}
    for parent in ("parent 1", "parent 2"):
        ordinary, reconstruction, purification, _ = series[parent]
        assert purification < reconstruction
        assert purification < ordinary
