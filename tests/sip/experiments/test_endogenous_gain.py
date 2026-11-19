"""Check test 8: the full mismatch circuit under an additive gain that depends on the confounder."""

import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest
from tjax import JaxRealArray

from cem.commands.registry import experiments
from cem.sip.experiments.datasets import FEATURE_OBSERVATION_VARIANCE
from cem.sip.experiments.prediction import (
    ENDOGENOUS_GAIN,
    ENDOGENOUS_GAIN_SCENARIOS,
    GAIN_CONTROL,
    EndogenousGainExperiment,
    EndogenousGainSolver,
    sip_endogenous_gain_experiment,
)
from cem.sip.experiments.scenarios import EndogenousGainScenario, GainMoments

_CONTROL = ENDOGENOUS_GAIN_SCENARIOS[GAIN_CONTROL]
_ENDOGENOUS = ENDOGENOUS_GAIN_SCENARIOS[ENDOGENOUS_GAIN]
_GAIN_NOISE = jnp.asarray([-0.2, 0.0, 0.2])
_SEEDS = 2
_TRAINING_EXAMPLES = 160_000
# Both conditions fit the slope to within about 0.02.
_SLOPE_TOLERANCE = 0.06
# The control's mean moment falls below about 0.016; the endogenous mean moment stays near 0.056.
_SETTLED_MOMENT = 0.03
# The proportional witness's mean coordinate correlates about 0.93 with its population target.
_WITNESS_CORRELATION = 0.8


def _moments(
    scenario: EndogenousGainScenario, intercept: float, variance: float, gain_noise: JaxRealArray
) -> GainMoments:
    return scenario.population_moments(
        slope=1.5,
        intercept=intercept,
        variance=variance,
        observation_variance=FEATURE_OBSERVATION_VARIANCE,
        parent_noise=jnp.zeros_like(gain_noise),
        gain_noise=gain_noise,
    )


def test_endogenous_gain_experiment_is_registered() -> None:
    assert experiments["sip-endogenous-gain"] is sip_endogenous_gain_experiment


def test_population_moments_match_monte_carlo() -> None:
    draws = _ENDOGENOUS.samples(jr.key(0), 400_000)
    prediction = draws.prediction
    # Condition on one instrument value by replacing every draw's instruments with it.
    parent_noise, gain_noise = 0.7, -0.1
    parents = prediction.parents[:, 0] - prediction.injected_noise[:, 0] + parent_noise
    gain = draws.gain - draws.gain_noise + gain_noise
    target = (
        1.5 * parents
        - 2.0 * prediction.confounder
        + (prediction.target - 1.5 * prediction.parents[:, 0] + 2.0 * prediction.confounder)
    )
    slope, intercept, variance = 1.3, 0.2, 3.5
    residual = target - (slope * parents + intercept)
    variance_residual = FEATURE_OBSERVATION_VARIANCE + jnp.square(residual) - variance
    moments = _ENDOGENOUS.population_moments(
        slope=slope,
        intercept=intercept,
        variance=variance,
        observation_variance=FEATURE_OBSERVATION_VARIANCE,
        parent_noise=jnp.asarray(parent_noise),
        gain_noise=jnp.asarray(gain_noise),
    )
    assert float(moments.mean) == pytest.approx(float(jnp.mean(gain * residual)), abs=0.01)
    assert float(moments.variance) == pytest.approx(
        float(jnp.mean(gain * variance_residual)), abs=0.05
    )


def test_control_moments_vanish_at_the_causal_prediction() -> None:
    moments = _moments(_CONTROL, 0.0, 4.26, _GAIN_NOISE)
    assert jnp.allclose(moments.mean, 0.0, atol=1e-6)
    assert jnp.allclose(moments.variance, 0.0, atol=1e-6)


def test_endogenous_gain_calls_for_an_intercept_that_varies() -> None:
    weighted_mean = _ENDOGENOUS.gain_weighted_confounder_mean(_GAIN_NOISE)
    assert bool(jnp.all(jnp.diff(weighted_mean) < 0))
    # At a zero intercept, the variance moment vanishes but the mean moment does not.
    shared = _moments(_ENDOGENOUS, 0.0, 4.26, _GAIN_NOISE)
    assert jnp.allclose(shared.variance, 0.0, atol=1e-5)
    # The mean moment is then -2 E[f(U) U] at every instrument value.
    assert jnp.allclose(shared.mean, shared.mean[0])
    assert float(shared.mean[0]) < 0
    # Each instrument value's own intercept and variance zero both moments there.
    for noise, mean in zip(_GAIN_NOISE, weighted_mean, strict=True):
        own = _moments(
            _ENDOGENOUS, float(-2.0 * mean), float(4.26 - 4.0 * mean**2), noise[jnp.newaxis]
        )
        assert jnp.allclose(own.mean, 0.0, atol=1e-5)
        assert jnp.allclose(own.variance, 0.0, atol=1e-5)


@pytest.mark.usefixtures("log")
def test_endogenous_gain_leaves_moments_the_slope_survives() -> None:
    experiment = EndogenousGainExperiment(
        EndogenousGainSolver(training_examples=_TRAINING_EXAMPLES), seeds=_SEEDS
    )
    runs = experiment.runs({})
    scores_plotter, conflict_plotter = experiment.plotters
    scores = scores_plotter.plotted_series(runs)["seed scores"]
    for condition in (GAIN_CONTROL, ENDOGENOUS_GAIN):
        assert len(scores[condition]) == _SEEDS
        for score in scores[condition]:
            assert abs(score["fit"]["slope"] - 1.5) < _SLOPE_TOLERANCE
    # Slope bias alone leaves both moments near 0.02 to 0.06, so only the mean moment separates
    # the conditions: the endogenous gain's intercept conflict keeps it near 0.06.
    for control, endogenous in zip(scores[GAIN_CONTROL], scores[ENDOGENOUS_GAIN], strict=True):
        assert control["mean_moment"] < _SETTLED_MOMENT < endogenous["mean_moment"]
        assert endogenous["mean_correlation"] > _WITNESS_CORRELATION
        # The intercept lies within the range that the gain instrument's values call for.
        called_for = -2.0 * _ENDOGENOUS.gain_weighted_confounder_mean(_GAIN_NOISE)
        assert float(called_for[0]) < endogenous["fit"]["intercept"] < float(called_for[-1])
    conflict = conflict_plotter.plotted_series(runs)
    intercepts = np.asarray(conflict[f"{ENDOGENOUS_GAIN} intercept"])
    assert np.all(np.diff(intercepts) > 0)
    assert np.allclose(conflict[f"{GAIN_CONTROL} intercept"], 0.0)
