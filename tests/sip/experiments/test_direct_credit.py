"""Check test 9: a controllable feature's mean credit follows the goal's causal gradient."""

import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest
from efax import NormalNP
from jax import tree
from tjax import JaxRealArray, create_streams

from cem.commands.registry import experiments
from cem.sip.experiments.datasets import DirectCreditDataSource, DirectCreditState
from cem.sip.experiments.emitter_credit import (
    DIRECT_CREDIT_SCENARIOS,
    ENDOGENOUS_GAIN,
    UNIT_GAIN,
    DirectCreditConfiguration,
    DirectCreditExperiment,
    DirectCreditModel,
    sip_direct_credit_experiment,
)
from cem.sip.experiments.scenarios import DirectCreditScenario

_SEEDS = 2
# Both conditions' mean credit lies within about 1° of the gradient that it estimates.
_ALIGNED = 4.0
# Endogenous gain tilts mean credit about 13.5° from the unweighted causal gradient.
_TILT = (8.0, 20.0)
# Shrinkage credit is about as large as mean credit and points along the weight.
_SHRINKAGE_TILT = 30.0
# The fitted goal baseline lies within this distance of the held-out mean goal.
_BASELINE_FIT = 0.1


def test_direct_credit_experiment_is_registered() -> None:
    assert experiments["sip-direct-credit"] is sip_direct_credit_experiment


def test_scenario_rejects_nonpositive_gain() -> None:
    with pytest.raises(ValueError, match="positive"):
        DirectCreditScenario(base_gain=0.5, gain_confounding=0.3, gain_noise_width=0.2)


@pytest.mark.parametrize("condition", list(DIRECT_CREDIT_SCENARIOS))
def test_confounded_correlation_opposes_the_causal_effect(condition: str) -> None:
    scenario = DIRECT_CREDIT_SCENARIOS[condition]
    sample = scenario.samples(jr.key(0), 100_000)
    noise = jr.normal(jr.key(1), sample.confounder.shape)
    # The operating point's value tracks the confounder and adds unit-variance noise.
    value = sample.confounder + noise
    goal = np.asarray(scenario.goal(value, sample))
    slope = np.cov(goal, np.asarray(value))[0, 1] / np.var(np.asarray(value), ddof=1)
    assert scenario.causal_effect > 0
    assert slope == pytest.approx(-0.5, abs=0.03)


def _model(*, baseline_bias: float = 0.0) -> DirectCreditModel:
    model = DirectCreditModel.create(
        DIRECT_CREDIT_SCENARIOS[ENDOGENOUS_GAIN],
        observation_weight=1.0,
        streams=create_streams({"parameters": jr.key(0)}),
    )
    return eqx.tree_at(
        lambda m: m.goal_baseline_map.layers[0].bias.value,
        model,
        jnp.full(1, baseline_bias),
    )


def _state() -> DirectCreditState:
    return DirectCreditDataSource(DIRECT_CREDIT_SCENARIOS[ENDOGENOUS_GAIN]).initial_problem_state(
        jr.key(1)
    )


def test_mean_credit_is_the_goal_residual_times_the_instrument_pullback() -> None:
    model = _model(baseline_bias=0.7)
    state = _state()
    result = model.infer(
        state, None, streams=create_streams({"inference": jr.key(2)}), inference=True
    )
    configuration = result.configurations["feature"]
    assert isinstance(configuration, DirectCreditConfiguration)
    # The feature has no parents, so the baseline reads an absent substrate.
    absent = NormalNP(jnp.zeros(1), jnp.zeros(1))
    baseline = float(model.goal_baseline_map.infer(absent).mean()[0])
    residual = float(configuration.goal) - baseline
    assert float(configuration.goal_residual) == pytest.approx(residual)
    # The instrument has unit precision and carries the injected noise, the value's departure from
    # the raw mean, which is the confounder's pooled observation.
    pooled = 100.0 * float(state.sample.confounder) / 101.0
    value = float(configuration.value)
    gain = float(configuration.gain)
    expected = residual * gain * (value - pooled) * configuration.mean_jacobian
    assert jnp.allclose(configuration.mean_credit, expected, rtol=1e-6)
    # Observational credit pairs the same residual with the emitted value.
    observational = residual * gain * value * configuration.mean_jacobian
    assert jnp.allclose(configuration.observational_credit, observational, rtol=1e-6)
    names = model.parameter_names()
    jacobian = dict(zip(names, configuration.mean_jacobian.tolist(), strict=True))
    assert jacobian["layers[0].bias"] == pytest.approx(1.0)
    assert jacobian["layers[0].weight"] == pytest.approx(pooled)
    assert jacobian["layers[0].bias_raw_variance"] == 0.0


def test_training_fits_only_the_credit_baselines() -> None:
    """The model's loss trains the goal and cost baselines and leaves the operating point fixed."""
    state = _state()

    def loss(model: DirectCreditModel) -> JaxRealArray:
        result = model.infer(
            state, None, streams=create_streams({"inference": jr.key(2)}), inference=False
        )
        return result.loss

    gradient = eqx.filter_grad(loss)(_model(baseline_bias=0.7))
    for fixed in (gradient.observation_map, gradient.instrument_map):
        assert all(jnp.allclose(leaf, 0.0) for leaf in tree.leaves(fixed))
    for fitted in (gradient.goal_baseline_map, gradient.quadratic_cost_baseline_map):
        assert jnp.any(fitted.layers[0].bias.value != 0)


@pytest.mark.usefixtures("log")
def test_mean_credit_follows_the_causal_gradient() -> None:
    experiment = DirectCreditExperiment(seeds=_SEEDS)
    (plotter,) = experiment.plotters
    series = plotter.plotted_series(experiment.runs({}))
    scores = series["seed scores"]
    for seed_scores in scores.values():
        for score in seed_scores:
            assert score["confounded_slope"] < 0
            assert score["causal"]["inner_product"] > 0
            assert score["gain_weighted"]["angle"] < _ALIGNED
            assert score["observational"]["inner_product"] < 0
            assert score["with_shrinkage"]["angle"] > _SHRINKAGE_TILT
            assert abs(score["goal_baseline"] - score["mean_goal"]) < _BASELINE_FIT
    assert all(score["causal"]["angle"] < _ALIGNED for score in scores[UNIT_GAIN])
    low, high = _TILT
    assert all(low < score["causal"]["angle"] < high for score in scores[ENDOGENOUS_GAIN])
