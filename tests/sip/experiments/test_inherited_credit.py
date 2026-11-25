"""Check test 10: an inherited feature's mean credit follows the goal's causal gradient."""

import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest
from efax import NormalPP
from jax import tree
from tjax import JaxRealArray, create_streams

from cem.commands.registry import experiments
from cem.sip.experiments.datasets import (
    FEATURE_OBSERVATION_VARIANCE,
    InheritedCreditDataSource,
    InheritedCreditState,
)
from cem.sip.experiments.emitter_credit import (
    CONFOUNDED_GAIN,
    CONTEXT_GAIN,
    EXOGENOUS_GAIN,
    INHERITED_CREDIT_SCENARIOS,
    UNIT_GAIN,
    InheritedCreditConfiguration,
    InheritedCreditExperiment,
    InheritedCreditModel,
    sip_inherited_credit_experiment,
)
from cem.sip.experiments.scenarios import InheritedCreditScenario

_SEEDS = 2
# Credit that follows its gradient lies within about 1.5° of it.
_ALIGNED = 4.0
# A constant baseline misses the context-dependent goal level by about 12°.
_CONSTANT_MISALIGNED = 8.0
# The hidden confounder tilts mean credit by about 39° from the gain-weighted gradient.
_CONFOUNDED_MISALIGNED = 20.0
# The goal offset turns raw-goal credit at least 37° from the gradient.
_RAW_MISALIGNED = 30.0
# Tolerances for the fitted first stage and goal baseline.
_FIRST_STAGE = 0.3
_INSTRUMENT_MEAN = 0.1
_BASELINE_SLOPE = 0.05


def test_inherited_credit_experiment_is_registered() -> None:
    assert experiments["sip-inherited-credit"] is sip_inherited_credit_experiment


def test_scenario_rejects_nonpositive_gain() -> None:
    with pytest.raises(ValueError, match="positive"):
        InheritedCreditScenario(base_gain=0.5, gain_confounding=0.3, gain_noise_width=0.2)


@pytest.mark.parametrize("condition", list(INHERITED_CREDIT_SCENARIOS))
def test_confounded_correlation_opposes_the_causal_effect(condition: str) -> None:
    scenario = INHERITED_CREDIT_SCENARIOS[condition]
    sample = scenario.samples(jr.key(0), 100_000)
    value = sample.context + sample.injected_noise + sample.confounder
    goal = np.asarray(scenario.goal(value, sample))
    slope = np.cov(goal, np.asarray(value))[0, 1] / np.var(np.asarray(value), ddof=1)
    assert scenario.causal_effect > 0
    assert slope == pytest.approx(-1 / 3, abs=0.03)


@pytest.mark.parametrize("condition", list(INHERITED_CREDIT_SCENARIOS))
def test_gain_weighted_content_mean_matches_samples(condition: str) -> None:
    scenario = INHERITED_CREDIT_SCENARIOS[condition]
    sample = scenario.samples(jr.key(1), 400_000)
    content = sample.context + sample.injected_noise + sample.confounder
    sampled = float(jnp.sum(sample.gain * content) / jnp.sum(sample.gain))
    assert scenario.gain_weighted_content_mean() == pytest.approx(sampled, abs=0.01)


def _state(condition: str) -> InheritedCreditState:
    scenario = INHERITED_CREDIT_SCENARIOS[condition]
    return InheritedCreditDataSource(scenario).initial_problem_state(jr.key(2))


def _model(condition: str) -> InheritedCreditModel:
    return InheritedCreditModel.create(
        INHERITED_CREDIT_SCENARIOS[condition], streams=create_streams({"parameters": jr.key(0)})
    )


def test_substrate_reveals_the_context_but_not_the_confounder() -> None:
    state = _state(CONFOUNDED_GAIN)
    model = _model(CONFOUNDED_GAIN)
    result = model.infer(
        state, None, streams=create_streams({"inference": jr.key(3)}), inference=True
    )
    configuration = result.configurations["feature"]
    assert isinstance(configuration, InheritedCreditConfiguration)
    assert float(configuration.substrate) == pytest.approx(float(state.sample.context))


def test_operating_point_centres_the_gain_weighted_raw_mean() -> None:
    scenario = INHERITED_CREDIT_SCENARIOS[CONTEXT_GAIN]
    model = _model(CONTEXT_GAIN)
    centre = NormalPP(
        jnp.full(1, scenario.gain_weighted_content_mean()),
        jnp.full(1, 1.0 / FEATURE_OBSERVATION_VARIANCE),
    ).to_nat()
    assert float(model.observation_map.infer(centre).mean()[0]) == pytest.approx(0.0, abs=1e-12)


def test_credit_scales_the_recorded_cotangent() -> None:
    """Mean credit uses the goal residual, and raw-goal credit uses the goal itself."""
    model = _model(EXOGENOUS_GAIN)
    result = model.infer(
        _state(EXOGENOUS_GAIN),
        None,
        streams=create_streams({"inference": jr.key(3)}),
        inference=True,
    )
    configuration = result.configurations["feature"]
    assert isinstance(configuration, InheritedCreditConfiguration)
    drift = configuration.mean_drift
    assert jnp.allclose(configuration.mean_credit, configuration.goal_residual * drift)
    assert jnp.allclose(configuration.raw_goal_credit, configuration.goal * drift)
    # The content carries the inherited draw, so the weight's derivative changes with the draw.
    names = model.parameter_names()
    jacobian = dict(zip(names, configuration.mean_jacobian.tolist(), strict=True))
    assert jacobian["layers[0].bias"] == pytest.approx(1.0)
    state = _state(EXOGENOUS_GAIN)
    content = float(state.content.mean()[0])
    assert jacobian["layers[0].weight"] == pytest.approx(100.0 * content / 101.0)


def test_training_fits_only_the_first_stage_and_baselines() -> None:
    """The loss trains the instrument map and both baselines and leaves the observation map."""
    state = _state(CONTEXT_GAIN)

    def loss(model: InheritedCreditModel) -> JaxRealArray:
        result = model.infer(
            state, None, streams=create_streams({"inference": jr.key(3)}), inference=False
        )
        return result.loss

    gradient = eqx.filter_grad(loss)(_model(CONTEXT_GAIN))
    assert all(jnp.allclose(leaf, 0.0) for leaf in tree.leaves(gradient.observation_map))
    for fitted in (
        gradient.instrument_map,
        gradient.goal_baseline_map,
        gradient.quadratic_cost_baseline_map,
    ):
        assert jnp.any(fitted.layers[0].bias.value != 0)


@pytest.mark.usefixtures("log")
def test_each_baseline_fixes_one_more_condition() -> None:
    """Raw-goal, constant-baseline, and substrate-baseline credit each fix one more condition.

    The goal offset defeats raw-goal credit everywhere. A constant baseline removes it, but under
    context gain the goal level the baseline misses varies with the gain, so it still fails. The
    substrate baseline captures that level, so only the confounded gain, which depends on a level
    the substrate cannot see, still defeats it. Observational credit opposes the causal gradient
    throughout.
    """
    experiment = InheritedCreditExperiment(seeds=_SEEDS)
    (plotter,) = experiment.plotters
    series = plotter.plotted_series(experiment.runs({}))
    scores = series["seed scores"]
    for seed_scores in scores.values():
        for score in seed_scores:
            assert score["confounded_slope"] < 0
            assert abs(score["first_stage_slope"] - 1.0) < _FIRST_STAGE
            assert abs(score["instrument_mean"]) < _INSTRUMENT_MEAN
            assert abs(score["baseline_slope"]) < _BASELINE_SLOPE
            assert score["observational"]["inner_product"] < 0
            assert score["raw_goal"]["angle"] > _RAW_MISALIGNED
    for condition in (UNIT_GAIN, CONTEXT_GAIN, EXOGENOUS_GAIN):
        assert all(score["target"]["angle"] < _ALIGNED for score in scores[condition])
    assert all(
        score["target"]["angle"] > _CONFOUNDED_MISALIGNED for score in scores[CONFOUNDED_GAIN]
    )
    for condition in (UNIT_GAIN, EXOGENOUS_GAIN):
        assert all(score["constant_baseline"]["angle"] < _ALIGNED for score in scores[condition])
    for condition in (CONTEXT_GAIN, CONFOUNDED_GAIN):
        assert all(
            score["constant_baseline"]["angle"] > _CONSTANT_MISALIGNED
            for score in scores[condition]
        )
