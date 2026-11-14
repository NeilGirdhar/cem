"""Check test 6: what the proportional witness and the variance residual each contribute."""

from dataclasses import replace

import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
import pytest
from jax import tree
from tjax import create_streams

from cem.commands.registry import experiments
from cem.sip import MismatchSettings
from cem.sip.experiments.datasets import (
    ConfoundedPredictionDataSource,
    ConfoundedPredictionObservation,
    ConfoundedPredictionProblem,
)
from cem.sip.experiments.prediction import (
    DAMPING_ARMS,
    INTEGRAL_AND_PROPORTIONAL,
    INTEGRAL_ONLY,
    PROPORTIONAL_ONLY,
    VARIANCE_MISMATCH_SETTINGS,
    FixedMeanVarianceModel,
    FixedMeanVarianceSolver,
    PurificationModel,
    WitnessDiagnosticsExperiment,
    sip_witness_diagnostics_experiment,
)
from cem.sip.experiments.scenarios import ConfoundedPredictionScenario

_SCENARIO = ConfoundedPredictionScenario()
_SEEDS = 2
_VARIANCE_TRAINING_EXAMPLES = 40_000
# Both witnesses settle within about 0.02 of the effect; integral only still swings by 0.1 to 0.3.
_SETTLED_ERROR = 0.1
# Proportional only settles about 0.375 short of the effect, as a finite quadratic penalty.
_PENALTY_BIAS = 0.25
# Reconstruction learns a slope of about 0.8 in y²; purification's slope is within about 0.06 of 0.
_OBSERVATIONAL_SLOPE = 0.6
_INTERVENTIONAL_SLOPE = 0.15


def _observation() -> ConfoundedPredictionObservation:
    state = ConfoundedPredictionDataSource(_SCENARIO).initial_problem_state(jr.key(0))
    return ConfoundedPredictionProblem(_SCENARIO).extract_observation(state)


def _gradient[M: eqx.Module](model: M, observation: ConfoundedPredictionObservation) -> M:
    """Return the loss gradient, giving every model the same inference streams."""

    def loss(model: M) -> jnp.ndarray:
        streams = create_streams({"inference": jr.key(2)})
        result = model.infer(observation, None, streams=streams, inference=False)  # ty: ignore
        return result.loss

    return eqx.filter_grad(loss)(model)


def _purification_model(*, purify: bool, integral: bool) -> PurificationModel:
    return PurificationModel.create(
        1,
        purify=purify,
        mismatch_settings=MismatchSettings(damping_weight=0.0),
        streams=create_streams({"parameters": jr.key(1)}),
        integral=integral,
    )


def _all_zero(gradient: eqx.Module) -> bool:
    return all(bool(jnp.all(leaf == 0)) for leaf in tree.leaves(gradient))


def test_dropping_the_integral_term_changes_only_the_prediction_update() -> None:
    observation = _observation()
    gradients = {
        (purify, integral): _gradient(
            _purification_model(purify=purify, integral=integral), observation
        )
        for purify, integral in ((False, True), (True, True), (True, False))
    }
    reconstruction = gradients[False, True].mismatch.prediction_map
    # With zero damping and no integral term, only reconstruction trains the prediction.
    proportional_only = gradients[True, False].mismatch
    assert tree.all(tree.map(jnp.allclose, proportional_only.prediction_map, reconstruction))
    assert not tree.all(
        tree.map(jnp.allclose, gradients[True, True].mismatch.prediction_map, reconstruction)
    )
    # The integral witness still trains on its own loss.
    assert not _all_zero(proportional_only.integral_witness_map)
    assert tree.all(
        tree.map(
            jnp.allclose,
            proportional_only.integral_witness_map,
            gradients[True, True].mismatch.integral_witness_map,
        )
    )


def test_fixed_mean_variance_model_purifies_only_the_variance() -> None:
    observation = _observation()
    model = FixedMeanVarianceModel.create(
        effect=1.5,
        purify=True,
        mismatch_settings=VARIANCE_MISMATCH_SETTINGS,
        streams=create_streams({"parameters": jr.key(1)}),
    )
    result = model.infer(
        observation, None, streams=create_streams({"inference": jr.key(2)}), inference=False
    )
    parent = observation.predictor.observation.to_variance_parametrization().mean[0]
    assert result.configurations["target"].predicted_mean == pytest.approx(1.5 * parent)  # ty: ignore

    gradient = _gradient(model, observation)
    assert gradient.variance_slope.value != 0
    for witness_map in (gradient.integral_witness_map, gradient.proportional_witness_map):
        # The output rows hold each channel's mean and variance coordinates in turn.
        mean_row, variance_row = witness_map.perceptron.layers[-1].weight.value
        assert jnp.all(mean_row == 0)
        assert jnp.any(variance_row != 0)

    reconstruction = _gradient(replace(model, purify=False), observation)
    assert reconstruction.variance_slope.value != 0
    assert _all_zero(reconstruction.integral_witness_map)
    assert _all_zero(reconstruction.proportional_witness_map)


def test_observational_reference_matches_its_closed_form() -> None:
    parents = jnp.asarray([[0.0], [1.0], [2.0]])
    # Var(U | Y) = 1 - 1 / 2.25 and E[U | Y] = Y / 2.25, scaled by 2², plus the disturbance's 0.5².
    expected = 4.0 * (1.0 - 1.0 / 2.25) + 0.25 + 4.0 * jnp.square(parents[:, 0] / 2.25)
    assert jnp.allclose(_SCENARIO.causal_residual_second_moment(parents), expected)
    # At E[Y²] = 2.25, the reference equals the interventional variance, so it averages to it.
    at_mean_square = _SCENARIO.causal_residual_second_moment(jnp.asarray([[1.5]]))
    assert float(at_mean_square[0]) == pytest.approx(_SCENARIO.interventional_variance())


def test_witness_diagnostics_experiment_is_registered() -> None:
    assert experiments["sip-witness-diagnostics"] is sip_witness_diagnostics_experiment


@pytest.mark.usefixtures("log")
def test_each_witness_mechanism_does_its_job() -> None:
    experiment = WitnessDiagnosticsExperiment(
        variance_solver=FixedMeanVarianceSolver(
            mismatch_settings=VARIANCE_MISMATCH_SETTINGS,
            training_examples=_VARIANCE_TRAINING_EXAMPLES,
        ),
        seeds=_SEEDS,
    )
    runs = experiment.runs({})
    damping_plotter, variance_plotter = experiment.plotters

    damping = damping_plotter.plotted_series(runs)
    errors = damping["late signed error"]
    variability = damping["late variability"]
    assert list(errors) == list(DAMPING_ARMS)
    assert all(abs(error) < _SETTLED_ERROR for error in errors[INTEGRAL_AND_PROPORTIONAL])
    assert all(error < -_PENALTY_BIAS for error in errors[PROPORTIONAL_ONLY])
    assert sum(variability[INTEGRAL_ONLY]) > sum(variability[INTEGRAL_AND_PROPORTIONAL])
    assert len(damping[f"{INTEGRAL_ONLY} 0"]) == len(damping["iteration"])

    variance = variance_plotter.plotted_series(runs)
    assert all(slope > _OBSERVATIONAL_SLOPE for slope in variance["slope"]["reconstruction"])
    assert all(abs(slope) < _INTERVENTIONAL_SLOPE for slope in variance["slope"]["purification"])
    reconstruction_error = variance["grid error"]["reconstruction"]
    purification_error = variance["grid error"]["purification"]
    assert all(
        purified < reconstructed / 2
        for purified, reconstructed in zip(purification_error, reconstruction_error, strict=True)
    )
    assert variance["interventional variance"] == pytest.approx(4.26)
