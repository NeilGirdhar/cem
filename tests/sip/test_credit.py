import math
from collections.abc import Mapping

import equinox as eqx
import jax.numpy as jnp
import pytest
from efax import NormalNP
from jax import tree, value_and_grad
from tjax import JaxRealArray, RngStream

from cem.npn.gaussian import GaussianNeuralProjection
from cem.sip.credit import Credit, CreditTraces
from cem.sip.settings import CreditSettings

from .tools import CREDIT_SETTINGS, _message


def _baseline_projection(
    obs_content_dim: int, streams: Mapping[str, RngStream]
) -> GaussianNeuralProjection:
    return GaussianNeuralProjection.create(
        in_features=obs_content_dim,
        out_features=1,
        hidden_features=(),
        streams=streams,
    )


def test_traces_zeros_initialization() -> None:
    traces = CreditTraces.zeros(3)
    assert traces.goal_penalty.shape == ()
    assert traces.goal_penalty == 0
    assert jnp.allclose(traces.obs_precision, jnp.zeros(3))


@pytest.mark.parametrize("rate", [0.0, -0.1, math.inf, math.nan])
def test_credit_settings_reject_invalid_decay_rates(rate: float) -> None:
    with pytest.raises(ValueError, match="finite and positive"):
        CreditSettings(trace_decay_rate=rate)


def test_step_advances_traces_from_zero(streams: Mapping[str, RngStream]) -> None:
    n = 2
    credit = Credit.create(n, _baseline_projection(1, streams), credit_settings=CREDIT_SETTINGS)
    initial_state = eqx.nn.State(credit)

    inst_mean = jnp.asarray([2.0, -3.0])
    inst_precision = jnp.asarray([0.5, 4.0])
    gain = jnp.asarray([1.0, 0.5])
    raw_mean = jnp.asarray([1.0, 0.0])
    raw_precision = jnp.asarray([0.25, 0.75])

    new_state, outputs = credit.step(
        initial_state,
        gain_observation=_message(gain, jnp.ones(n)),
        noised_instrument=_message(inst_mean, inst_precision),
        observation_content=_message(jnp.asarray([0.0]), jnp.asarray([1.0])),
        raw_observation=_message(raw_mean, raw_precision),
        goal=jnp.asarray(0.0),
    )
    traces = new_state.get(credit.trace_index)

    # The initial traces are zero, so this step contains only the new contribution.
    assert jnp.allclose(outputs.mean_contribution, gain * inst_precision * inst_mean)
    assert jnp.allclose(outputs.precision_contribution, -0.5 * gain * jnp.square(inst_mean))
    assert jnp.allclose(
        traces.goal_penalty, jnp.sum(raw_precision * outputs.precision_contribution)
    )
    assert jnp.allclose(traces.obs_precision, gain * raw_precision)


def test_step_shrinkage_uses_observation_precision_trace(
    streams: Mapping[str, RngStream],
) -> None:
    n = 2
    credit = Credit.create(n, _baseline_projection(1, streams), credit_settings=CREDIT_SETTINGS)
    initial_state = eqx.nn.State(credit)

    inst_mean = jnp.asarray([2.0, -1.0])
    inst_precision = jnp.asarray([1.0, 2.0])
    gain = jnp.asarray([1.0, 1.0])
    raw_mean = jnp.asarray([3.0, -2.0])
    raw_precision = jnp.asarray([0.5, 1.5])
    goal = jnp.asarray(2.5)

    _, outputs = credit.step(
        initial_state,
        gain_observation=_message(gain, jnp.ones(n)),
        noised_instrument=_message(inst_mean, inst_precision),
        observation_content=_message(jnp.asarray([0.0]), jnp.asarray([1.0])),
        raw_observation=_message(raw_mean, raw_precision),
        goal=goal,
    )

    expected_obs_precision_trace = gain * raw_precision
    assert jnp.allclose(outputs.shrinkage_cotangent, -raw_mean * expected_obs_precision_trace)


def test_step_goal_prediction_error_uses_goal_penalty(
    streams: Mapping[str, RngStream],
) -> None:
    n = 2
    credit = Credit.create(n, _baseline_projection(1, streams), credit_settings=CREDIT_SETTINGS)
    initial_state = eqx.nn.State(credit)

    inst_mean = jnp.asarray([1.0, 2.0])
    inst_precision = jnp.asarray([1.0, 1.0])
    gain = jnp.asarray([1.0, 1.0])
    raw_mean = jnp.asarray([0.0, 0.0])
    raw_precision = jnp.asarray([1.0, 1.0])
    content_mean = jnp.asarray([0.0])
    content_precision = jnp.asarray([1.0])
    goal = jnp.asarray(4.0)

    def prediction_error(target: JaxRealArray) -> JaxRealArray:
        _, outputs = credit.step(
            initial_state,
            gain_observation=_message(gain, jnp.ones(n)),
            noised_instrument=_message(inst_mean, inst_precision),
            observation_content=_message(content_mean, content_precision),
            raw_observation=_message(raw_mean, raw_precision),
            goal=target,
        )
        return outputs.goal_prediction_error

    # Recover baseline the way Credit does, so the test does not depend on NPN init.
    baseline_message = credit.goal_baseline_map.infer(_message(content_mean, content_precision))
    baseline = baseline_message.mean_times_precision[0] / (
        -2 * baseline_message.negative_half_precision[0]
    )
    expected_quadratic_trace = gain * jnp.square(inst_mean)
    expected_prediction = baseline - 0.5 * jnp.sum(raw_precision * expected_quadratic_trace)
    expected_error = goal - expected_prediction
    error, goal_gradient = value_and_grad(prediction_error)(goal)
    assert jnp.allclose(error, expected_error)
    assert jnp.allclose(goal_gradient, 0.0)


def test_goal_prediction_loss_trains_only_the_baseline_map(
    streams: Mapping[str, RngStream],
) -> None:
    """The loss holds the observation content and the goal penalty fixed."""
    n = 1
    credit = Credit.create(n, _baseline_projection(1, streams), credit_settings=CREDIT_SETTINGS)
    initial_state = eqx.nn.State(credit)

    content = _message(jnp.ones(n), jnp.ones(n))
    raw = _message(jnp.zeros(n), jnp.ones(n))

    def loss(inputs: tuple[Credit, NormalNP, NormalNP]) -> tuple[JaxRealArray, JaxRealArray]:
        circuit, observation_content, raw_observation = inputs
        _, outputs = circuit.step(
            initial_state,
            gain_observation=_message(jnp.ones(n), jnp.ones(n)),
            noised_instrument=_message(jnp.ones(n), jnp.ones(n)),
            observation_content=observation_content,
            raw_observation=raw_observation,
            goal=jnp.asarray(1.0),
        )
        return outputs.goal_prediction_loss, outputs.goal_prediction_error

    (value, error), (credit_gradient, content_gradient, raw_gradient) = eqx.filter_value_and_grad(
        loss, has_aux=True
    )((credit, content, raw))
    assert jnp.allclose(value, 0.5 * jnp.square(error))
    assert all(jnp.allclose(leaf, 0.0) for leaf in tree.leaves((content_gradient, raw_gradient)))
    assert jnp.any(credit_gradient.goal_baseline_map.layers[-1].bias.value != 0)


def test_goal_penalty_retains_precision_from_emission(streams: Mapping[str, RngStream]) -> None:
    n = 1
    decay_rate = jnp.log(2.0).item()  # exp(-ln 2) = 0.5
    credit = Credit.create(
        n,
        _baseline_projection(1, streams),
        credit_settings=CreditSettings(trace_decay_rate=decay_rate),
    )
    state = eqx.nn.State(credit)

    inst_mean = jnp.asarray([2.0])
    inst_precision = jnp.asarray([1.0])
    gain = jnp.asarray([1.0])
    raw_precision = jnp.asarray([3.0])

    # Step once to populate traces.
    state, _ = credit.step(
        state,
        gain_observation=_message(gain, jnp.ones(n)),
        noised_instrument=_message(inst_mean, inst_precision),
        observation_content=_message(jnp.asarray([0.0]), jnp.asarray([1.0])),
        raw_observation=_message(jnp.zeros(n), raw_precision),
        goal=jnp.asarray(0.0),
    )
    first_penalty = state.get(credit.trace_index).goal_penalty
    assert jnp.allclose(first_penalty, -6.0)

    # Step a second time with zero contribution so only decay acts.
    state, _ = credit.step(
        state,
        gain_observation=_message(gain, jnp.ones(n)),
        noised_instrument=_message(jnp.zeros(n), jnp.ones(n)),
        observation_content=_message(jnp.asarray([0.0]), jnp.asarray([1.0])),
        raw_observation=_message(jnp.zeros(n), 2 * raw_precision),
        goal=jnp.asarray(0.0),
    )
    second_penalty = state.get(credit.trace_index).goal_penalty
    assert jnp.allclose(second_penalty, 0.5 * first_penalty)
