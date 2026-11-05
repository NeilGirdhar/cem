import math
from collections.abc import Mapping

import equinox as eqx
import jax.numpy as jnp
import pytest
from tjax import RngStream

from cem.npn.gaussian import GaussianNPN
from cem.sip.credit import Credit, CreditTraces
from cem.sip.settings import CreditSettings

from .tools import CREDIT_SETTINGS, _message


def _baseline_npn(obs_content_dim: int, streams: Mapping[str, RngStream]) -> GaussianNPN:
    return GaussianNPN.create(
        in_features=obs_content_dim,
        out_features=1,
        hidden_features=(),
        streams=streams,
    )


def test_traces_zeros_initialization() -> None:
    traces = CreditTraces.zeros(3)
    assert jnp.allclose(traces.linear, jnp.zeros(3))
    assert jnp.allclose(traces.quadratic, jnp.zeros(3))
    assert jnp.allclose(traces.obs_precision, jnp.zeros(3))


@pytest.mark.parametrize("rate", [0.0, -0.1, math.inf, math.nan])
def test_credit_settings_reject_invalid_decay_rates(rate: float) -> None:
    with pytest.raises(ValueError, match="finite and positive"):
        CreditSettings(trace_decay_rate=rate)


def test_step_advances_traces_from_zero(streams: Mapping[str, RngStream]) -> None:
    n = 2
    credit = Credit.create(n, _baseline_npn(1, streams), credit_settings=CREDIT_SETTINGS)
    initial_state = eqx.nn.State(credit)

    inst_mean = jnp.asarray([2.0, -3.0])
    inst_precision = jnp.asarray([0.5, 4.0])
    gain = jnp.asarray([1.0, 0.5])
    raw_mean = jnp.asarray([1.0, 0.0])
    raw_precision = jnp.asarray([0.25, 0.75])

    new_state, _ = credit.step(
        initial_state,
        gain_observation=_message(gain, jnp.ones(n)),
        noised_instrument=_message(inst_mean, inst_precision),
        observation_content=_message(jnp.asarray([0.0]), jnp.asarray([1.0])),
        raw_observation=_message(raw_mean, raw_precision),
        goal=jnp.asarray(0.0),
    )
    traces = new_state.get(credit.trace_index)

    # The initial traces are zero, so this step contains only the new contribution.
    assert jnp.allclose(traces.linear, gain * (inst_precision * inst_mean))
    assert jnp.allclose(traces.quadratic, gain * jnp.square(inst_mean))
    assert jnp.allclose(traces.obs_precision, gain * raw_precision)


def test_step_mean_cotangent_equals_goal_times_linear_minus_shrinkage(
    streams: Mapping[str, RngStream],
) -> None:
    n = 2
    credit = Credit.create(n, _baseline_npn(1, streams), credit_settings=CREDIT_SETTINGS)
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

    expected_linear_trace = gain * (inst_precision * inst_mean)
    expected_obs_precision_trace = gain * raw_precision
    expected_mean_cotangent = goal * expected_linear_trace - raw_mean * expected_obs_precision_trace
    assert jnp.allclose(outputs.cotangent[:, 0], expected_mean_cotangent)


def test_step_precision_cotangent_uses_quadratic_trace_and_prediction_error(
    streams: Mapping[str, RngStream],
) -> None:
    n = 2
    credit = Credit.create(n, _baseline_npn(1, streams), credit_settings=CREDIT_SETTINGS)
    initial_state = eqx.nn.State(credit)

    inst_mean = jnp.asarray([1.0, 2.0])
    inst_precision = jnp.asarray([1.0, 1.0])
    gain = jnp.asarray([1.0, 1.0])
    raw_mean = jnp.asarray([0.0, 0.0])
    raw_precision = jnp.asarray([1.0, 1.0])
    content_mean = jnp.asarray([0.0])
    content_precision = jnp.asarray([1.0])
    goal = jnp.asarray(4.0)

    _, outputs = credit.step(
        initial_state,
        gain_observation=_message(gain, jnp.ones(n)),
        noised_instrument=_message(inst_mean, inst_precision),
        observation_content=_message(content_mean, content_precision),
        raw_observation=_message(raw_mean, raw_precision),
        goal=goal,
    )

    # Recover baseline the way Credit does, so the test does not depend on NPN init.
    baseline_message = credit.goal_baseline_map.infer(content_mean, content_precision)
    baseline = baseline_message.mean_times_precision[0] / (
        -2 * baseline_message.negative_half_precision[0]
    )
    expected_quadratic_trace = gain * jnp.square(inst_mean)
    expected_prediction = baseline - 0.5 * jnp.sum(raw_precision * expected_quadratic_trace)
    expected_error = goal - expected_prediction
    expected_precision_cotangent = -0.5 * expected_quadratic_trace * expected_error

    assert jnp.allclose(outputs.cotangent[:, 1], expected_precision_cotangent)


def test_step_loss_is_scalar_and_promotes_baseline(
    streams: Mapping[str, RngStream],
) -> None:
    n = 1
    credit = Credit.create(n, _baseline_npn(1, streams), credit_settings=CREDIT_SETTINGS)
    initial_state = eqx.nn.State(credit)

    _, outputs = credit.step(
        initial_state,
        gain_observation=_message(jnp.ones(n), jnp.ones(n)),
        noised_instrument=_message(jnp.ones(n), jnp.ones(n)),
        observation_content=_message(jnp.asarray([1.0]), jnp.asarray([1.0])),
        raw_observation=_message(jnp.zeros(n), jnp.ones(n)),
        goal=jnp.asarray(1.0),
    )

    assert outputs.loss.shape == ()


def test_decay_reduces_previous_trace(streams: Mapping[str, RngStream]) -> None:
    n = 1
    decay_rate = jnp.log(2.0).item()  # exp(-ln 2) = 0.5
    credit = Credit.create(
        n, _baseline_npn(1, streams), credit_settings=CreditSettings(trace_decay_rate=decay_rate)
    )
    state = eqx.nn.State(credit)

    inst_mean = jnp.asarray([2.0])
    inst_precision = jnp.asarray([1.0])
    gain = jnp.asarray([1.0])
    raw_precision = jnp.asarray([0.0])

    # Step once to populate traces.
    state, _ = credit.step(
        state,
        gain_observation=_message(gain, jnp.ones(n)),
        noised_instrument=_message(inst_mean, inst_precision),
        observation_content=_message(jnp.asarray([0.0]), jnp.asarray([1.0])),
        raw_observation=_message(jnp.zeros(n), raw_precision + 1e-8),
        goal=jnp.asarray(0.0),
    )
    first_linear = state.get(credit.trace_index).linear

    # Step a second time with zero contribution so only decay acts.
    state, _ = credit.step(
        state,
        gain_observation=_message(gain, jnp.ones(n)),
        noised_instrument=_message(jnp.zeros(n), jnp.ones(n)),
        observation_content=_message(jnp.asarray([0.0]), jnp.asarray([1.0])),
        raw_observation=_message(jnp.zeros(n), raw_precision + 1e-8),
        goal=jnp.asarray(0.0),
    )
    second_linear = state.get(credit.trace_index).linear

    assert jnp.allclose(second_linear, 0.5 * first_linear)
