from collections.abc import Mapping
from typing import TypedDict

import equinox as eqx
import jax.numpy as jnp
from efax import NormalNP
from tjax import RngStream

from cem.sip import InstrumentedMessage
from cem.sip.messages import mean_and_precision
from cem.sip.settings import TDSettings
from cem.sip.td_error import TDError

from .tools import _i_message, _message


class _StepInputs(TypedDict):
    reward_observation: NormalNP
    predictor: InstrumentedMessage


def _td_error(n: int, streams: Mapping[str, RngStream], *, discount: float = 1.0) -> TDError:
    return TDError.create(
        n,
        predictor_features=2,
        td_settings=TDSettings(discount=discount),
        streams=streams,
    )


def _inputs(n: int) -> _StepInputs:
    return _StepInputs(
        reward_observation=_message(jnp.ones(n), jnp.ones(n)),
        predictor=_i_message(
            jnp.asarray([0.3, -0.2]),
            jnp.asarray([1.0, 1.0]),
            jnp.asarray([0.5, 0.1]),
            jnp.asarray([1.0, 1.0]),
        ),
    )


def test_advantage_equals_bootstrap_minus_delayed_mean(
    streams: Mapping[str, RngStream],
) -> None:
    n = 2
    td = _td_error(n, streams)
    state = eqx.nn.State(td)
    inputs = _inputs(n)

    # Capture initial delayed inputs before stepping (Equinox invalidates state
    # after step()).
    delayed = state.get(td.delay_index)
    _, outputs = td.step(**inputs, state=state, streams=streams, inference=True)

    delayed_prediction = td.prediction_map.infer(*mean_and_precision(delayed.predictor.observation))
    current_prediction = td.prediction_map.infer(
        *mean_and_precision(inputs["predictor"].observation)
    )
    reward_mean, _ = mean_and_precision(inputs["reward_observation"])
    cur_mean, _ = mean_and_precision(current_prediction)
    del_mean, _ = mean_and_precision(delayed_prediction)

    expected_advantage = reward_mean + td.discount.value * cur_mean - del_mean
    assert jnp.allclose(outputs.advantage, expected_advantage)


def test_state_delays_predictor_inputs_by_one_step(
    streams: Mapping[str, RngStream],
) -> None:
    n = 2
    td = _td_error(n, streams)
    state = eqx.nn.State(td)
    inputs = _inputs(n)

    new_state, _ = td.step(**inputs, state=state, streams=streams, inference=True)
    stored = new_state.get(td.delay_index)
    assert jnp.allclose(
        stored.predictor.observation.mean_times_precision,
        inputs["predictor"].observation.mean_times_precision,
    )
    assert jnp.allclose(
        stored.predictor.instrument.mean_times_precision,
        inputs["predictor"].instrument.mean_times_precision,
    )


def test_second_step_delayed_prediction_matches_first_step_current_prediction(
    streams: Mapping[str, RngStream],
) -> None:
    n = 2
    td = _td_error(n, streams)
    state = eqx.nn.State(td)
    first = _inputs(n)
    state, first_outputs = td.step(**first, state=state, streams=streams, inference=True)

    # Second step uses different inputs; its delayed_prediction should come from
    # the first step's predictor_observations.
    second = _StepInputs(
        reward_observation=_message(jnp.zeros(n), jnp.ones(n)),
        predictor=_i_message(
            jnp.asarray([-0.5, 0.4]),
            jnp.asarray([1.0, 1.0]),
            jnp.asarray([0.2, -0.3]),
            jnp.asarray([1.0, 1.0]),
        ),
    )
    _, second_outputs = td.step(**second, state=state, streams=streams, inference=True)

    # On step two, delayed = first step's predictors → delayed prediction equals
    # what step one computed as its current prediction (recoverable from step
    # one's advantage).
    first_current = td.prediction_map.infer(*mean_and_precision(first["predictor"].observation))
    first_cur_mean, _ = mean_and_precision(first_current)
    # second advantage = reward + gamma * (second's current mean) - (first's current mean).
    second_current = td.prediction_map.infer(*mean_and_precision(second["predictor"].observation))
    second_cur_mean, _ = mean_and_precision(second_current)
    reward_mean, _ = mean_and_precision(second["reward_observation"])
    expected = reward_mean + td.discount.value * second_cur_mean - first_cur_mean
    assert jnp.allclose(second_outputs.advantage, expected)
    del first_outputs


def test_reconstruction_cost_trains_prediction_map(
    streams: Mapping[str, RngStream],
) -> None:
    n = 2
    td = _td_error(n, streams)
    state = eqx.nn.State(td)
    inputs = _inputs(n)

    def loss_fn(m: TDError) -> jnp.ndarray:
        _, out = m.step(**inputs, state=state, streams=streams, inference=True)
        return out.reconstruction_cost

    grads = eqx.filter_grad(loss_fn)(td)
    assert jnp.any(jnp.abs(grads.prediction_map.layers[-1].weight.value) > 0)


def test_uncertainty_loss_trains_prediction_map(
    streams: Mapping[str, RngStream],
) -> None:
    """The uncertainty penalty applies suppressing cotangent to delayed variance."""
    n = 2
    td = _td_error(n, streams)
    state = eqx.nn.State(td)
    inputs = _inputs(n)

    def loss_fn(m: TDError) -> jnp.ndarray:
        _, out = m.step(**inputs, state=state, streams=streams, inference=True)
        return out.uncertainty_loss

    grads = eqx.filter_grad(loss_fn)(td)
    assert jnp.any(jnp.abs(grads.prediction_map.layers[-1].weight.value) > 0)


def test_confounding_cost_does_not_train_witness_map(
    streams: Mapping[str, RngStream],
) -> None:
    n = 2
    td = _td_error(n, streams)
    state = eqx.nn.State(td)
    inputs = _inputs(n)

    def loss_fn(m: TDError) -> jnp.ndarray:
        _, out = m.step(**inputs, state=state, streams=streams, inference=True)
        return out.confounding_cost

    grads = eqx.filter_grad(loss_fn)(td)
    assert jnp.allclose(grads.witness_map.layers[-1].weight.value, 0.0)


def test_witness_loss_does_not_train_prediction_map(
    streams: Mapping[str, RngStream],
) -> None:
    n = 2
    td = _td_error(n, streams)
    state = eqx.nn.State(td)
    inputs = _inputs(n)

    def loss_fn(m: TDError) -> jnp.ndarray:
        _, out = m.step(**inputs, state=state, streams=streams, inference=True)
        return out.witness_loss

    grads = eqx.filter_grad(loss_fn)(td)
    assert jnp.allclose(grads.prediction_map.layers[-1].weight.value, 0.0)


def test_witness_loss_trains_witness_map(streams: Mapping[str, RngStream]) -> None:
    n = 2
    td = _td_error(n, streams)
    state = eqx.nn.State(td)
    inputs = _inputs(n)

    def loss_fn(m: TDError) -> jnp.ndarray:
        _, out = m.step(**inputs, state=state, streams=streams, inference=True)
        return out.witness_loss

    grads = eqx.filter_grad(loss_fn)(td)
    assert jnp.any(jnp.abs(grads.witness_map.layers[-1].weight.value) > 0)


def test_discount_scales_current_prediction_contribution_to_advantage(
    streams: Mapping[str, RngStream],
) -> None:
    """At gamma=1, advantage - (at gamma=0) equals the current prediction mean."""
    n = 2
    inputs = _inputs(n)

    td_zero = _td_error(n, streams, discount=0.0)
    state_zero = eqx.nn.State(td_zero)
    _, out_zero = td_zero.step(**inputs, state=state_zero, streams=streams, inference=True)

    td_one = eqx.tree_at(lambda m: m.discount.value, td_zero, jnp.asarray(1.0, dtype=jnp.float64))
    state_one = eqx.nn.State(td_one)
    _, out_one = td_one.step(**inputs, state=state_one, streams=streams, inference=True)

    current_prediction = td_zero.prediction_map.infer(
        *mean_and_precision(inputs["predictor"].observation)
    )
    cur_mean, _ = mean_and_precision(current_prediction)
    assert jnp.allclose(out_one.advantage - out_zero.advantage, cur_mean)
