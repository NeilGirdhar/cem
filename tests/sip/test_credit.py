import math
from collections.abc import Mapping

import equinox as eqx
import jax.numpy as jnp
import pytest
from efax import NormalNP
from jax import tree, value_and_grad
from tjax import JaxRealArray, RngStream

from cem.npn.gaussian import GaussianNeuralProjection
from cem.sip.credit import Credit, CreditOutputs, CreditTraces
from cem.sip.settings import CreditSettings

from .tools import CREDIT_SETTINGS, _message


def _credit(
    n: int, m: int, streams: Mapping[str, RngStream], settings: CreditSettings = CREDIT_SETTINGS
) -> Credit:
    baseline_map = GaussianNeuralProjection.create(
        in_features=1, out_features=m, hidden_features=(), streams=streams
    )
    return Credit.create(n, m, baseline_map, credit_settings=settings)


def _step(  # ruff: ignore[too-many-arguments]
    credit: Credit,
    state: eqx.nn.State,
    *,
    inst_mean: JaxRealArray,
    gain: JaxRealArray,
    raw_mean: JaxRealArray,
    curvature: JaxRealArray,
    goals: JaxRealArray,
    inst_precision: JaxRealArray | None = None,
    content: NormalNP | None = None,
    raw_observation: NormalNP | None = None,
) -> tuple[eqx.nn.State, CreditOutputs]:
    """Step the circuit with a raw precision equal to the sum of the component curvatures."""
    n = inst_mean.shape[0]
    return credit.step(
        state,
        gain_observation=_message(gain, jnp.ones(n)),
        noised_instrument=_message(
            inst_mean, jnp.ones(n) if inst_precision is None else inst_precision
        ),
        observation_content=_message(jnp.zeros(1), jnp.ones(1)) if content is None else content,
        raw_observation=(
            _message(raw_mean, jnp.sum(curvature, axis=0))
            if raw_observation is None
            else raw_observation
        ),
        curvature=curvature,
        goals=goals,
    )


def test_traces_zeros_initialization() -> None:
    traces = CreditTraces.zeros(3, 2)
    assert jnp.array_equal(traces.goal_penalty, jnp.zeros(2))
    assert jnp.array_equal(traces.curvature, jnp.zeros((2, 3)))


@pytest.mark.parametrize("rate", [0.0, -0.1, math.inf, math.nan])
def test_credit_settings_reject_invalid_decay_rates(rate: float) -> None:
    with pytest.raises(ValueError, match="finite and positive"):
        CreditSettings(trace_decay_rate=rate)


def test_step_advances_traces_from_zero(streams: Mapping[str, RngStream]) -> None:
    credit = _credit(2, 2, streams)
    inst_mean = jnp.asarray([2.0, -3.0])
    inst_precision = jnp.asarray([0.5, 4.0])
    gain = jnp.asarray([1.0, 0.5])
    curvature = jnp.asarray([[0.25, 0.5], [0.0, 0.25]])

    new_state, outputs = _step(
        credit,
        eqx.nn.State(credit),
        inst_mean=inst_mean,
        inst_precision=inst_precision,
        gain=gain,
        raw_mean=jnp.asarray([1.0, 0.0]),
        curvature=curvature,
        goals=jnp.zeros(2),
    )
    traces = new_state.get(credit.trace_index)

    # The initial traces are zero, so this step contains only the new contribution.
    assert jnp.allclose(outputs.mean_contribution, gain * inst_precision * inst_mean)
    assert jnp.allclose(outputs.curvature_contribution, -0.5 * gain * jnp.square(inst_mean))
    assert jnp.allclose(traces.goal_penalty, curvature @ outputs.curvature_contribution)
    assert jnp.allclose(traces.curvature, gain[jnp.newaxis, :] * curvature)


def test_component_shrinkage_sums_to_precision_weighted_shrinkage(
    streams: Mapping[str, RngStream],
) -> None:
    credit = _credit(2, 2, streams)
    gain = jnp.asarray([1.0, 0.5])
    raw_mean = jnp.asarray([3.0, -2.0])
    curvature = jnp.asarray([[0.5, 1.0], [0.25, 0.5]])

    _, outputs = _step(
        credit,
        eqx.nn.State(credit),
        inst_mean=jnp.asarray([2.0, -1.0]),
        gain=gain,
        raw_mean=raw_mean,
        curvature=curvature,
        goals=jnp.asarray([2.5, -1.0]),
    )

    curvature_trace = gain[jnp.newaxis, :] * curvature
    assert jnp.allclose(outputs.shrinkage_cotangent, -raw_mean[jnp.newaxis, :] * curvature_trace)
    raw_precision = jnp.sum(curvature, axis=0)
    assert jnp.allclose(
        jnp.sum(outputs.shrinkage_cotangent, axis=0), -raw_mean * gain * raw_precision
    )


def test_step_goal_prediction_error_uses_each_goal_penalty(
    streams: Mapping[str, RngStream],
) -> None:
    credit = _credit(2, 2, streams)
    inst_mean = jnp.asarray([1.0, 2.0])
    gain = jnp.ones(2)
    curvature = jnp.asarray([[1.0, 0.0], [0.5, 2.0]])
    content = _message(jnp.zeros(1), jnp.ones(1))
    goals = jnp.asarray([4.0, -1.0])

    def prediction_error(targets: JaxRealArray) -> tuple[JaxRealArray, JaxRealArray]:
        _, outputs = _step(
            credit,
            eqx.nn.State(credit),
            inst_mean=inst_mean,
            gain=gain,
            raw_mean=jnp.zeros(2),
            curvature=curvature,
            goals=targets,
            content=content,
        )
        return jnp.sum(outputs.goal_prediction_error), outputs.goal_prediction_error

    baseline = credit.goal_baseline_map.infer(content).mean()
    expected_prediction = baseline - 0.5 * curvature @ (gain * jnp.square(inst_mean))
    (_, error), goal_gradient = value_and_grad(prediction_error, has_aux=True)(goals)
    assert jnp.allclose(error, goals - expected_prediction)
    assert jnp.allclose(goal_gradient, 0.0)


def test_goal_prediction_loss_trains_only_the_baseline_map(
    streams: Mapping[str, RngStream],
) -> None:
    """The loss holds the observation content, the curvatures, and the goal penalties fixed."""
    credit = _credit(1, 2, streams)

    def loss(
        inputs: tuple[Credit, NormalNP, NormalNP, JaxRealArray],
    ) -> tuple[JaxRealArray, JaxRealArray]:
        circuit, observation_content, raw_observation, curvature = inputs
        _, outputs = _step(
            circuit,
            eqx.nn.State(circuit),
            inst_mean=jnp.ones(1),
            gain=jnp.ones(1),
            raw_mean=jnp.zeros(1),
            curvature=curvature,
            goals=jnp.asarray([1.0, -2.0]),
            content=observation_content,
            raw_observation=raw_observation,
        )
        return outputs.goal_prediction_loss, outputs.goal_prediction_error

    inputs = (
        credit,
        _message(jnp.ones(1), jnp.ones(1)),
        _message(jnp.zeros(1), jnp.ones(1)),
        jnp.asarray([[0.25], [0.75]]),
    )
    (value, error), (credit_gradient, *input_gradients) = eqx.filter_value_and_grad(
        loss, has_aux=True
    )(inputs)
    assert jnp.allclose(value, 0.5 * jnp.sum(jnp.square(error)))
    assert all(jnp.allclose(leaf, 0.0) for leaf in tree.leaves(input_gradients))
    assert jnp.any(credit_gradient.goal_baseline_map.layers[-1].bias.value != 0)


def test_goal_penalty_retains_curvature_from_emission(streams: Mapping[str, RngStream]) -> None:
    decay_rate = jnp.log(2.0).item()  # exp(-ln 2) = 0.5
    credit = _credit(1, 2, streams, CreditSettings(trace_decay_rate=decay_rate))
    curvature = jnp.asarray([[3.0], [1.0]])

    # Step once to populate traces.
    state, _ = _step(
        credit,
        eqx.nn.State(credit),
        inst_mean=jnp.asarray([2.0]),
        gain=jnp.ones(1),
        raw_mean=jnp.zeros(1),
        curvature=curvature,
        goals=jnp.zeros(2),
    )
    first_penalty = state.get(credit.trace_index).goal_penalty
    assert jnp.allclose(first_penalty, jnp.asarray([-6.0, -2.0]))

    # Step a second time with zero contribution so only decay acts.
    state, _ = _step(
        credit,
        state,
        inst_mean=jnp.zeros(1),
        gain=jnp.ones(1),
        raw_mean=jnp.zeros(1),
        curvature=2 * curvature,
        goals=jnp.zeros(2),
    )
    second_penalty = state.get(credit.trace_index).goal_penalty
    assert jnp.allclose(second_penalty, 0.5 * first_penalty)
