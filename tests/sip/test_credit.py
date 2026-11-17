from collections.abc import Mapping

import equinox as eqx
import jax.numpy as jnp
from efax import NormalNP
from jax import tree, value_and_grad
from tjax import JaxRealArray, RngStream

from cem.npn.gaussian import GaussianNeuralProjection
from cem.sip.credit import Credit, CreditEmission

from .tools import _message


def _baseline_projection(
    obs_content_dim: int, streams: Mapping[str, RngStream]
) -> GaussianNeuralProjection:
    return GaussianNeuralProjection.create(
        in_features=obs_content_dim,
        out_features=1,
        hidden_features=(),
        streams=streams,
    )


def _emit(
    credit: Credit,
    *,
    inst_mean: JaxRealArray,
    inst_precision: JaxRealArray | None = None,
    gain: JaxRealArray | None = None,
    raw_mean: JaxRealArray | None = None,
    raw_precision: JaxRealArray | None = None,
    content: NormalNP | None = None,
) -> CreditEmission:
    n = inst_mean.shape[0]
    ones = jnp.ones(n)
    return credit.emit(
        gain_observation=_message(ones if gain is None else gain, ones),
        noised_instrument=_message(inst_mean, ones if inst_precision is None else inst_precision),
        observation_content=(
            _message(jnp.asarray([0.0]), jnp.asarray([1.0])) if content is None else content
        ),
        raw_observation=_message(
            jnp.zeros(n) if raw_mean is None else raw_mean,
            ones if raw_precision is None else raw_precision,
        ),
    )


def _baseline(credit: Credit, content: NormalNP) -> JaxRealArray:
    """Recover the baseline the way Credit does, so tests do not depend on NPN init."""
    message = credit.goal_baseline_map.infer(content)
    return message.mean_times_precision[0] / (-2 * message.negative_half_precision[0])


def test_credit_starts_with_nothing_retained(streams: Mapping[str, RngStream]) -> None:
    credit = Credit.create(_baseline_projection(2, streams))
    lag_filter = credit.lag_filter.value
    retained = eqx.nn.State(credit).get(credit.state_index)
    assert jnp.allclose(lag_filter.value(retained.goal_penalty), 0.0)
    assert jnp.allclose(lag_filter.value(retained.emission_weight), 0.0)
    assert jnp.allclose(retained.observation_content.negative_half_precision, jnp.zeros(2))


def test_emit_records_contributions(streams: Mapping[str, RngStream]) -> None:
    credit = Credit.create(_baseline_projection(1, streams))
    inst_mean = jnp.asarray([2.0, -3.0])
    inst_precision = jnp.asarray([0.5, 4.0])
    gain = jnp.asarray([1.0, 0.5])
    raw_mean = jnp.asarray([1.0, -2.0])
    raw_precision = jnp.asarray([0.25, 0.75])

    emission = _emit(
        credit,
        inst_mean=inst_mean,
        inst_precision=inst_precision,
        gain=gain,
        raw_mean=raw_mean,
        raw_precision=raw_precision,
    )

    assert jnp.allclose(emission.mean_contribution, gain * inst_precision * inst_mean)
    assert jnp.allclose(emission.precision_contribution, -0.5 * gain * jnp.square(inst_mean))
    assert jnp.allclose(
        emission.goal_penalty, jnp.sum(raw_precision * emission.precision_contribution)
    )
    # Shrinkage is weighted by the emission's own gain-weighted raw precision.
    assert jnp.allclose(emission.shrinkage_cotangent, -raw_mean * gain * raw_precision)


def test_goal_prediction_uses_the_previous_emission(streams: Mapping[str, RngStream]) -> None:
    """The goal is predicted from the content and penalty of the emission one step before it."""
    n = 2
    credit = Credit.create(_baseline_projection(1, streams))
    inst_mean = jnp.asarray([1.0, 2.0])
    first_content = _message(jnp.asarray([0.7]), jnp.asarray([1.0]))
    later_content = _message(jnp.asarray([-0.4]), jnp.asarray([2.0]))

    state, _ = credit.credit(
        eqx.nn.State(credit),
        _emit(credit, inst_mean=inst_mean, content=first_content),
        jnp.asarray(0.0),
    )
    emission = _emit(credit, inst_mean=jnp.zeros(n), content=later_content)
    goal = jnp.asarray(4.0)

    def prediction_error(target: JaxRealArray) -> JaxRealArray:
        _, outputs = credit.credit(state, emission, target)
        return outputs.goal_prediction_error

    expected_penalty = -0.5 * jnp.sum(jnp.square(inst_mean))
    expected_error = goal - (_baseline(credit, first_content) + expected_penalty)
    error, goal_gradient = value_and_grad(prediction_error)(goal)
    assert jnp.allclose(error, expected_error)
    assert jnp.allclose(goal_gradient, 0.0)


def test_goal_prediction_loss_trains_only_the_baseline_map(
    streams: Mapping[str, RngStream],
) -> None:
    """The loss holds the observation content and the goal penalty fixed."""
    n = 1
    credit = Credit.create(_baseline_projection(1, streams))
    initial_state = eqx.nn.State(credit)

    content = _message(jnp.ones(n), jnp.ones(n))
    raw = _message(jnp.zeros(n), jnp.ones(n))

    def loss(inputs: tuple[Credit, NormalNP, NormalNP]) -> tuple[JaxRealArray, JaxRealArray]:
        circuit, observation_content, raw_observation = inputs
        state = initial_state
        for goal in (0.0, 1.0):
            emission = circuit.emit(
                gain_observation=_message(jnp.ones(n), jnp.ones(n)),
                noised_instrument=_message(jnp.ones(n), jnp.ones(n)),
                observation_content=observation_content,
                raw_observation=raw_observation,
            )
            state, outputs = circuit.credit(state, emission, jnp.asarray(goal))
        return outputs.goal_prediction_loss, outputs.goal_prediction_error

    (value, error), (credit_gradient, content_gradient, raw_gradient) = eqx.filter_value_and_grad(
        loss, has_aux=True
    )((credit, content, raw))
    assert jnp.allclose(value, 0.5 * jnp.square(error))
    assert all(jnp.allclose(leaf, 0.0) for leaf in tree.leaves((content_gradient, raw_gradient)))
    assert jnp.any(credit_gradient.goal_baseline_map.layers[-1].bias.value != 0)


def test_first_goal_trains_nothing(streams: Mapping[str, RngStream]) -> None:
    """A goal that follows no emission leaves the baseline map untrained."""
    credit = Credit.create(_baseline_projection(1, streams))
    initial_state = eqx.nn.State(credit)

    def loss(circuit: Credit) -> JaxRealArray:
        emission = _emit(circuit, inst_mean=jnp.ones(1))
        _, outputs = circuit.credit(initial_state, emission, jnp.asarray(3.0))
        return outputs.goal_prediction_loss

    value, gradient = eqx.filter_value_and_grad(loss)(credit)
    assert jnp.allclose(value, 0.0)
    assert all(jnp.allclose(leaf, 0.0) for leaf in tree.leaves(gradient))


def test_goal_penalty_applies_only_to_the_next_goal(streams: Mapping[str, RngStream]) -> None:
    credit = Credit.create(_baseline_projection(1, streams))
    lag_filter = credit.lag_filter.value
    raw_precision = jnp.asarray([3.0])

    state, _ = credit.credit(
        eqx.nn.State(credit),
        _emit(credit, inst_mean=jnp.asarray([2.0]), raw_precision=raw_precision),
        jnp.asarray(0.0),
    )
    first_penalty = lag_filter.value(state.get(credit.state_index).goal_penalty)
    assert jnp.allclose(first_penalty, -6.0)

    # An emission with no contribution replaces the penalty rather than adding to it.
    state, _ = credit.credit(
        state,
        _emit(credit, inst_mean=jnp.zeros(1), raw_precision=raw_precision),
        jnp.asarray(0.0),
    )
    second_penalty = lag_filter.value(state.get(credit.state_index).goal_penalty)
    assert jnp.allclose(second_penalty, 0.0)
