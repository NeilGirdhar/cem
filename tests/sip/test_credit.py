from collections.abc import Mapping

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
from efax import NormalNP
from jax import tree
from tjax import JaxRealArray, RngStream

from cem.npn.gaussian import GaussianNeuralProjection
from cem.sip.credit import Credit, CreditEmission, CreditOutputs, predictor_substrate
from cem.sip.emitter import Emitter
from cem.sip.messages import InstrumentedMessage

from .tools import _clone, _message

# Larger than sampling error in the mean of 4096 recorded weight cotangents.
_NONZERO_MEAN = 0.5


def _baseline_projection(
    substrate_dim: int, streams: Mapping[str, RngStream]
) -> GaussianNeuralProjection:
    return GaussianNeuralProjection.create(
        in_features=substrate_dim,
        out_features=1,
        hidden_features=(),
        streams=streams,
    )


def _credit(substrate_dim: int, streams: Mapping[str, RngStream]) -> Credit:
    return Credit.create(
        _baseline_projection(substrate_dim, streams), _baseline_projection(substrate_dim, streams)
    )


def _predictor(observation_mean: float, instrument_mean: float) -> InstrumentedMessage:
    """Return a one-component predictor with unit-precision observation and instrument."""
    return InstrumentedMessage(
        observation=_message(jnp.asarray([observation_mean]), jnp.ones(1)),
        instrument=_message(jnp.asarray([instrument_mean]), jnp.ones(1)),
    )


def _emit(
    credit: Credit,
    *,
    inst_mean: JaxRealArray,
    inst_precision: JaxRealArray | None = None,
    gain: JaxRealArray | None = None,
    raw_mean: JaxRealArray | None = None,
    raw_precision: JaxRealArray | None = None,
    predictor: InstrumentedMessage | None = None,
) -> CreditEmission:
    n = inst_mean.shape[0]
    ones = jnp.ones(n)
    return credit.emit(
        gain_observation=_message(ones if gain is None else gain, ones),
        noised_instrument=_message(inst_mean, ones if inst_precision is None else inst_precision),
        predictor=_predictor(0.0, 0.0) if predictor is None else predictor,
        raw_observation=_message(
            jnp.zeros(n) if raw_mean is None else raw_mean,
            ones if raw_precision is None else raw_precision,
        ),
    )


def _mean_output(projection: GaussianNeuralProjection, substrate: NormalNP) -> JaxRealArray:
    """Recover a baseline the way Credit does, so tests do not depend on NPN init."""
    message = projection.infer(substrate)
    return message.mean_times_precision[0] / (-2 * message.negative_half_precision[0])


def test_credit_starts_with_nothing_retained(streams: Mapping[str, RngStream]) -> None:
    credit = _credit(2, streams)
    lag_filter = credit.lag_filter.value
    retained = eqx.nn.State(credit).get(credit.state_index)
    assert jnp.allclose(lag_filter.value(retained.quadratic_cost), 0.0)
    assert jnp.allclose(lag_filter.value(retained.emission_weight), 0.0)
    assert jnp.allclose(retained.predictor_substrate.negative_half_precision, jnp.zeros(2))


def test_predictor_substrate_subtracts_the_instrument_mean() -> None:
    """The substrate shifts each observation mean by its instrument's and keeps its precision."""
    predictor = InstrumentedMessage(
        observation=_message(jnp.asarray([1.5, -0.5, 2.0]), jnp.asarray([4.0, 0.5, 2.0])),
        instrument=_message(jnp.asarray([0.5, 1.0, 3.0]), jnp.asarray([2.0, 1.0, 0.0])),
    )
    substrate = predictor_substrate(predictor).to_precision_parametrization()
    # The third instrument is absent, so it subtracts nothing.
    assert jnp.allclose(substrate.mean, jnp.asarray([1.0, -1.5, 2.0]))
    assert jnp.allclose(substrate.precision, jnp.asarray([4.0, 0.5, 2.0]))


def test_emit_records_contributions(streams: Mapping[str, RngStream]) -> None:
    credit = _credit(1, streams)
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
        predictor=_predictor(1.25, 0.5),
    )

    # Gain enters each contribution, the cost, and the shrinkage weight once.
    assert jnp.allclose(emission.mean_contribution, gain * inst_precision * inst_mean)
    assert jnp.allclose(emission.precision_contribution, -0.5 * gain * jnp.square(inst_mean))
    assert jnp.allclose(
        emission.quadratic_cost, 0.5 * jnp.sum(raw_precision * gain * jnp.square(inst_mean))
    )
    assert jnp.allclose(emission.shrinkage_cotangent, -raw_mean * gain * raw_precision)
    assert jnp.allclose(emission.predictor_substrate.mean(), 0.75)


def test_residuals_use_the_previous_emission(streams: Mapping[str, RngStream]) -> None:
    """Both residuals use the substrate and cost of the emission one step before the goal."""
    credit = _credit(1, streams)
    inst_mean = jnp.asarray([1.0, 2.0])
    raw_precision = jnp.asarray([3.0, 0.5])
    first_predictor = _predictor(0.7, 0.2)
    later_predictor = _predictor(-0.4, 0.3)

    state, _ = credit.credit(
        eqx.nn.State(credit),
        _emit(credit, inst_mean=inst_mean, raw_precision=raw_precision, predictor=first_predictor),
        jnp.asarray(0.0),
    )
    emission = _emit(credit, inst_mean=jnp.zeros(2), predictor=later_predictor)
    goal = jnp.asarray(4.0)

    def residuals(target: JaxRealArray) -> tuple[JaxRealArray, JaxRealArray]:
        _, outputs = credit.credit(_clone(state), emission, target)
        return outputs.baseline_goal_residual, outputs.quadratic_goal_residual

    substrate = predictor_substrate(first_predictor)
    goal_baseline = _mean_output(credit.goal_baseline_map, substrate)
    cost_baseline = _mean_output(credit.quadratic_cost_baseline_map, substrate)
    cost = 0.5 * jnp.sum(raw_precision * jnp.square(inst_mean))
    baseline_residual, quadratic_residual = residuals(goal)
    assert jnp.allclose(baseline_residual, goal - goal_baseline)
    assert jnp.allclose(quadratic_residual, goal - goal_baseline + cost - cost_baseline)
    # Both residuals scale recorded cotangents, so neither passes gradients to the goal.
    assert all(jnp.allclose(gradient, 0.0) for gradient in jax.jacobian(residuals)(goal))


def test_baseline_losses_train_only_their_baseline_maps(
    streams: Mapping[str, RngStream],
) -> None:
    """The goal baseline fits the goal, the cost baseline fits the cost, and inputs stay fixed."""
    n = 1
    credit = _credit(1, streams)
    predictor = _predictor(1.0, 0.25)
    raw = _message(jnp.zeros(n), jnp.ones(n))

    def losses(
        inputs: tuple[Credit, InstrumentedMessage, NormalNP],
    ) -> tuple[JaxRealArray, JaxRealArray]:
        circuit, predictor_message, raw_observation = inputs
        state = eqx.nn.State(circuit)
        for goal in (0.0, 1.0):
            emission = circuit.emit(
                gain_observation=_message(jnp.ones(n), jnp.ones(n)),
                noised_instrument=_message(jnp.ones(n), jnp.ones(n)),
                predictor=predictor_message,
                raw_observation=raw_observation,
            )
            state, outputs = circuit.credit(state, emission, jnp.asarray(goal))
        return outputs.goal_baseline_loss, outputs.quadratic_cost_loss

    def goal_loss(inputs: tuple[Credit, InstrumentedMessage, NormalNP]) -> JaxRealArray:
        return losses(inputs)[0]

    def cost_loss(inputs: tuple[Credit, InstrumentedMessage, NormalNP]) -> JaxRealArray:
        return losses(inputs)[1]

    for loss, trained, untrained in (
        (goal_loss, "goal_baseline_map", "quadratic_cost_baseline_map"),
        (cost_loss, "quadratic_cost_baseline_map", "goal_baseline_map"),
    ):
        credit_gradient, predictor_gradient, raw_gradient = eqx.filter_grad(loss)(
            (credit, predictor, raw)
        )
        assert jnp.any(getattr(credit_gradient, trained).layers[-1].bias.value != 0)
        assert all(
            jnp.allclose(leaf, 0.0) for leaf in tree.leaves(getattr(credit_gradient, untrained))
        )
        assert all(
            jnp.allclose(leaf, 0.0) for leaf in tree.leaves((predictor_gradient, raw_gradient))
        )


def test_first_goal_trains_nothing(streams: Mapping[str, RngStream]) -> None:
    """A goal that follows no emission leaves both baseline maps untrained."""
    credit = _credit(1, streams)
    initial_state = eqx.nn.State(credit)

    def loss(circuit: Credit) -> JaxRealArray:
        emission = _emit(circuit, inst_mean=jnp.ones(1))
        _, outputs = circuit.credit(initial_state, emission, jnp.asarray(3.0))
        return outputs.goal_baseline_loss + outputs.quadratic_cost_loss

    value, gradient = eqx.filter_value_and_grad(loss)(credit)
    assert jnp.allclose(value, 0.0)
    assert all(jnp.allclose(leaf, 0.0) for leaf in tree.leaves(gradient))


def test_quadratic_cost_applies_only_to_the_next_goal(streams: Mapping[str, RngStream]) -> None:
    credit = _credit(1, streams)
    lag_filter = credit.lag_filter.value
    raw_precision = jnp.asarray([3.0])

    state, _ = credit.credit(
        eqx.nn.State(credit),
        _emit(credit, inst_mean=jnp.asarray([2.0]), raw_precision=raw_precision),
        jnp.asarray(0.0),
    )
    first_cost = lag_filter.value(state.get(credit.state_index).quadratic_cost)
    assert jnp.allclose(first_cost, 6.0)

    # An emission with no displacement replaces the cost rather than adding to it.
    state, _ = credit.credit(
        state,
        _emit(credit, inst_mean=jnp.zeros(1), raw_precision=raw_precision),
        jnp.asarray(0.0),
    )
    second_cost = lag_filter.value(state.get(credit.state_index).quadratic_cost)
    assert jnp.allclose(second_cost, 0.0)


def _with_affine(
    projection: GaussianNeuralProjection, weight: float, bias: float
) -> GaussianNeuralProjection:
    (layer,) = projection.layers
    projection = eqx.tree_at(
        lambda p: p.layers[0].weight.value, projection, jnp.full_like(layer.weight.value, weight)
    )
    return eqx.tree_at(
        lambda p: p.layers[0].bias.value, projection, jnp.full_like(layer.bias.value, bias)
    )


def _echo_emissions(emitter: Emitter, draws: JaxRealArray) -> tuple[CreditEmission, JaxRealArray]:
    """Emit once per upstream draw that the feature echoes.

    The feature's content, its parent's observation, and its parent's instrument all carry the
    draw precisely. Return each emission's credit record and its recorded weight cotangent.
    """

    def precise(value: JaxRealArray) -> NormalNP:
        return _message(value[jnp.newaxis], jnp.asarray([1e4]))

    def recorded(draw: JaxRealArray) -> tuple[CreditEmission, JaxRealArray]:
        _, emission = emitter.emit(
            eqx.nn.State(emitter),
            jr.key(0),
            observation_content=precise(draw),
            predictor=InstrumentedMessage(precise(draw), precise(draw)),
            gain_observation=None,
        )
        return emission.credit_emission, emission.mean_drift.layers[0].weight.value[0, 0]

    return jax.vmap(recorded)(draws)


def _credit_outputs(
    circuit: Credit, emissions: CreditEmission, goals: JaxRealArray
) -> CreditOutputs:
    """Return each emission's credit outputs when the goal that follows it arrives."""

    def one(emission: CreditEmission, goal: JaxRealArray) -> CreditOutputs:
        state, _ = circuit.credit(eqx.nn.State(circuit), emission, jnp.asarray(0.0))
        _, outputs = circuit.credit(state, emission, goal)
        return outputs

    return jax.vmap(one)(emissions, goals)


def _fit_goal_baseline(circuit: Credit, emissions: CreditEmission, goals: JaxRealArray) -> Credit:
    """Descend the mean goal baseline loss along the baseline's bias.

    With a constant substrate the baseline is its bias, so each step halves the gap between the
    baseline and the mean goal.
    """

    def with_bias(bias: JaxRealArray) -> Credit:
        return eqx.tree_at(lambda c: c.goal_baseline_map.layers[0].bias.value, circuit, bias)

    def batch_loss(bias: JaxRealArray) -> JaxRealArray:
        return jnp.mean(_credit_outputs(with_bias(bias), emissions, goals).goal_baseline_loss)

    descend = jax.jit(lambda bias: bias - 0.5 * jax.grad(batch_loss)(bias))
    bias = circuit.goal_baseline_map.layers[0].bias.value
    for _ in range(60):
        bias = descend(bias)
    return with_bias(bias)


def test_fitted_goal_baseline_removes_a_goal_offset_from_inherited_mean_credit(
    streams: Mapping[str, RngStream],
) -> None:
    """A goal offset leaves mean credit unchanged once the goal baseline has absorbed it.

    A non-controllable feature echoes a standard normal upstream draw A, and its maps pass A
    through, so its observation and inherited instrument are both A. The recorded weight cotangent
    is then proportional to A², whose mean is positive, so pairing it with the raw goal turns a
    constant offset into a weight update. The substrate is constant, so the fitted goal baseline
    is the mean goal, and the residual no longer depends on the offset.
    """
    emitter = Emitter.create(
        observation_map=_with_affine(_baseline_projection(1, streams), 1.0, 0.0),
        instrument_map=_with_affine(_baseline_projection(1, streams), 1.0, 0.0),
        credit=_credit(1, streams),
        controllable=False,
    )
    draws = jr.normal(jr.key(3), (4096,))
    emissions, weight_drifts = _echo_emissions(emitter, draws)
    goals = 0.5 * draws  # The goal rises with the echoed draw.

    def mean_credit(offset: float) -> tuple[JaxRealArray, JaxRealArray]:
        """Return the fitted and raw-goal mean weight credit for the offset goals."""
        offset_goals = goals + offset
        circuit = _fit_goal_baseline(emitter.credit, emissions, offset_goals)
        residuals = _credit_outputs(circuit, emissions, offset_goals).baseline_goal_residual
        assert jnp.allclose(jnp.mean(residuals), 0.0, atol=1e-9)
        return jnp.mean(residuals * weight_drifts), jnp.mean(offset_goals * weight_drifts)

    fitted, raw = mean_credit(0.0)
    offset_fitted, offset_raw = mean_credit(5.0)
    # The inherited draw changes the pullback, so the recorded weight cotangent has a positive mean.
    assert jnp.mean(weight_drifts) > _NONZERO_MEAN
    assert jnp.allclose(offset_fitted, fitted, atol=1e-6)
    assert jnp.allclose(offset_raw - raw, 5.0 * jnp.mean(weight_drifts))
