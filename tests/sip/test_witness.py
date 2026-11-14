"""Check the witnesses, their losses, and the multiplier projection."""

from collections.abc import Mapping

import jax
import jax.numpy as jnp
import pytest
from efax import NormalPP, NormalVP
from tjax import JaxRealArray, RngStream

from cem.framework.graph import DisModel, Model
from cem.sip import IntegralWitnessMap, ProportionalWitnessMap, SIPModel
from cem.sip.parameters import WitnessMultiplierParameter
from cem.sip.witness import (
    WitnessLosses,
    integral_witness_loss,
    proportional_witness_loss,
    witness_losses,
    witness_norm_loss,
    witness_residuals,
)

_N = 3
_FEATURES = 4


def _witness(
    witness_map: IntegralWitnessMap,
    mean: JaxRealArray,
    precision: JaxRealArray,
    streams: Mapping[str, RngStream],
) -> JaxRealArray:
    instruments = NormalPP(mean, precision).to_nat()
    return witness_map.infer(instruments, streams=streams, inference=False)


@pytest.fixture
def witness_map(streams: Mapping[str, RngStream]) -> IntegralWitnessMap:
    return IntegralWitnessMap.create(
        _N, instrument_features=_FEATURES, hidden_features=(8,), streams=streams
    )


def test_certain_zero_context_is_neutral(
    witness_map: IntegralWitnessMap, streams: Mapping[str, RngStream]
) -> None:
    witness = _witness(witness_map, jnp.zeros(_FEATURES), jnp.full(_FEATURES, 1e8), streams)
    assert witness.shape == (_N, 2)
    assert jnp.allclose(witness, 0.0, atol=1e-6)


def test_high_precision_does_not_amplify_small_displacements(
    witness_map: IntegralWitnessMap, streams: Mapping[str, RngStream]
) -> None:
    witness = _witness(witness_map, jnp.full(_FEATURES, 1e-4), jnp.full(_FEATURES, 1e8), streams)
    assert jnp.allclose(witness, 0.0, atol=1e-2)


@pytest.mark.parametrize("mean", [-5.0, 5.0])
def test_vanishing_presence_approaches_missing_instrument(
    witness_map: IntegralWitnessMap, streams: Mapping[str, RngStream], mean: float
) -> None:
    missing = _witness(witness_map, jnp.zeros(_FEATURES), jnp.zeros(_FEATURES), streams)
    faint = _witness(witness_map, jnp.full(_FEATURES, mean), jnp.full(_FEATURES, 1e-9), streams)
    assert jnp.allclose(faint, missing, atol=1e-6)


def test_witness_is_bounded_for_extreme_instruments(
    witness_map: IntegralWitnessMap, streams: Mapping[str, RngStream]
) -> None:
    witness = _witness(
        witness_map,
        jnp.asarray([1e6, -1e6, 3.0, 0.0]),
        jnp.asarray([1e8, 1e8, 0.0, 1e-3]),
        streams,
    )
    assert jnp.all(jnp.isfinite(witness))
    assert jnp.all(jnp.abs(witness) <= 1.0)


_RESIDUALS = jnp.asarray([[0.5, -1.0], [2.0, 0.25]])
_WITNESS = jnp.asarray([[0.8, -0.6], [0.3, 0.9]])
_GAIN = jnp.asarray([1.0, 0.5])


@pytest.mark.parametrize("gain", [None, _GAIN])
def test_witness_gradients_follow_the_lagrangian(gain: JaxRealArray | None) -> None:
    multiplier = jnp.asarray(0.7)
    unit_gain = jnp.ones(2) if gain is None else gain
    excess = jnp.sum(unit_gain[:, None] * jnp.square(_WITNESS)) - jnp.mean(unit_gain)

    def lagrangian(witness: JaxRealArray, multiplier: JaxRealArray) -> JaxRealArray:
        return -integral_witness_loss(_RESIDUALS, witness, gain) + witness_norm_loss(
            witness, gain, multiplier
        )

    witness_gradient, multiplier_gradient = jax.grad(lagrangian, argnums=(0, 1))(
        _WITNESS, multiplier
    )
    expected_witness_gradient = unit_gain[:, None] * (2.0 * multiplier * _WITNESS - _RESIDUALS)
    assert jnp.allclose(witness_gradient, expected_witness_gradient)
    # Descending on the negated excess makes the multiplier ascend on the excess.
    assert jnp.isclose(multiplier_gradient, -excess)


def test_integral_witness_loss_weights_products_by_gain() -> None:
    loss = integral_witness_loss(_RESIDUALS, _WITNESS, _GAIN)
    assert jnp.isclose(loss, jnp.sum(_GAIN[:, None] * _RESIDUALS * _WITNESS))


def test_witness_norm_loss_contributes_gradients_only() -> None:
    assert witness_norm_loss(_WITNESS, _GAIN, jnp.asarray(3.0)) == 0.0


@pytest.mark.parametrize("gain", [None, _GAIN])
def test_proportional_witness_regresses_on_the_residuals(gain: JaxRealArray | None) -> None:
    """Ascending the loss moves the witness towards the residuals, weighted by gain."""
    unit_gain = jnp.ones(2) if gain is None else gain
    witness_gradient = jax.grad(proportional_witness_loss, argnums=1)(_RESIDUALS, _WITNESS, gain)
    assert jnp.allclose(witness_gradient, unit_gain[:, None] * (_RESIDUALS - _WITNESS))


def test_proportional_witness_loss_pushes_residuals_by_the_witness() -> None:
    residual_gradient = jax.grad(proportional_witness_loss)(_RESIDUALS, _WITNESS, _GAIN)
    assert jnp.allclose(residual_gradient, _GAIN[:, None] * _WITNESS)


def test_witness_losses_combine_and_route_gradients() -> None:
    """The prediction sees both witnesses fixed; each witness sees the residuals fixed."""
    proportional = 0.5 * _WITNESS
    damping_weight = jnp.asarray(2.0)

    def losses(
        residuals: JaxRealArray, integral: JaxRealArray, proportional: JaxRealArray
    ) -> WitnessLosses:
        return witness_losses(
            residuals,
            integral_witness=integral,
            proportional_witness=proportional,
            gain=_GAIN,
            multiplier=jnp.asarray(0.0),
            damping_weight=damping_weight,
        )

    result = losses(_RESIDUALS, _WITNESS, proportional)
    assert jnp.isclose(
        result.confounding_cost,
        integral_witness_loss(_RESIDUALS, _WITNESS, _GAIN)
        + damping_weight * proportional_witness_loss(_RESIDUALS, proportional, _GAIN),
    )
    cost_gradients = jax.grad(lambda *args: losses(*args).confounding_cost, argnums=(0, 1, 2))(
        _RESIDUALS, _WITNESS, proportional
    )
    assert jnp.allclose(
        cost_gradients[0], _GAIN[:, None] * (_WITNESS + damping_weight * proportional)
    )
    assert jnp.allclose(cost_gradients[1], 0.0)
    assert jnp.allclose(cost_gradients[2], 0.0)
    for name in ("integral_witness_loss", "proportional_witness_loss"):
        residual_gradient = jax.grad(
            lambda r, n=name: getattr(losses(r, _WITNESS, proportional), n)
        )(_RESIDUALS)
        assert jnp.allclose(residual_gradient, 0.0)


def test_proportional_witness_is_neutral_and_unbounded(
    streams: Mapping[str, RngStream],
) -> None:
    proportional_map = ProportionalWitnessMap.create(
        _N, instrument_features=_FEATURES, hidden_features=(), streams=streams
    )

    def proportional(mean: JaxRealArray) -> JaxRealArray:
        instruments = NormalPP(mean, jnp.full(_FEATURES, 1e8)).to_nat()
        return proportional_map.infer(instruments, streams=streams, inference=False)

    assert jnp.allclose(proportional(jnp.zeros(_FEATURES)), 0.0, atol=1e-6)
    small = proportional(jnp.full(_FEATURES, 1.0))
    large = proportional(jnp.full(_FEATURES, 100.0))
    # Without a final activation, a linear map scales with the displacement.
    assert jnp.allclose(large, 100.0 * small, rtol=1e-4)


class _MultiplierModel(Model):
    multiplier: WitnessMultiplierParameter


@pytest.mark.parametrize(("gradient", "expect_positive"), [(1.0, False), (-1.0, True)])
def test_sip_routing_projects_the_multiplier_onto_nonnegative_values(
    gradient: float, *, expect_positive: bool
) -> None:
    transform = SIPModel.gradient_transformations(0.1)
    parameter_types = tuple(transform.learnable_parameter_types())

    def partition(value: float) -> DisModel:
        model = _MultiplierModel(WitnessMultiplierParameter(jnp.asarray(value)))
        return DisModel.create(model, parameter_types)

    model = partition(0.0)
    delta, _ = transform.update(partition(gradient), transform.init(model), model)
    update = delta.assembled()
    assert isinstance(update, _MultiplierModel)
    multiplier = update.multiplier.value
    assert multiplier > 0 if expect_positive else multiplier == 0


def test_witness_holds_instruments_fixed(
    witness_map: IntegralWitnessMap, streams: Mapping[str, RngStream]
) -> None:
    """Witness learning leaves instrument fitting to the instrument loss."""
    mean = jnp.asarray([0.5, -0.3, 0.2, 0.1])
    precision = jnp.asarray([1.0, 2.0, 0.5, 4.0])
    gradients = jax.grad(
        lambda m, p: jnp.sum(_witness(witness_map, m, p, streams)), argnums=(0, 1)
    )(mean, precision)
    assert all(jnp.allclose(gradient, 0.0) for gradient in gradients)


def test_witness_residuals_vanish_for_a_calibrated_variance() -> None:
    """A calibrated variance leaves only the mean residual."""
    target = NormalVP(jnp.zeros(2), jnp.ones(2))
    prediction = NormalVP(jnp.asarray([0.5, -1.0]), jnp.asarray([1.25, 2.0]))
    assert jnp.allclose(
        witness_residuals(target, prediction), jnp.asarray([[0.5, 0.0], [-1.0, 0.0]])
    )
