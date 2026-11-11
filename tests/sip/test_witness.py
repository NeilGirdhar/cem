"""Check the confounding witness, its training loss, and the multiplier projection."""

from collections.abc import Mapping

import jax
import jax.numpy as jnp
import pytest
from efax import NormalPP, NormalVP
from jax.experimental import checkify
from tjax import JaxRealArray, RngStream

from cem.sip import SIPModel, WitnessMap
from cem.sip.parameters import WitnessMultiplierParameter
from cem.sip.witness import witness_loss, witness_norm_penalty, witness_residuals
from cem.structure.graph import DisModel, Model
from cem.structure.graph.losses import LossGradients

_N = 3
_FEATURES = 4


def _witness(
    witness_map: WitnessMap,
    mean: JaxRealArray,
    precision: JaxRealArray,
    streams: Mapping[str, RngStream],
) -> JaxRealArray:
    instruments = NormalPP(mean, precision).to_nat()
    return witness_map.infer(instruments, streams=streams, inference=False)


@pytest.fixture
def witness_map(streams: Mapping[str, RngStream]) -> WitnessMap:
    return WitnessMap.create(
        _N, instrument_features=_FEATURES, hidden_features=(8,), streams=streams
    )


def test_certain_zero_context_is_neutral(
    witness_map: WitnessMap, streams: Mapping[str, RngStream]
) -> None:
    witness = _witness(witness_map, jnp.zeros(_FEATURES), jnp.full(_FEATURES, 1e8), streams)
    assert witness.shape == (_N, 2)
    assert jnp.allclose(witness, 0.0, atol=1e-6)


def test_high_precision_does_not_amplify_small_displacements(
    witness_map: WitnessMap, streams: Mapping[str, RngStream]
) -> None:
    witness = _witness(witness_map, jnp.full(_FEATURES, 1e-4), jnp.full(_FEATURES, 1e8), streams)
    assert jnp.allclose(witness, 0.0, atol=1e-2)


@pytest.mark.parametrize("mean", [-5.0, 5.0])
def test_vanishing_presence_approaches_missing_instrument(
    witness_map: WitnessMap, streams: Mapping[str, RngStream], mean: float
) -> None:
    missing = _witness(witness_map, jnp.zeros(_FEATURES), jnp.zeros(_FEATURES), streams)
    faint = _witness(witness_map, jnp.full(_FEATURES, mean), jnp.full(_FEATURES, 1e-9), streams)
    assert jnp.allclose(faint, missing, atol=1e-6)


def test_witness_is_bounded_for_extreme_instruments(
    witness_map: WitnessMap, streams: Mapping[str, RngStream]
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
        return witness_loss(_RESIDUALS, witness, gain) + witness_norm_penalty(
            witness, gain, multiplier
        )

    witness_gradient, multiplier_gradient = jax.grad(lagrangian, argnums=(0, 1))(
        _WITNESS, multiplier
    )
    expected_witness_gradient = unit_gain[:, None] * (2.0 * multiplier * _WITNESS - _RESIDUALS)
    assert jnp.allclose(witness_gradient, expected_witness_gradient)
    # Descending on the negated excess makes the multiplier ascend on the excess.
    assert jnp.isclose(multiplier_gradient, -excess)


def test_witness_loss_is_the_negated_confounding_cost() -> None:
    loss = witness_loss(_RESIDUALS, _WITNESS, _GAIN)
    assert jnp.isclose(loss, -jnp.sum(_GAIN[:, None] * _RESIDUALS * _WITNESS))


def test_witness_norm_penalty_contributes_gradients_only() -> None:
    assert witness_norm_penalty(_WITNESS, _GAIN, jnp.asarray(3.0)) == 0.0


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
    gradients = LossGradients(
        {name: partition(0.0) for name in transform.requested_loss_ids()}, partition(gradient)
    )
    checked = checkify.checkify(transform.update, errors=checkify.user_checks)
    error, (delta, _) = checked(gradients, transform.init(model), model)
    error.throw()
    update = delta.assembled()
    assert isinstance(update, _MultiplierModel)
    multiplier = update.multiplier.value
    assert multiplier > 0 if expect_positive else multiplier == 0


def test_witness_holds_instruments_fixed(
    witness_map: WitnessMap, streams: Mapping[str, RngStream]
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
