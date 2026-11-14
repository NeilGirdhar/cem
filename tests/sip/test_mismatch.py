from collections.abc import Mapping
from typing import TypedDict

import equinox as eqx
import jax
import jax.numpy as jnp
import pytest
from efax import NormalNP, NormalPP
from jax import tree
from tjax import JaxRealArray, RngStream

from cem.sip import InstrumentedMessage
from cem.sip.mismatch import Mismatch

from .tools import MISMATCH_SETTINGS, _i_message, _message


class _StepInputs(TypedDict):
    raw_observation: NormalNP
    local_variance: JaxRealArray
    predictor: InstrumentedMessage
    gain: InstrumentedMessage | None


def _mismatch(n: int, streams: Mapping[str, RngStream]) -> Mismatch:
    return Mismatch.create(
        n,
        predictor_features=2,
        gain_features=1,
        mismatch_settings=MISMATCH_SETTINGS,
        streams=streams,
    )


def _inputs(n: int) -> _StepInputs:
    return _StepInputs(
        raw_observation=_message(jnp.zeros(n), jnp.ones(n)),
        local_variance=jnp.zeros(n),
        predictor=_i_message(
            NormalPP(jnp.asarray([0.3, -0.2]), jnp.ones(2)),
            NormalPP(jnp.asarray([0.5, 0.1]), jnp.ones(2)),
        ),
        gain=_i_message(
            NormalPP(jnp.ones(n), jnp.ones(n)), NormalPP(jnp.asarray([0.4]), jnp.ones(1))
        ),
    )


def test_mismatch_message_has_unit_presence(streams: Mapping[str, RngStream]) -> None:
    n = 3
    mismatch = _mismatch(n, streams)
    outputs = mismatch.step(**_inputs(n), streams=streams, inference=True)
    assert outputs.mismatch.negative_half_precision.shape == (2 * n,)
    assert jnp.allclose(outputs.mismatch.negative_half_precision, -0.5 * jnp.ones(2 * n))


def test_mismatch_loss_sums_prediction_and_witness_losses(
    streams: Mapping[str, RngStream],
) -> None:
    n = 2
    mismatch = _mismatch(n, streams)
    outputs = mismatch.step(**_inputs(n), streams=streams, inference=True)
    assert jnp.allclose(
        outputs.loss,
        outputs.reconstruction_cost
        + outputs.confounding_cost
        - outputs.integral_witness_loss
        - outputs.proportional_witness_loss
        + outputs.witness_norm_loss,
    )


def test_mismatch_holds_target_observation_fixed(streams: Mapping[str, RngStream]) -> None:
    """No mismatch loss or mismatch component sends gradient into the raw observation or noise."""
    n = 2
    mismatch = _mismatch(n, streams)
    inputs = _inputs(n)

    def loss_fn(raw_observation: NormalNP, local_variance: JaxRealArray) -> JaxRealArray:
        target_inputs = _StepInputs(
            raw_observation=raw_observation,
            local_variance=local_variance,
            predictor=inputs["predictor"],
            gain=inputs["gain"],
        )
        outputs = mismatch.step(**target_inputs, streams=streams, inference=True)
        return outputs.loss + jnp.sum(outputs.mismatch.mean_times_precision)

    gradients = jax.grad(loss_fn, argnums=(0, 1))(
        inputs["raw_observation"], inputs["local_variance"] + 0.5
    )
    assert all(jnp.allclose(leaf, 0.0) for leaf in tree.leaves(gradients))


def test_reconstruction_cost_trains_prediction_map(streams: Mapping[str, RngStream]) -> None:
    """The prediction map's last-layer weight has a nonzero gradient from reconstruction cost."""
    n = 2
    mismatch = _mismatch(n, streams)
    inputs = _inputs(n)

    def loss_fn(m: Mismatch) -> jnp.ndarray:
        return m.step(**inputs, streams=streams, inference=True).reconstruction_cost

    grads = eqx.filter_grad(loss_fn)(mismatch)
    assert jnp.any(jnp.abs(grads.prediction_map.layers[-1].weight.value) > 0)


_WITNESS_MAPS = ("integral_witness_map", "proportional_witness_map")
_WITNESS_LOSSES = ("integral_witness_loss", "proportional_witness_loss")


def _last_weight(m: Mismatch, name: str) -> JaxRealArray:
    return getattr(m, name).perceptron.layers[-1].weight.value


@pytest.mark.parametrize("witness_map", _WITNESS_MAPS)
def test_confounding_cost_does_not_train_witness_maps(
    streams: Mapping[str, RngStream], witness_map: str
) -> None:
    """Confounding cost holds both witnesses fixed."""
    n = 2
    mismatch = _mismatch(n, streams)
    inputs = _inputs(n)

    def loss_fn(m: Mismatch) -> jnp.ndarray:
        return m.step(**inputs, streams=streams, inference=True).confounding_cost

    grads = eqx.filter_grad(loss_fn)(mismatch)
    assert jnp.allclose(_last_weight(grads, witness_map), 0.0)


@pytest.mark.parametrize("witness_loss", _WITNESS_LOSSES)
def test_witness_losses_do_not_train_prediction_map(
    streams: Mapping[str, RngStream], witness_loss: str
) -> None:
    """Each witness loss holds the residuals fixed."""
    n = 2
    mismatch = _mismatch(n, streams)
    inputs = _inputs(n)

    def loss_fn(m: Mismatch) -> jnp.ndarray:
        return getattr(m.step(**inputs, streams=streams, inference=True), witness_loss)

    grads = eqx.filter_grad(loss_fn)(mismatch)
    assert jnp.allclose(grads.prediction_map.layers[-1].weight.value, 0.0)


@pytest.mark.parametrize(
    ("witness_loss", "witness_map"), zip(_WITNESS_LOSSES, _WITNESS_MAPS, strict=True)
)
def test_each_witness_loss_trains_only_its_witness_map(
    streams: Mapping[str, RngStream], witness_loss: str, witness_map: str
) -> None:
    n = 2
    mismatch = _mismatch(n, streams)
    inputs = _inputs(n)
    # Give the proportional witness a nonzero output so both losses have witness gradients.
    mismatch = eqx.tree_at(
        lambda m: m.proportional_witness_map.perceptron.layers[-1].bias.value,
        mismatch,
        jnp.full(2 * n, 0.3),
    )

    def loss_fn(m: Mismatch) -> jnp.ndarray:
        return getattr(m.step(**inputs, streams=streams, inference=True), witness_loss)

    grads = eqx.filter_grad(loss_fn)(mismatch)
    (other_map,) = (name for name in _WITNESS_MAPS if name != witness_map)
    assert jnp.any(jnp.abs(_last_weight(grads, witness_map)) > 0)
    assert jnp.allclose(_last_weight(grads, other_map), 0.0)


def _recruiter_mismatch(n: int, streams: Mapping[str, RngStream]) -> Mismatch:
    return Mismatch.create(
        n,
        predictor_features=2,
        gain_features=0,
        mismatch_settings=MISMATCH_SETTINGS,
        streams=streams,
    )


def _recruiter_inputs(n: int) -> _StepInputs:
    return _StepInputs(
        raw_observation=_message(jnp.zeros(n), jnp.ones(n)),
        local_variance=jnp.zeros(n),
        predictor=_i_message(
            NormalPP(jnp.asarray([0.3, -0.2]), jnp.ones(2)),
            NormalPP(jnp.asarray([0.5, 0.1]), jnp.ones(2)),
        ),
        gain=None,
    )


def test_mismatch_accepts_none_gain(streams: Mapping[str, RngStream]) -> None:
    """A recruiter-style Mismatch (gain_features=0) runs with gain=None."""
    n = 2
    mismatch = _recruiter_mismatch(n, streams)
    outputs = mismatch.step(**_recruiter_inputs(n), streams=streams, inference=True)
    assert jnp.isfinite(outputs.loss)
    assert outputs.mismatch.mean_times_precision.shape == (2 * n,)


def test_mismatch_holds_predictor_and_gain_fixed(streams: Mapping[str, RngStream]) -> None:
    """No mismatch loss sends gradient into the parents' observations, instruments, or gain."""
    n = 2
    mismatch = _mismatch(n, streams)
    inputs = _inputs(n)

    def loss_fn(predictor: InstrumentedMessage, gain: InstrumentedMessage) -> JaxRealArray:
        parent_inputs = _StepInputs(
            raw_observation=inputs["raw_observation"],
            local_variance=inputs["local_variance"],
            predictor=predictor,
            gain=gain,
        )
        return mismatch.step(**parent_inputs, streams=streams, inference=True).loss

    gradients = jax.grad(loss_fn, argnums=(0, 1))(inputs["predictor"], inputs["gain"])
    assert all(jnp.allclose(leaf, 0.0) for leaf in tree.leaves(gradients))


def test_mismatch_message_carries_mean_and_deviation_errors(
    streams: Mapping[str, RngStream],
) -> None:
    """Each channel carries the mean error and predicted deviation minus actual deviation."""
    n = 2
    mismatch = _mismatch(n, streams)
    inputs = _inputs(n)
    outputs = mismatch.step(**inputs, streams=streams, inference=True)
    prediction = mismatch.prediction_map.infer(
        inputs["predictor"].observation
    ).to_variance_parametrization()
    target = inputs["raw_observation"].to_variance_parametrization()
    assert inputs["gain"] is not None
    gain = inputs["gain"].observation.mean()
    mean_error = prediction.mean - target.mean
    deviation_error = jnp.sqrt(prediction.variance) - jnp.sqrt(
        target.variance + jnp.square(mean_error)
    )
    expected = gain[:, None] * jnp.stack([mean_error, deviation_error], axis=-1)
    assert jnp.allclose(outputs.mismatch.mean().reshape(n, 2), expected)
