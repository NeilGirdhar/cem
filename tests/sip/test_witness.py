"""Check the confounding witness at the instrument contexts that the thesis constrains."""

from collections.abc import Mapping

import jax.numpy as jnp
import pytest
from tjax import JaxRealArray, RngStream

from cem.npn.gaussian import normal_from_mean_and_presence
from cem.sip import WitnessMap

_N = 3
_FEATURES = 4


def _witness(
    witness_map: WitnessMap,
    mean: JaxRealArray,
    precision: JaxRealArray,
    streams: Mapping[str, RngStream],
) -> JaxRealArray:
    instruments = normal_from_mean_and_presence(mean, precision)
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
