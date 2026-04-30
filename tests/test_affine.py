from __future__ import annotations

from collections.abc import Mapping

import jax.numpy as jnp
from tjax import RngStream

from cem.transforms import Affine, AffineWithDropout


def test_affine_real_initialization(streams: Mapping[str, RngStream]) -> None:
    f = Affine.create(3, 5, complex_matrix=False, streams=streams)

    assert f.weight.value.shape == (5, 3)
    assert f.weight.value.dtype == jnp.float64
    assert f.bias.value.shape == (5,)
    assert f.bias.value.dtype == jnp.float64


def test_affine_complex_initialization(streams: Mapping[str, RngStream]) -> None:
    f = Affine.create(3, 5, complex_matrix=True, streams=streams)

    assert f.weight.value.shape == (5, 3)
    assert f.weight.value.dtype == jnp.complex128
    assert f.bias.value.shape == (5,)
    assert f.bias.value.dtype == jnp.complex128


def test_affine_complex_infer_shape(streams: Mapping[str, RngStream]) -> None:
    f = Affine.create(3, 5, complex_matrix=True, streams=streams)
    x = jnp.ones((7, 3), dtype=jnp.complex128)

    assert f.infer(x).shape == (7, 5)


def test_affine_with_dropout_zero_rate_matches_affine(
    streams: Mapping[str, RngStream],
) -> None:
    f = AffineWithDropout.create(3, 5, dropout_rate=0.0, streams=streams)
    x = jnp.ones(3, dtype=jnp.float64)

    assert jnp.allclose(
        f.infer(x, streams=streams, inference=False),
        Affine.infer(f, x),
    )
