from collections.abc import Mapping

import jax.numpy as jnp
from tjax import RngStream

from cem.transforms import Affine


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


def test_affine_complex_project_shape(streams: Mapping[str, RngStream]) -> None:
    f = Affine.create(3, 5, complex_matrix=True, streams=streams)
    x = jnp.ones((7, 3), dtype=jnp.complex128)

    assert f.project(x).shape == (7, 5)
