"""Shared numerical settings and reproducible random streams for tests."""

import logging
from collections.abc import Generator, Mapping

import jax
import jax.random as jr
import numpy as np
import pytest
from jax import enable_x64
from tjax import RngStream, create_streams


@pytest.fixture(autouse=True)  # ruff: ignore[pytest-fixture-autouse]
def _jax_fixture() -> Generator[None]:
    """Use double precision and reject implicit rank promotion in JAX operations."""
    with jax.numpy_rank_promotion("raise"), enable_x64():
        yield


@pytest.fixture
def log() -> None:
    logging.disable()


@pytest.fixture
def np_rng() -> np.random.Generator:
    return np.random.default_rng(123)


@pytest.fixture
def streams() -> Mapping[str, RngStream]:
    """Provide separate reproducible streams for parameter initialization and inference."""
    return create_streams({"parameters": jr.key(123), "inference": jr.key(456)})
