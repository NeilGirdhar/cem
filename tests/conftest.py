import logging
from collections.abc import Generator, Mapping

import jax
import jax.random as jr
import networkx as nx
import numpy as np
import pytest
from jax import enable_x64
from tjax import RngStream, create_streams, register_graph_as_jax_pytree


@pytest.fixture(autouse=True)
def _jax_fixture() -> Generator[None]:
    # jax.debug_key_reuse(True) is too slow.
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
    return create_streams({"parameters": jr.key(123), "inference": jr.key(456)})


@pytest.fixture(autouse=True, scope="session")
def fixture_register_graph_as_jax_pytree() -> None:
    register_graph_as_jax_pytree(nx.DiGraph)
