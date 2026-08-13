from collections.abc import Mapping

import jax.numpy as jnp
from tjax import RngStream

from cem.phasor.evidence_pooling import EvidencePooling, EvidencePoolingWithDropout


def test_evidence_pooling_initialization(streams: Mapping[str, RngStream]) -> None:
    f = EvidencePooling.create(3, 5, streams=streams)

    assert f.weight.value.shape == (5, 3)
    assert f.weight.value.dtype == jnp.complex128


def test_evidence_pooling_project_shape(streams: Mapping[str, RngStream]) -> None:
    f = EvidencePooling.create(3, 5, streams=streams)
    x = jnp.ones((7, 3), dtype=jnp.complex128)

    assert f.project(x).shape == (7, 5)


def test_evidence_pooling_absent_input_gives_absent_output(
    streams: Mapping[str, RngStream],
) -> None:
    f = EvidencePooling.create(3, 5, streams=streams)
    x = jnp.zeros(3, dtype=jnp.complex128)

    assert jnp.allclose(f.project(x), jnp.zeros(5, dtype=jnp.complex128))


def test_evidence_pooling_with_dropout_zero_rate_matches_evidence_pooling(
    streams: Mapping[str, RngStream],
) -> None:
    f = EvidencePoolingWithDropout.create(3, 5, dropout_rate=0.0, streams=streams)
    x = jnp.ones(3, dtype=jnp.complex128)

    assert jnp.allclose(
        f.infer(x, streams=streams, inference=False),
        EvidencePooling.project(f, x),
    )
