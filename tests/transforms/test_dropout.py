from collections.abc import Mapping

import jax.numpy as jnp
from tjax import RngStream

from cem.transforms import dropout

# ── dropout ───────────────────────────────────────────────────────────────────


def test_dropout_zero_rate_is_identity(streams: Mapping[str, RngStream]) -> None:
    p = jnp.array([1 + 1j, 2 - 1j, 0.5 + 0.5j])
    assert jnp.allclose(dropout(p, streams["inference"].key(), 0.0), p)


def test_dropout_preserves_expected_value(streams: Mapping[str, RngStream]) -> None:
    p = jnp.array([1 + 0j, 0 + 2j, -1 + 1j])
    stream = streams["inference"]
    samples = jnp.stack([dropout(p, stream.key(), 0.3) for _ in range(2000)])
    assert jnp.allclose(jnp.mean(samples, axis=0), p, atol=0.1)
