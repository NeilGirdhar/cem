from __future__ import annotations

from typing import Any

import jax.numpy as jnp
from efax import Flattener, NaturalParametrization
from tjax import JaxArray, JaxRealArray


def make_frequency_grid[NP: NaturalParametrization[Any, Any]](
    flattener: Flattener[NP],
    frequencies: JaxRealArray,
) -> NP:
    """Build the frequency grid ``t`` used by ``PhasorMessage.to_distribution``.

    Constructs a NaturalParametrization of shape ``(m * d,)`` where element ``j * d + k``
    equals ``frequencies[j] * e_k`` — the k-th standard basis vector scaled by the j-th
    frequency.  This is the ``t`` argument expected by
    :meth:`~cem.phasor.message.PhasorMessage.to_distribution`.

    Args:
        flattener: Flattener for the distribution family, with ``mapped_to_plane=False``.
            Defines the natural-parameter dimension d and the pytree structure of t.
        frequencies: Geometric frequency grid, shape ``(m,)``.

    Returns:
        NaturalParametrization of shape ``(m * d,)``.
    """
    assert not flattener.mapped_to_plane
    assert frequencies.ndim == 1
    d = flattener.final_dimension_size()
    m = frequencies.shape[0]
    eye = jnp.eye(d, dtype=frequencies.dtype)  # (d, d)
    t_flat = (frequencies[:, None, None] * eye[None, :, :]).reshape(m * d, d)  # (m*d, d)
    return flattener.unflatten(t_flat)


def geometric_frequencies(num_features: int, base: float = 2.0 * jnp.pi) -> JaxArray:
    """Generate geometrically spaced frequencies k_j = base * 2^j for scalar encoding.

    Args:
        num_features: Number of frequency components m.
        base: Base frequency. Default is 2*pi as in the thesis.

    Returns:
        Float array of shape (m,) with k_j = base * 2^j for j = 0, ..., m-1.
    """
    return base * jnp.pow(2.0, jnp.arange(num_features, dtype=jnp.float64))
