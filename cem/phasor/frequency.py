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


def geometric_frequencies(num_features: int, base: float = 1.0) -> JaxArray:
    """Generate geometrically spaced angular frequencies f_j = base · 2^j (rad/unit).

    **Discriminatory power.**  The phasor encoding of Normal(μ, σ²) at angular frequency f has

        phase = f · μ  (increases with f — higher frequencies separate means more),
        amplitude = exp(−f² · σ² / 2)  (decreases with f — higher frequencies are noisier).

    Discriminatory power between means separated by Δμ is f · Δμ · exp(−f² · σ² / 2).
    Since Δμ is a positive scalar it cancels in the argmax, giving f* = 1/σ rad/unit.  For
    unit-variance standardised data (σ = 1) this peak is at f* = 1 rad/unit.  Frequencies well
    below f* contribute little phase; frequencies well above f* have negligible amplitude.

    The grid descends geometrically from ``base`` so that the highest frequency is always
    anchored at the discrimination peak: set ``base = f* = 1/σ``.  Lower frequencies resolve
    phase ambiguity: at frequency f, means separated by Δμ = 2π/f produce identical phasors,
    so each halving of f doubles the unambiguous range.

    Args:
        num_features: Number of frequency components m.
        base: Highest angular frequency in rad/unit, equal to the discrimination peak f* = 1/σ.
            Default 1.0 is correct for unit-variance standardised data.

    Returns:
        Float array of shape (m,) with f_j = base / 2^j rad/unit, for j = 0, ..., m-1.
    """
    return base / jnp.pow(2.0, jnp.arange(num_features, dtype=jnp.float64))
