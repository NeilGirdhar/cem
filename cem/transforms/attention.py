from __future__ import annotations

import jax
import jax.numpy as jnp
from tjax import JaxArray


def select(keys: JaxArray, query: JaxArray) -> JaxArray:
    """Compute softmax selection weights from key-query phasor concordance.

    s = Re(K conj(q)),  p = softmax(s)

    Each score s_i measures how well the i-th key row aligns with the query via constructive
    and destructive interference across channels.  The softmax induces competition among
    candidates, producing a probability distribution over them.

    Args:
        keys: Key phasors, shape (m, n).
        query: Query phasor, shape (n,).

    Returns:
        Selection weights, shape (m,).
    """
    scores = jnp.real(keys @ jnp.conj(query))  # (m,)
    return jax.nn.softmax(scores)


def interpolate(weights: JaxArray, content: JaxArray) -> JaxArray:
    """Compute a convex combination of content rows using selection weights.

    y = p^T V

    Args:
        weights: Selection weights (e.g. from select()), shape (m,).
        content: Content phasors, shape (m, d).

    Returns:
        Weighted combination, shape (d,).
    """
    return weights @ content
