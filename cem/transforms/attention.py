from __future__ import annotations

import jax
import jax.numpy as jnp
from tjax import JaxArray


def select(keys: JaxArray, query: JaxArray) -> JaxArray:
    """Compute per-head softmax selection weights from key-query phasor concordance.

    s^(i) = Re(K^(i) conj(q^(i))),  p^(i) = softmax(s^(i))

    Each head operates independently: scores measure concordance between each candidate key and
    the head's query, and softmax induces competition among candidates within each head.

    Args:
        keys: Key phasors, shape (*batch, h, m, n) — h heads, m candidates, n alignment features.
        query: Query phasors, shape (*batch, h, n) — one query vector per head.

    Returns:
        Selection weights, shape (*batch, h, m) — a probability distribution over candidates per
        head.
    """
    scores = jnp.real(jnp.einsum("...hmn,...hn->...hm", keys, jnp.conj(query)))
    return jax.nn.softmax(scores, axis=-1)


def interpolate(weights: JaxArray, content: JaxArray) -> JaxArray:
    """Compute a convex combination of content rows per head and concatenate across heads.

    y^(i) = p^(i)^T V^(i),  y = (y^(1), ..., y^(h))

    Args:
        weights: Selection weights (e.g. from select()), shape (*batch, h, m).
        content: Content phasors, shape (*batch, h, m, d) — one content matrix per head.

    Returns:
        Concatenated per-head outputs, shape (*batch, h * d).
    """
    per_head = jnp.einsum("...hm,...hmd->...hd", weights, content)
    return per_head.reshape((*per_head.shape[:-2], -1))
