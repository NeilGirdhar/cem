"""Auto-balanced multi-loss combiner driven by EMAs of per-component magnitudes."""

from __future__ import annotations

from typing import Self

import equinox as eqx
import jax.numpy as jnp
from jax.lax import stop_gradient
from tjax import JaxRealArray

from cem.structure.graph.parameters import TrackingParameter


class LossBalancer(eqx.Module):
    """Auto-balance an ``N``-component loss by weighting each term by its own EMA.

    The combined objective is::

        Σ_i sg(v_i) · L_i  +  ½ Σ_i (v_i − sg(|L_i|))²

    The first term scales each loss by its current EMA estimate ``v_i``; the
    ``stop_gradient`` ensures the combination weights themselves do not receive
    gradient from the weighted sum, only from the auxiliary term.

    The second term is the EMA driver: when ``v`` is updated by SGD with learning
    rate ``η``, the gradient ``(v_i − |L_i|)`` produces the update
    ``v_i ← (1−η)·v_i + η·|L_i|``.  That is exactly an exponential moving average
    of ``|L_i|`` with decay ``1 − η``.

    The result is that loss components with persistently large magnitudes get
    proportionally more gradient pressure on shared parameters; once a component
    shrinks, its weight drops and gradient redistributes to whichever components
    remain unsatisfied.

    The ``v`` parameter is wrapped in :class:`TrackingParameter` so it is updated
    by a separate, slower optimizer than the model's :class:`LearnableParameter`
    weights.
    """

    v: TrackingParameter[JaxRealArray]

    @classmethod
    def create(cls, n_components: int) -> Self:
        return cls(v=TrackingParameter(jnp.ones(n_components)))

    def total_with_meta_aux(self, losses: JaxRealArray) -> JaxRealArray:
        """Combine ``losses`` of shape ``(N,)`` into a single scalar objective."""
        v = self.v.value
        weighted = jnp.sum(stop_gradient(v) * losses)
        meta_aux = 0.5 * jnp.sum((v - stop_gradient(jnp.abs(losses))) ** 2)
        return weighted + meta_aux
