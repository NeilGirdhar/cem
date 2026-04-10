from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
from tjax import JaxArray, frozendict


class NodeConfiguration(eqx.Module):
    """The output produced by a node during one inference step."""

    def total_loss(self) -> JaxArray:
        """Return this node's contribution to the scalar model loss."""
        return jnp.zeros(())


class TargetConfiguration(NodeConfiguration):
    """NodeConfiguration mixin for nodes that compute a per-field loss."""

    loss: frozendict[str, JaxArray]

    def total_loss(self) -> JaxArray:
        return sum((jnp.sum(v) for v in self.loss.values()), start=jnp.asarray(0.0))
