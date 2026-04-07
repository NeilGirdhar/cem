from __future__ import annotations

from collections.abc import Mapping
from typing import Self

import equinox as eqx
import jax
import jax.numpy as jnp
from jax.nn.initializers import normal
from tjax import JaxArray, RngStream

from cem.structure.graph import LearnableParameter

_logit_init = normal(1.0)


class RivalryGroups(eqx.Module):
    """Learned group-membership matrix for rivalry normalization.

    W[i, j] = exp(U[i, j]) / sum_{i'} exp(U[i', j])  (columnwise softmax)

    Each column j defines a probability distribution over rivalry groups for feature j.

    Attributes:
        logits: Unconstrained logits U, shape (num_groups, num_features).
    """

    logits: LearnableParameter[JaxArray]

    @classmethod
    def create(
        cls,
        num_features: int,
        num_groups: int,
        *,
        streams: Mapping[str, RngStream],
    ) -> Self:
        stream = streams["parameters"]
        return cls(
            logits=LearnableParameter(
                _logit_init(stream.key(), (num_groups, num_features), jnp.float64)
            )
        )

    @property
    def weights(self) -> JaxArray:
        """Columnwise softmax of logits, shape (num_groups, num_features)."""
        return jax.nn.softmax(self.logits.value, axis=0)


class RivalryNorm(eqx.Module):
    """Rivalry normalization: removes the gauge symmetry of global presence scaling.

    Subtracts from each feature's log-presence the log total presence of the rivalry groups it
    belongs to:

        logpresences -= W^T log(W presences)

    This leaves all pairwise log-presence differences within each group intact while adjusting
    cross-group comparisons, inducing competition between groups.  Phase is unchanged.

    Attributes:
        groups: Rivalry group membership matrix.
    """

    groups: RivalryGroups

    @classmethod
    def create(
        cls,
        num_features: int,
        num_groups: int,
        *,
        streams: Mapping[str, RngStream],
    ) -> Self:
        return cls(groups=RivalryGroups.create(num_features, num_groups, streams=streams))

    def infer(self, z: JaxArray) -> JaxArray:
        """Apply rivalry normalization to phasors.

        Args:
            z: Input phasors, shape (..., m).

        Returns:
            Normalized phasors with the same shape and phases as z.
        """
        w = self.groups.weights  # (g, m)
        presences = jnp.abs(z)  # (..., m)

        # group totals: W presences, shape (..., g)
        group_totals = presences @ w.T  # (..., g)

        # log correction: W^T log(W presences), shape (..., m)
        log_correction = jnp.log(jnp.maximum(group_totals, 1e-8)) @ w  # (..., m)

        # new presences: exp(logpresences - W^T log(W presences))
        log_presences = jnp.log(jnp.maximum(presences, 1e-8))
        new_presences = jnp.exp(log_presences - log_correction)

        # rescale z by new_presences / presences to preserve phase
        scale = jnp.where(presences > 0, new_presences / presences, jnp.zeros_like(presences))
        return z * scale
