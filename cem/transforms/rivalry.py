from __future__ import annotations

from collections.abc import Mapping
from dataclasses import InitVar
from typing import override

import equinox as eqx
import jax
import jax.numpy as jnp
from jax.nn.initializers import normal
from tjax import JaxArray, RngStream

from cem.structure import LearnableParameter, Module

_logit_init = normal(1.0)


class RivalryGroups(Module):
    """Learned group-membership matrix for rivalry normalization.

    W[i, j] = exp(U[i, j]) / sum_{i'} exp(U[i', j])  (columnwise softmax)

    Each column j defines a probability distribution over rivalry groups for feature j.

    Attributes:
        logits: Unconstrained logits U, shape (num_groups, num_features).
    """

    num_features: InitVar[int]
    num_groups: InitVar[int]
    logits: LearnableParameter[JaxArray] = eqx.field(init=False)

    @override
    def __post_init__(
        self,
        streams: Mapping[str, RngStream],
        num_features: int,
        num_groups: int,
    ) -> None:
        super().__post_init__(streams=streams)
        stream = streams["parameters"]
        self.logits = LearnableParameter(
            _logit_init(stream.key(), (num_groups, num_features), jnp.float64)
        )

    @property
    def weights(self) -> JaxArray:
        """Columnwise softmax of logits, shape (num_groups, num_features)."""
        return jax.nn.softmax(self.logits.value, axis=0)


class RivalryNorm(Module):
    """Rivalry normalization: removes the gauge symmetry of global presence scaling.

    Subtracts from each feature's log-presence the log total presence of the rivalry groups it
    belongs to:

        logpresences -= W^T log(W presences)

    This leaves all pairwise log-presence differences within each group intact while adjusting
    cross-group comparisons, inducing competition between groups.  Phase is unchanged.

    Attributes:
        groups: Rivalry group membership matrix.
    """

    num_features: InitVar[int]
    num_groups: InitVar[int]
    groups: RivalryGroups = eqx.field(init=False)

    @override
    def __post_init__(
        self,
        streams: Mapping[str, RngStream],
        num_features: int,
        num_groups: int,
    ) -> None:
        super().__post_init__(streams=streams)
        self.groups = RivalryGroups(num_features, num_groups, streams=streams)

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
