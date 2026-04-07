from __future__ import annotations

from typing import Any, Self

import equinox as eqx
import jax.numpy as jnp
from efax import Flattener, HasEntropyEP, NaturalParametrization
from tjax import JaxArray, JaxRealArray

from cem.structure.graph.node import NodeConfiguration


class PerceptronTargetConfiguration[EP: HasEntropyEP](NodeConfiguration):
    """Output of a PerceptronTargetNode: loss and recovered predicted distribution.

    Attributes:
        loss: Per-element cross-entropy loss between observed and predicted distributions.
        predicted_exp: Expectation parameters of the predicted distribution.
    """

    loss: JaxArray
    predicted_exp: EP

    def total_loss(self) -> JaxArray:
        return jnp.sum(self.loss)


class PerceptronTargetNode(eqx.Module):
    """Target node for observed variables: scores flat real predictions against a distribution.

    Stores a :class:`efax.Flattener` (``mapped_to_plane=True``) to unflatten the network's
    flat real output back to a :class:`efax.NaturalParametrization`, then computes the
    cross-entropy loss against the observed distribution.

    Attributes:
        flattener: Flattener for the distribution family, with ``mapped_to_plane=True``.
    """

    flattener: Flattener[Any] = eqx.field(static=True)

    @classmethod
    def create(
        cls,
        flattener: Flattener[NaturalParametrization[Any, Any]],
    ) -> Self:
        """Construct a PerceptronTargetNode from a flattener.

        Args:
            flattener: Flattener for the distribution family, with ``mapped_to_plane=True``.

        Returns:
            A new :class:`PerceptronTargetNode`.
        """
        assert flattener.mapped_to_plane
        return cls(flattener=flattener)

    def infer[EP: HasEntropyEP](
        self, observed_exp: EP, y_hat: JaxRealArray
    ) -> PerceptronTargetConfiguration[EP]:
        """Compute loss and predicted distribution from a flat predicted output.

        Unflattens ``y_hat`` to a :class:`efax.NaturalParametrization` and computes the
        cross-entropy against ``observed_exp``.

        Args:
            observed_exp: Expectation parameters of the observed distribution.
            y_hat: Flat predicted natural parameters, shape matching the flattener's output.

        Returns:
            PerceptronTargetConfiguration with per-element loss and predicted expectation
            parameters.
        """
        predicted_np = self.flattener.unflatten(y_hat)
        predicted_exp = predicted_np.to_exp()
        assert isinstance(predicted_exp, type(observed_exp))
        losses = observed_exp.cross_entropy(predicted_np)
        return PerceptronTargetConfiguration(loss=losses, predicted_exp=predicted_exp)
