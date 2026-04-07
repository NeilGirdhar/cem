from __future__ import annotations

from typing import Any, Self

import equinox as eqx
import jax
import jax.numpy as jnp
from efax import Flattener, HasEntropyEP, NaturalParametrization
from tjax import JaxArray

from cem.phasor.frequency import make_frequency_grid
from cem.phasor.message import PhasorMessage
from cem.structure.graph.node import NodeConfiguration


class PhasorTargetConfiguration[EP: HasEntropyEP](NodeConfiguration):
    """Output of a PhasorTargetNode: score, loss, and recovered predicted distribution.

    Attributes:
        score: ∂loss/∂ẑ — gradient of the loss w.r.t. predicted phasors.
        loss: Per-element reconstruction loss between observation and prediction.
        predicted_exp: Expectation parameters of the predicted distribution, recovered from z_hat.
    """

    score: PhasorMessage
    loss: JaxArray
    predicted_exp: EP

    def total_loss(self) -> JaxArray:
        return jnp.sum(self.loss)


class PhasorTargetNode(eqx.Module):
    """Target node for observed variables: scores predictions against a known distribution family.

    Stores the frequency grid needed to recover expectation parameters from predicted phasors via
    PhasorMessage.to_distribution.  The score ∂loss/∂ẑ is in phasor space, computed by
    autodiff through to_distribution and to_nat, serving as both a learning cotangent and a
    predictive-coding inference signal.

    Attributes:
        t: Frequency grid of shape (m * d,), built from the distribution family and frequencies.
    """

    t: NaturalParametrization[Any, Any]

    @classmethod
    def create(
        cls,
        flattener: Flattener[NaturalParametrization[Any, Any]],
        frequencies: JaxArray,
    ) -> Self:
        """Build the frequency grid from a flattener and frequency array.

        Args:
            flattener: Flattener for the distribution family, defining the natural-parameter
                dimension d.
            frequencies: Geometric frequency grid, shape (m,).

        Returns:
            PhasorTargetNode with t of shape (m * d,).
        """
        return cls(t=make_frequency_grid(flattener, frequencies))

    def infer[EP: HasEntropyEP](
        self, observed_exp: EP, z_hat: PhasorMessage
    ) -> PhasorTargetConfiguration[EP]:
        """Compute score, loss, and predicted distribution from a predicted phasor.

        Args:
            observed_exp: Expectation parameters of the observed distribution, e.g. dist.to_exp().
            z_hat: Predicted phasors from the network.

        Returns:
            PhasorTargetConfiguration with score ∂loss/∂ẑ, per-element reconstruction
            cross-entropy, and recovered expectation parameters.
        """

        def forward(z: PhasorMessage) -> tuple[JaxArray, tuple[JaxArray, EP]]:
            predicted_exp = z.to_distribution(self.t)
            assert isinstance(predicted_exp, type(observed_exp))
            losses = observed_exp.cross_entropy(predicted_exp.to_nat())
            return jnp.sum(losses), (losses, predicted_exp)

        (_, (losses, predicted_exp)), score = jax.value_and_grad(forward, has_aux=True)(z_hat)
        return PhasorTargetConfiguration(score=score, loss=losses, predicted_exp=predicted_exp)
