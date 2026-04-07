from __future__ import annotations

from typing import Any, Self

import equinox as eqx
import jax
import jax.numpy as jnp
from efax import Flattener, HasEntropyEP, NaturalParametrization
from tjax import JaxArray, JaxRealArray

from cem.phasor_calculus.loss import reconstruction_loss
from cem.phasor_calculus.message import PhasorMessage, make_frequency_grid
from cem.structure.graph.node import NodeConfiguration


class ScoreOutput(NodeConfiguration):
    """Base class for score node outputs: phasor-space score and reconstruction loss.

    Attributes:
        score: ∂loss/∂ẑ — gradient of the loss w.r.t. predicted phasors.
        loss: Reconstruction loss between observation and prediction.
    """

    score: PhasorMessage
    loss: JaxArray

    def total_loss(self) -> JaxArray:
        return jnp.sum(self.loss)


class ObservedScoreOutput[EP: HasEntropyEP](ScoreOutput):
    """Score node output for observed variables, adding the recovered predicted distribution.

    Attributes:
        predicted_exp: Expectation parameters of the predicted distribution, recovered from z_hat.
    """

    predicted_exp: EP


def latent_score(observed: PhasorMessage, z_hat: PhasorMessage) -> ScoreOutput:
    """Compute score and reconstruction loss for a latent variable in phasor space.

    No distribution family is assumed.  Observed and predicted phasors are treated as von Mises
    natural parameters.  The score ∂loss/∂ẑ is the gradient of the total (summed) reconstruction
    loss w.r.t. the predicted phasors; the per-element losses are stored in the output.

    Args:
        observed: Observed phasors.
        z_hat: Predicted phasors from the network.

    Returns:
        ScoreOutput with score ∂loss/∂ẑ and per-element reconstruction cross-entropy.
    """

    def loss_fn(z: PhasorMessage) -> tuple[JaxArray, JaxArray]:
        losses = reconstruction_loss(observed.data, z.data)
        return jnp.sum(losses), losses

    (_, losses), score = jax.value_and_grad(loss_fn, has_aux=True)(z_hat)
    return ScoreOutput(score=score, loss=losses)


class ObservedScoreNode(eqx.Module):
    """Score node for observed variables: recovers distributions via a known family and frequencies.

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
        frequencies: JaxRealArray,
    ) -> Self:
        """Build the frequency grid from a flattener and frequency array.

        Args:
            flattener: Flattener for the distribution family, defining the natural-parameter
                dimension d.
            frequencies: Geometric frequency grid, shape (m,).

        Returns:
            ObservedScoreNode with t of shape (m * d,).
        """
        return cls(t=make_frequency_grid(flattener, frequencies))

    def infer[EP: HasEntropyEP](
        self, observed_exp: EP, z_hat: PhasorMessage
    ) -> ObservedScoreOutput[EP]:
        """Compute score and reconstruction loss.

        Args:
            observed_exp: Expectation parameters of the observed distribution, e.g. dist.to_exp().
            z_hat: Predicted phasors from the network.

        Returns:
            ObservedScoreOutput with score ∂loss/∂ẑ and per-element reconstruction cross-entropy.
        """

        def forward(z: PhasorMessage) -> tuple[JaxArray, tuple[JaxArray, EP]]:
            predicted_exp = z.to_distribution(self.t)
            assert isinstance(predicted_exp, type(observed_exp))
            losses = observed_exp.cross_entropy(predicted_exp.to_nat())
            return jnp.sum(losses), (losses, predicted_exp)

        (_, (losses, predicted_exp)), score = jax.value_and_grad(forward, has_aux=True)(z_hat)
        return ObservedScoreOutput(predicted_exp=predicted_exp, score=score, loss=losses)
