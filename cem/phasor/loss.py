from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
from efax import ComplexVonMisesNP
from tjax import JaxArray

from cem.phasor.message import PhasorMessage


class LossAndScore(eqx.Module):
    """Reconstruction loss and its phasor-space gradient, computed jointly via autodiff.

    Attributes:
        loss: Per-element reconstruction cross-entropy, same shape as the input phasors.
        score: ∂loss/∂ẑ — gradient of the summed loss w.r.t. predicted phasors.
    """

    loss: JaxArray
    score: PhasorMessage

    def total_loss(self) -> JaxArray:
        """Return the summed scalar reconstruction loss."""
        return jnp.sum(self.loss)


def reconstruction_loss_and_score(observed: PhasorMessage, z_hat: PhasorMessage) -> LossAndScore:
    """Compute reconstruction loss and score jointly for a latent variable in phasor space.

    No distribution family is assumed.  Observed and predicted phasors are treated as von Mises
    natural parameters.  Uses ``jax.value_and_grad`` to obtain both the per-element losses and
    the score ∂loss/∂ẑ in a single pass.

    Args:
        observed: Observed phasors.
        z_hat: Predicted phasors from the network.

    Returns:
        LossAndScore with per-element reconstruction cross-entropy and score ∂loss/∂ẑ.
    """

    def loss_fn(z: PhasorMessage) -> tuple[JaxArray, JaxArray]:
        losses = reconstruction_loss(observed.data, z.data)
        return jnp.sum(losses), losses

    (_, losses), score = jax.value_and_grad(loss_fn, has_aux=True)(z_hat)
    return LossAndScore(loss=losses, score=score)


def reconstruction_loss(z: JaxArray, z_hat: JaxArray) -> JaxArray:
    """Von Mises cross-entropy: log-normalizer minus expected log-likelihood.

    L(z, ẑ) = log(2π I₀(|ẑ|)) − Re(g(z) conj(ẑ))

    This is the standard exponential-family cross-entropy: log-normalizer of the prediction
    minus the inner product of the observation's expectation parameters with the prediction's
    natural parameters.  The score is the Wirtinger derivative of this loss with respect to ẑ.

    Args:
        z: Observed phasors.
        z_hat: Predicted phasors.

    Returns:
        Elementwise cross-entropy, same shape as z and z_hat (real-valued).
    """
    return ComplexVonMisesNP(z).to_exp().cross_entropy(ComplexVonMisesNP(z_hat))


def centering_loss(z: JaxArray) -> JaxArray:
    """Centering penalty: suppresses non-zero mean phasor over the batch.

    L_center = Σⱼ |E_batch[zⱼ]|²

    Encourages circular symmetry of the phasor distribution — the missing-at-random (MAR)
    condition that evidential strength should not depend on value direction.

    Args:
        z: Batch of phasors, shape (batch, features) or (..., features).

    Returns:
        Scalar penalty, ≥ 0.
    """
    batch_axes = tuple(range(z.ndim - 1))
    return jnp.sum(jnp.abs(jnp.mean(z, axis=batch_axes)) ** 2)


def strength_loss(z: JaxArray) -> JaxArray:
    """Strength penalty: prevents collapse to zero presence.

    L_strength = −Σⱼ E_batch[log I₀(|zⱼ|)]

    Minimizing this encourages larger presence.  Together with centering_loss it maximizes
    the entropy of the von Mises conjugate prior, keeping representations non-degenerate.

    Args:
        z: Batch of phasors, shape (batch, features) or (..., features).

    Returns:
        Scalar penalty, ≤ 0.
    """
    log_i0 = ComplexVonMisesNP(z).log_normalizer() - jnp.log(2.0 * jnp.pi)
    batch_axes = tuple(range(z.ndim - 1))
    return -jnp.sum(jnp.mean(log_i0, axis=batch_axes))


def decorrelation_loss(prediction: JaxArray, target: JaxArray) -> JaxArray:
    """Adversarial decorrelation alignment: Re(prediction^H target).

    L_crit = Re(Σᵢ conj(predictionᵢ) · targetᵢ)

    The critic h is trained to maximize this over its parameters by predicting target u from z,
    while the producer of z minimizes it, removing from z any information about u that the
    critic can detect.

    Args:
        prediction: Critic output h(z), shape (..., features).
        target: Target phasors u, shape (..., features).

    Returns:
        Alignment, shape (...) — a scalar per batch element.
    """
    return jnp.real(jnp.sum(jnp.conj(prediction) * target, axis=-1))
