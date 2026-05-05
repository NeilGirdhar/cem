from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
from efax import ComplexVonMisesNP
from tjax import JaxArray

from cem.phasor.message import JaxComplexArray


class LossAndScore(eqx.Module):
    """Spectral reconstruction loss and its phasor-space gradient, computed jointly via autodiff.

    Attributes:
        loss: Summed von Mises cross-entropy over the final phasor-feature axis.
        score: ∂sum(loss)/∂ẑ — gradient of the total summed loss w.r.t. predicted phasors.
    """

    loss: JaxArray
    score: JaxComplexArray

    def total_loss(self) -> JaxArray:
        """Return the scalar spectral reconstruction objective."""
        return jnp.sum(self.loss)


def spectral_reconstruction_loss_and_score(
    observed: JaxComplexArray, z_hat: JaxComplexArray
) -> LossAndScore:
    """Compute spectral reconstruction loss and score jointly.

    Treats observed and predicted phasors as von Mises natural parameters and computes their
    cross-entropy directly in phasor space.  The final axis is the phasor-feature axis and is
    summed into the per-example objective.

    Args:
        observed: Observed phasors, shape ``(..., features)``.
        z_hat: Predicted phasors from the network, shape ``(..., features)``.

    Returns:
        ``LossAndScore`` with:

        - ``loss``: summed spectral objective, shape ``(...)``.
        - ``score``: gradient of ``jnp.sum(loss)`` w.r.t. ``z_hat``, shape ``(..., features)``.
    """

    def loss_fn(z: JaxComplexArray) -> tuple[JaxArray, JaxArray]:
        elementwise_loss = spectral_reconstruction_loss(observed, z)
        loss = jnp.sum(elementwise_loss, axis=-1)
        return jnp.sum(loss), loss

    (_, loss), score = jax.value_and_grad(loss_fn, has_aux=True)(z_hat)
    return LossAndScore(loss=loss, score=score)


def spectral_reconstruction_loss(z: JaxArray, z_hat: JaxArray) -> JaxArray:
    """Spectral reconstruction loss: von Mises cross-entropy between observed and predicted phasors.

    L(z, ẑ) = log(2π I₀(|ẑ|)) − Re(g(z) conj(ẑ))

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
    """Adversarial decorrelation concordance: Re(prediction^H target).

    L_crit = Re(Σᵢ conj(predictionᵢ) · targetᵢ)

    The critic h is trained to maximize this over its parameters by predicting target u from z,
    while the producer of z minimizes it, removing from z any information about u that the
    critic can detect.

    Args:
        prediction: Critic output h(z), shape (..., features).
        target: Target phasors u, shape (..., features).

    Returns:
        Concordance, shape (...) — a scalar per batch element.
    """
    return jnp.real(jnp.sum(jnp.conj(prediction) * target, axis=-1))
