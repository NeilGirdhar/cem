import equinox as eqx
import jax
import jax.numpy as jnp
from efax import ComplexVonMisesNP
from tjax import JaxArray

from cem.phasor.message import JaxComplexArray


class LossAndScore(eqx.Module):
    """Phasor reconstruction loss and its phasor-space gradient, computed jointly via autodiff.

    Attributes:
        loss: Mean von Mises KL divergence over the final phasor-feature axis.
        score: ∂sum(loss)/∂ẑ — gradient of the total loss w.r.t. predicted phasors.
    """

    loss: JaxArray
    score: JaxComplexArray

    def total_loss(self) -> JaxArray:
        """Return the scalar phasor reconstruction objective."""
        return jnp.sum(self.loss)


def phasor_reconstruction_loss_and_score(
    observed: JaxComplexArray, z_hat: JaxComplexArray
) -> LossAndScore:
    """Compute phasor reconstruction loss and score jointly.

    Treats observed and predicted phasors as von Mises natural parameters and computes their
    KL divergence directly in phasor space.  The final axis is the phasor-feature axis and is
    averaged into the per-example objective.

    Args:
        observed: Observed phasors, shape ``(..., features)``.
        z_hat: Predicted phasors from the network, shape ``(..., features)``.

    Returns:
        ``LossAndScore`` with:

        - ``loss``: mean reconstruction objective over phasor features, shape ``(...)``.
        - ``score``: gradient of ``jnp.sum(loss)`` w.r.t. ``z_hat``, shape ``(..., features)``.
    """

    def loss_fn(z: JaxComplexArray) -> tuple[JaxArray, JaxArray]:
        elementwise_loss = _elementwise_phasor_reconstruction_loss(observed, z)
        loss = jnp.mean(elementwise_loss, axis=-1)
        return jnp.sum(loss), loss

    (_, loss), score = jax.value_and_grad(loss_fn, has_aux=True)(z_hat)
    return LossAndScore(loss=loss, score=score)


def _elementwise_phasor_reconstruction_loss(z: JaxArray, z_hat: JaxArray) -> JaxArray:
    """Elementwise von Mises KL divergence from observed to predicted phasors.

    L(z, ẑ) = KL(ComplexVonMises(z) || ComplexVonMises(ẑ)).

    Args:
        z: Observed phasors.
        z_hat: Predicted phasors.

    Returns:
        Elementwise KL divergence, same shape as z and z_hat (real-valued).
    """
    observed = ComplexVonMisesNP(z)
    predicted = ComplexVonMisesNP(z_hat)
    return observed.to_exp().kl_divergence(predicted, self_nat=observed)


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
    """Adversarial information-leak loss: negative cross-entropy of target under prediction.

    Interprets ``prediction`` and ``target`` as natural parameters of independent
    ``ComplexVonMises`` distributions on the final axis.  Returns
    ``−Σ_features cross_entropy(VM(target_i), VM(prediction_i))``.

    Producer minimizing this maximizes the cross-entropy — i.e. makes samples from
    ``VM(target)`` hard for the critic's predicted distribution ``VM(prediction)`` to
    explain.  With ``negate_cotangent`` on ``prediction`` the critic sees the flipped
    gradient and effectively minimizes the cross-entropy, sharpening its prediction.

    Unlike the inner-product form ``Re(prediction^H · target)``, this loss
    correctly returns ~0 (gain ≈ 0) whenever either distribution is uniform
    (``|prediction|`` or ``|target|`` near zero), since neither side carries
    information to leak.  Its gradient on ``target`` is bounded by the
    Bessel-ratio geometry of the von Mises manifold.

    Args:
        prediction: Critic output, shape (..., features).
        target: Target phasors, shape (..., features).

    Returns:
        Negative summed cross-entropy, shape (...) — scalar per batch element.
    """
    target_exp = ComplexVonMisesNP(target).to_exp()
    predicted = ComplexVonMisesNP(prediction)
    # Σ (log 2π − cross_entropy) = Σ critic's log-likelihood gain over uniform.
    # The constant log(2π) shifts the loss so it reads 0 at the no-leak floor
    # (uniform critic predicting uniform producer), positive when leak is present,
    # and negative when the critic is doing worse than uniform — without changing
    # gradients.
    return jnp.sum(jnp.log(2.0 * jnp.pi) - target_exp.cross_entropy(predicted), axis=-1)
