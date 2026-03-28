from __future__ import annotations

import jax.numpy as jnp
import jax.scipy.special as jss
from tjax import JaxArray


def expectation_params(z: JaxArray) -> JaxArray:
    """Map von Mises natural parameters to expectation parameters.

    g(z) = A₁(|z|) / |z| * z,  where A₁(r) = I₁(r) / I₀(r).

    The magnitude |g(z)| = A₁(|z|) lies in [0, 1); phase is preserved.  Uses
    exponentially scaled Bessel functions for numerical stability at large |z|.
    At z = 0, the limit A₁(r)/r → 1/2 is applied.

    Args:
        z: Natural-parameter phasors, any shape.

    Returns:
        Expectation-parameter phasors, same shape as z.
    """
    r = jnp.abs(z)
    a1 = jss.i1e(r) / jss.i0e(r)  # I₁(r) / I₀(r), stable for all r ≥ 0
    scale = jnp.where(r > 0, a1 / r, 0.5)
    return scale * z


def score(z: JaxArray, z_hat: JaxArray) -> JaxArray:
    """Von Mises score: difference of expectation parameters.

    Score(z, ẑ) = g(ẑ) − g(z)

    Points from the predicted distribution toward the observation in natural-parameter space.
    Serves as both a learning cotangent for the predictor and a primal signal for
    predictive-coding inference.

    Args:
        z: Observed phasors.
        z_hat: Predicted phasors.

    Returns:
        Score, same shape as z and z_hat.
    """
    return expectation_params(z_hat) - expectation_params(z)


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
    r_hat = jnp.abs(z_hat)
    log_normalizer = jnp.log(2.0 * jnp.pi) + r_hat + jnp.log(jss.i0e(r_hat))
    return log_normalizer - jnp.real(expectation_params(z) * jnp.conj(z_hat))


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
    r = jnp.abs(z)
    log_i0 = r + jnp.log(jss.i0e(r))  # log I₀(r) = r + log i0e(r), numerically stable
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
