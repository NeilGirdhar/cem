"""SIP confounding witness: a bounded test of residuals against instrument context."""

from collections.abc import Mapping
from typing import Self

import equinox as eqx
import jax
import jax.numpy as jnp
from efax import NormalNP, NormalVP
from tjax import JaxRealArray, RngStream

from cem.perceptron.mlp import MLP


class WitnessMap(eqx.Module):
    """Map instrument messages to one bounded mean and one variance coefficient per channel.

    The witness must stay finite and small for certain instruments with tiny displacements, treat
    every missing instrument alike regardless of its mean, and stay neutral for an instrument
    context that supplies neither displacement nor spread. See `sip-confounding-witness` in the
    thesis.

    The instruments enter held fixed, which leaves instrument fitting to each emitter's instrument
    mismatch.

    Each Gaussian instrument coordinate enters as the pair (u, r) with r = 1 / (1 + σ²) and
    u = r m. The perceptron output is centred at the all-certain-zero context, whose pairs all equal
    (0, 1), and passed through tanh.
    """

    perceptron: MLP

    @classmethod
    def create(
        cls,
        n: int,
        *,
        instrument_features: int,
        hidden_features: int | tuple[int, ...],
        streams: Mapping[str, RngStream],
    ) -> Self:
        """Build the witness map for `n` channels from `instrument_features` instrument inputs."""
        return cls(
            perceptron=MLP.create(
                in_features=2 * instrument_features,
                out_features=2 * n,
                hidden_features=hidden_features,
                streams=streams,
            )
        )

    def infer(
        self,
        instruments: NormalNP,
        *,
        streams: Mapping[str, RngStream],
        inference: bool,
    ) -> JaxRealArray:
        """Return the witness, shape (n, 2), for the concatenated instrument messages.

        The perceptron has no dropout, so subtracting its output at the reference is exact.
        """
        encoded = _encode(jax.lax.stop_gradient(instruments))
        reference = jnp.concat([jnp.zeros_like(encoded[0]), jnp.ones_like(encoded[1])])
        batch = jnp.stack([jnp.concat(encoded), reference])
        output, reference_output = self.perceptron.infer(
            batch, streams=streams, inference=inference
        )
        return jnp.tanh(output - reference_output).reshape(-1, 2)


def _encode(instruments: NormalNP) -> tuple[JaxRealArray, JaxRealArray]:
    """Return (u, r) from natural parameters, giving (0, 0) exactly at zero precision.

    With precision κ, r = κ / (1 + κ) and u = (m κ) / (1 + κ). Both stay bounded as κ grows.
    """
    precision = -2.0 * instruments.negative_half_precision
    scale = 1.0 / (1.0 + precision)
    return scale * instruments.mean_times_precision, scale * precision


def witness_residuals(target: NormalVP, prediction: NormalVP) -> JaxRealArray:
    """Return the mean and variance residuals of a prediction, shape (n, 2).

    The variance residual compares the predicted variance with the target's expected squared
    error about the predicted mean, so each residual has zero expectation for a calibrated
    prediction. See `sip-witness-residuals` in the thesis.
    """
    mean_residual = prediction.mean - target.mean
    variance_residual = prediction.variance - target.variance - jnp.square(mean_residual)
    return jnp.stack([mean_residual, variance_residual], axis=-1)


def witness_loss(
    witness_residuals: JaxRealArray,
    witness: JaxRealArray,
    gain: JaxRealArray | None,
    multiplier: JaxRealArray,
) -> JaxRealArray:
    """Return the witness loss plus its Lagrangian norm-constraint term.

    Descending this loss trains the witness map to increase confounding error within the
    participation-weighted norm bound and trains the multiplier to ascend on the bound's excess.
    The constraint term contributes gradients only, so the value is the negated confounding
    covariance. Without a gain, every channel has unit gain. See `sip-norm-constraint` in the
    thesis.

    Args:
        witness_residuals: The mean and variance residuals, shape (n, 2), held fixed.
        witness: The confounding witness, shape (n, 2).
        gain: The gain observation, shape (n,), held fixed, or None.
        multiplier: The witness multiplier, a nonnegative scalar.
    """
    if gain is None:
        gated_residuals = jax.lax.stop_gradient(witness_residuals)
        squared_norm = jnp.sum(jnp.square(witness))
        norm_bound = 1.0
    else:
        gain = jax.lax.stop_gradient(gain)
        gated_residuals = jax.lax.stop_gradient(gain[:, None] * witness_residuals)
        squared_norm = jnp.sum(gain[:, None] * jnp.square(witness))
        norm_bound = jnp.mean(gain)
    covariance = jnp.sum(gated_residuals * witness)
    excess = squared_norm - norm_bound
    # The witness map descends on the multiplier times the excess, while the multiplier ascends.
    fixed_multiplier = jax.lax.stop_gradient(multiplier)
    constraint = fixed_multiplier * excess - multiplier * jax.lax.stop_gradient(excess)
    return constraint - covariance
