"""SIP witnesses: integral and proportional tests of residuals against instrument context."""

from collections.abc import Mapping
from typing import Self

import equinox as eqx
import jax
import jax.numpy as jnp
from efax import NormalNP, NormalVP
from tjax import JaxRealArray, RngStream

from cem.perceptron.mlp import MLP


class _InstrumentPerceptron(eqx.Module):
    """A perceptron over encoded instruments, centred at the all-certain-zero context.

    Each Gaussian instrument coordinate enters as the pair (u, r) with r = 1 / (1 + σ²) and
    u = r m. The instruments enter held fixed, which leaves instrument fitting to each emitter's
    instrument loss. The all-certain-zero context, whose pairs all equal (0, 1), supplies neither
    displacement nor spread, so the output is centred there.
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
        """Build the perceptron for `n` channels from `instrument_features` instrument inputs."""
        return cls(
            perceptron=MLP.create(
                in_features=2 * instrument_features,
                out_features=2 * n,
                hidden_features=hidden_features,
                streams=streams,
            )
        )

    def centred_output(
        self,
        instruments: NormalNP,
        *,
        streams: Mapping[str, RngStream],
        inference: bool,
    ) -> JaxRealArray:
        """Return the perceptron output minus its output at the reference, shape (n, 2).

        The perceptron has no dropout, so subtracting its output at the reference is exact.
        """
        encoded = _encode(jax.lax.stop_gradient(instruments))
        reference = jnp.concat([jnp.zeros_like(encoded[0]), jnp.ones_like(encoded[1])])
        batch = jnp.stack([jnp.concat(encoded), reference])
        output, reference_output = self.perceptron.infer(
            batch, streams=streams, inference=inference
        )
        return (output - reference_output).reshape(-1, 2)


class IntegralWitnessMap(_InstrumentPerceptron):
    """Map instrument messages to one bounded mean and one variance coefficient per channel.

    The integral witness accumulates instrument-predictable residuals into the correction that
    holds the prediction at the conditional moment restriction. It must stay finite and small for
    certain instruments with tiny displacements, treat every missing instrument alike regardless
    of its mean, and stay neutral for an instrument context that supplies neither displacement nor
    spread. See `sip-confounding-witness` in the thesis.
    """

    def infer(
        self,
        instruments: NormalNP,
        *,
        streams: Mapping[str, RngStream],
        inference: bool,
    ) -> JaxRealArray:
        """Return the integral witness, shape (n, 2), bounded to (-1, 1) by tanh."""
        return jnp.tanh(self.centred_output(instruments, streams=streams, inference=inference))


class ProportionalWitnessMap(_InstrumentPerceptron):
    """Estimate the gain-weighted conditional mean of each residual given the instruments.

    The estimate's scale carries information, so the output is unbounded. See
    `sip-proportional-witness` in the thesis.
    """

    def infer(
        self,
        instruments: NormalNP,
        *,
        streams: Mapping[str, RngStream],
        inference: bool,
    ) -> JaxRealArray:
        """Return the proportional witness, shape (n, 2)."""
        return self.centred_output(instruments, streams=streams, inference=inference)


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


def integral_witness_loss(
    residuals: JaxRealArray, witness: JaxRealArray, gain: JaxRealArray | None
) -> JaxRealArray:
    """Return the integral witness loss, the gain-weighted sum of residual-witness products.

    The prediction descends this loss with the witness held fixed, and the integral witness ascends
    it with the residuals held fixed; callers apply those stop-gradients. Without a gain, every
    channel has unit gain. See `sip-confounding-witness` in the thesis.

    Args:
        residuals: The mean and variance residuals, shape (n, 2).
        witness: The integral witness, shape (n, 2).
        gain: The gain observation, shape (n,), held fixed, or None.
    """
    return jnp.sum(_gain_weights(gain) * residuals * witness)


def proportional_witness_loss(
    residuals: JaxRealArray, witness: JaxRealArray, gain: JaxRealArray | None
) -> JaxRealArray:
    """Return the proportional witness loss, Σ γ (w δ - w² / 2).

    Ascending it with the residuals held fixed is a gain-weighted regression of the residuals on
    the instruments. Descending it with the witness held fixed descends the squared conditional
    moment. Callers apply those stop-gradients. Without a gain, every channel has unit gain. See
    `sip-proportional-witness` in the thesis.

    Args:
        residuals: The mean and variance residuals, shape (n, 2).
        witness: The proportional witness, shape (n, 2).
        gain: The gain observation, shape (n,), held fixed, or None.
    """
    return jnp.sum(_gain_weights(gain) * (witness * residuals - 0.5 * jnp.square(witness)))


def witness_norm_loss(
    witness: JaxRealArray, gain: JaxRealArray | None, multiplier: JaxRealArray
) -> JaxRealArray:
    """Return the witness norm loss, which bounds the integral witness's gain-weighted norm.

    The loss is the multiplier times the excess of the gain-weighted squared norm over mean gain.
    Descending it trains the witness map with the multiplier held fixed, and trains the multiplier
    to ascend on the excess with the witness held fixed. The two terms cancel in value, so the loss
    contributes gradients only. Without a gain, every channel has unit gain and the bound is one.
    See `sip-norm-constraint` in the thesis.

    Args:
        witness: The integral witness, shape (n, 2).
        gain: The gain observation, shape (n,), held fixed, or None.
        multiplier: The witness multiplier, a nonnegative scalar.
    """
    if gain is None:
        squared_norm = jnp.sum(jnp.square(witness))
        norm_bound = 1.0
    else:
        gain = jax.lax.stop_gradient(gain)
        squared_norm = jnp.sum(gain[:, None] * jnp.square(witness))
        norm_bound = jnp.mean(gain)
    excess = squared_norm - norm_bound
    fixed_multiplier = jax.lax.stop_gradient(multiplier)
    return fixed_multiplier * excess - multiplier * jax.lax.stop_gradient(excess)


class WitnessLosses(eqx.Module):
    """The confounding cost and the losses that train the two witnesses."""

    confounding_cost: JaxRealArray  # scalar; descended by the prediction
    integral_witness_loss: JaxRealArray  # scalar; ascended by the integral witness
    proportional_witness_loss: JaxRealArray  # scalar; ascended by the proportional witness
    witness_norm_loss: JaxRealArray  # scalar; descended by the integral witness

    @property
    def loss(self) -> JaxRealArray:
        """The scalar whose descent applies every witness-related update."""
        return (
            self.confounding_cost
            - self.integral_witness_loss
            - self.proportional_witness_loss
            + self.witness_norm_loss
        )


def witness_losses(
    residuals: JaxRealArray,
    *,
    integral_witness: JaxRealArray,
    proportional_witness: JaxRealArray,
    gain: JaxRealArray | None,
    multiplier: JaxRealArray,
    damping_weight: JaxRealArray,
) -> WitnessLosses:
    """Return the confounding cost and the witness losses for one prediction.

    The confounding cost combines the two witness losses with the witnesses held fixed, weighting
    the proportional loss by the damping weight; the prediction descends it. Each witness ascends
    its own loss with the residuals held fixed. See `sip-prediction-purification` and
    `sip-witness-learning` in the thesis.

    Args:
        residuals: The mean and variance residuals, shape (n, 2).
        integral_witness: The integral witness, shape (n, 2).
        proportional_witness: The proportional witness, shape (n, 2).
        gain: The gain observation, shape (n,), held fixed, or None for unit gain.
        multiplier: The witness multiplier, a nonnegative scalar.
        damping_weight: The fixed positive weight on the proportional loss.
    """
    fixed_residuals = jax.lax.stop_gradient(residuals)
    fixed_integral = jax.lax.stop_gradient(integral_witness)
    fixed_proportional = jax.lax.stop_gradient(proportional_witness)
    confounding_cost = integral_witness_loss(
        residuals, fixed_integral, gain
    ) + damping_weight * proportional_witness_loss(residuals, fixed_proportional, gain)
    return WitnessLosses(
        confounding_cost=confounding_cost,
        integral_witness_loss=integral_witness_loss(fixed_residuals, integral_witness, gain),
        proportional_witness_loss=proportional_witness_loss(
            fixed_residuals, proportional_witness, gain
        ),
        witness_norm_loss=witness_norm_loss(integral_witness, gain, multiplier),
    )


def _gain_weights(gain: JaxRealArray | None) -> JaxRealArray | float:
    """Return the fixed gain repeated across each channel's two residual coordinates."""
    return 1.0 if gain is None else jax.lax.stop_gradient(gain)[:, None]
