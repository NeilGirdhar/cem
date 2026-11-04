"""SIP mismatch circuit: prediction map, witness, and purification losses.

See `sip-mismatch-circuit` in `~/src/typst/thesis.typ`.
"""

from collections.abc import Mapping
from typing import Self

import equinox as eqx
import jax
import jax.numpy as jnp
from efax import NormalNP
from tjax import JaxRealArray, RngStream

from cem.npn.gaussian import GaussianNPN, normal_from_mean_and_presence
from cem.perceptron.mlp import MLP
from cem.sip.messages import mean_and_precision, promoting_link_loss
from cem.structure.graph import LearnableParameter


class MismatchOutputs(eqx.Module):
    """One step of the mismatch circuit."""

    mismatch: NormalNP  # 2n unit-presence components; demand for explanation
    reconstruction_loss: JaxRealArray  # scalar
    confounding_loss: JaxRealArray  # scalar
    witness_loss: JaxRealArray  # scalar
    dissonance: JaxRealArray  # scalar; reconstruction + confounding
    disutility: JaxRealArray  # scalar; reconstruction - confounding
    loss: JaxRealArray  # scalar; combined surrogate for first-pass training


class _Moments(eqx.Module):
    target_mean: JaxRealArray
    target_variance: JaxRealArray
    pred_mean: JaxRealArray
    pred_variance: JaxRealArray
    gain: JaxRealArray


class Mismatch(eqx.Module):
    """One feature's mismatch circuit: predict, witness, purify.

    The prediction map fits the target observation; one joint witness MLP takes
    the natural parameters of the predictor and gain instruments and searches
    for instrument-predictable residuals; a Lagrange multiplier constrains the
    witness's gain-weighted squared norm.
    """

    prediction_map: GaussianNPN
    witness_map: MLP
    reconstruction_loss_weight: LearnableParameter[JaxRealArray]  # raw (n,)
    confounding_loss_weight: LearnableParameter[JaxRealArray]  # raw (n, 2)
    witness_multiplier: LearnableParameter[JaxRealArray]  # raw ()

    @classmethod
    def create(
        cls,
        n: int,
        *,
        predictor_observation_features: int,
        predictor_instrument_features: int,
        gain_instrument_features: int,
        prediction_hidden_features: int | tuple[int, ...] = (),
        witness_hidden_features: int | tuple[int, ...] = (),
        streams: Mapping[str, RngStream],
    ) -> Self:
        return cls(
            prediction_map=GaussianNPN.create(
                in_features=predictor_observation_features,
                out_features=n,
                hidden_features=prediction_hidden_features,
                streams=streams,
            ),
            witness_map=MLP.create(
                in_features=2 * (predictor_instrument_features + gain_instrument_features),
                out_features=2 * n,
                hidden_features=witness_hidden_features,
                streams=streams,
            ),
            reconstruction_loss_weight=LearnableParameter(jnp.zeros(n, dtype=jnp.float64)),
            confounding_loss_weight=LearnableParameter(jnp.zeros((n, 2), dtype=jnp.float64)),
            witness_multiplier=LearnableParameter(jnp.asarray(0.0, dtype=jnp.float64)),
        )

    def step(
        self,
        *,
        raw_observation: NormalNP,
        local_variance: JaxRealArray,
        predictor_observations: NormalNP,
        predictor_instruments: NormalNP,
        gain_observation: NormalNP,
        gain_instrument: NormalNP,
        streams: Mapping[str, RngStream],
        inference: bool,
    ) -> MismatchOutputs:
        """Advance the mismatch circuit one time step."""
        moments = self._moments(
            raw_observation, local_variance, predictor_observations, gain_observation
        )
        # Loss weights are LearnableParameters now, but their adaptive rule
        # lives in the (not-yet-implemented) multi-loss gradient transform.
        # Stop-gradient the readout so single-loss Adam doesn't collapse them
        # to zero. Remove the stop_gradient when the multi-loss transform lands.
        r_weight = jax.nn.softplus(jax.lax.stop_gradient(self.reconstruction_loss_weight.value))
        c_weight = jax.nn.softplus(jax.lax.stop_gradient(self.confounding_loss_weight.value))

        reconstruction_loss, mismatch_message = _reconstruction_and_mismatch(moments, r_weight)
        witness_residuals = _witness_residuals(moments)
        confounding_witness = self._confounding_witness(
            predictor_instruments, gain_instrument, streams=streams, inference=inference
        )
        confounding_loss = jnp.sum(
            c_weight
            * moments.gain[:, None]
            * witness_residuals
            * jax.lax.stop_gradient(confounding_witness)
        )
        witness_loss, multiplier_loss = self._witness_training(
            moments.gain, witness_residuals, confounding_witness
        )

        return MismatchOutputs(
            mismatch=mismatch_message,
            reconstruction_loss=reconstruction_loss,
            confounding_loss=confounding_loss,
            witness_loss=witness_loss,
            dissonance=reconstruction_loss + confounding_loss,
            disutility=reconstruction_loss - confounding_loss,
            loss=reconstruction_loss + confounding_loss + witness_loss + multiplier_loss,
        )

    def _moments(
        self,
        raw_observation: NormalNP,
        local_variance: JaxRealArray,
        predictor_observations: NormalNP,
        gain_observation: NormalNP,
    ) -> _Moments:
        raw_mean, raw_precision = mean_and_precision(raw_observation)
        pred_mean, pred_precision = mean_and_precision(
            self.prediction_map.infer(*mean_and_precision(predictor_observations))
        )
        gain, _ = mean_and_precision(gain_observation)
        return _Moments(
            target_mean=raw_mean,
            target_variance=jnp.reciprocal(raw_precision) + local_variance,
            pred_mean=pred_mean,
            pred_variance=jnp.reciprocal(pred_precision),
            gain=gain,
        )

    def _confounding_witness(
        self,
        predictor_instruments: NormalNP,
        gain_instrument: NormalNP,
        *,
        streams: Mapping[str, RngStream],
        inference: bool,
    ) -> JaxRealArray:
        witness_input = jnp.concatenate(
            [
                predictor_instruments.mean_times_precision,
                predictor_instruments.negative_half_precision,
                gain_instrument.mean_times_precision,
                gain_instrument.negative_half_precision,
            ]
        )
        return self.witness_map.infer(witness_input, streams=streams, inference=inference).reshape(
            -1, 2
        )

    def _witness_training(
        self,
        gain: JaxRealArray,
        witness_residuals: JaxRealArray,
        confounding_witness: JaxRealArray,
    ) -> tuple[JaxRealArray, JaxRealArray]:
        covariance = -jnp.sum(
            gain[:, None] * jax.lax.stop_gradient(witness_residuals) * confounding_witness
        )
        constraint_violation = jnp.sum(jnp.square(confounding_witness) * gain[:, None]) - jnp.mean(
            gain
        )
        lambda_value = jax.nn.softplus(self.witness_multiplier.value)
        witness_loss = covariance + jax.lax.stop_gradient(lambda_value) * constraint_violation
        multiplier_loss = promoting_link_loss(
            lambda_value, jax.lax.stop_gradient(constraint_violation)
        )
        return witness_loss, multiplier_loss


def _reconstruction_and_mismatch(
    moments: _Moments, r_weight: JaxRealArray
) -> tuple[JaxRealArray, NormalNP]:
    mean_resid = moments.pred_mean - moments.target_mean
    recon_error = (
        moments.gain
        * 0.5
        * (
            jnp.log(2.0 * jnp.pi * moments.pred_variance)
            + (moments.target_variance + jnp.square(mean_resid)) / moments.pred_variance
        )
    )
    reconstruction_loss = jnp.sum(r_weight * recon_error)

    second_moment_diff = (
        jnp.square(moments.pred_mean)
        + moments.pred_variance
        - jnp.square(moments.target_mean)
        - moments.target_variance
    )
    mismatch_pair = jnp.stack([mean_resid, second_moment_diff], axis=-1)
    mismatch_flat = ((r_weight * moments.gain)[:, None] * mismatch_pair).reshape(-1)
    mismatch_message = normal_from_mean_and_presence(mismatch_flat, jnp.ones_like(mismatch_flat))
    return reconstruction_loss, mismatch_message


def _witness_residuals(moments: _Moments) -> JaxRealArray:
    mean_resid = moments.pred_mean - moments.target_mean
    var_resid = moments.pred_variance - moments.target_variance - jnp.square(mean_resid)
    return jnp.stack([mean_resid, var_resid], axis=-1)
