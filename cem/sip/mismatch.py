"""SIP mismatch circuit: prediction map, witness, and purification losses.

See `sip-mismatch-circuit` in the thesis.
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
from cem.sip.messages import InstrumentedMessage, mean_and_precision, promoting_link_loss
from cem.structure.graph import LearnableParameter


class MismatchOutputs(eqx.Module):
    """One step of the mismatch circuit."""

    mismatch: NormalNP  # 2n unit-presence components; demand for explanation
    reconstruction_cost: JaxRealArray  # scalar
    confounding_cost: JaxRealArray  # scalar
    witness_loss: JaxRealArray  # scalar
    dissonance: JaxRealArray  # scalar; reconstruction and confounding
    disutility: JaxRealArray  # scalar; reconstruction - confounding
    loss: JaxRealArray  # scalar; combined surrogate for first-pass training


class _Moments(eqx.Module):
    target_mean: JaxRealArray
    target_variance: JaxRealArray
    pred_mean: JaxRealArray
    pred_variance: JaxRealArray
    gain: JaxRealArray | None


class Mismatch(eqx.Module):
    """A circuit that learns a causal predictor of a feature's observation.

    See `sip-mismatch-circuit` in the thesis.
    """

    prediction_map: GaussianNPN
    witness_map: MLP
    reconstruction_cost_weight: LearnableParameter[JaxRealArray]  # raw (n,)
    confounding_cost_weight: LearnableParameter[JaxRealArray]  # raw (n, 2)
    witness_multiplier: LearnableParameter[JaxRealArray]  # raw ()

    @classmethod
    def create(
        cls,
        n: int,
        *,
        predictor_features: int,
        gain_features: int,
        prediction_hidden_features: int | tuple[int, ...] = (),
        witness_hidden_features: int | tuple[int, ...] = (),
        streams: Mapping[str, RngStream],
    ) -> Self:
        return cls(
            prediction_map=GaussianNPN.create(
                in_features=predictor_features,
                out_features=n,
                hidden_features=prediction_hidden_features,
                streams=streams,
            ),
            witness_map=MLP.create(
                in_features=2 * (predictor_features + gain_features),
                out_features=2 * n,
                hidden_features=witness_hidden_features,
                streams=streams,
            ),
            reconstruction_cost_weight=LearnableParameter(jnp.zeros(n, dtype=jnp.float64)),
            confounding_cost_weight=LearnableParameter(jnp.zeros((n, 2), dtype=jnp.float64)),
            witness_multiplier=LearnableParameter(jnp.asarray(0.0, dtype=jnp.float64)),
        )

    def step(
        self,
        *,
        raw_observation: NormalNP,
        local_variance: JaxRealArray,
        predictor: InstrumentedMessage,
        gain: InstrumentedMessage | None,
        streams: Mapping[str, RngStream],
        inference: bool,
    ) -> MismatchOutputs:
        """Advance the mismatch circuit one time step.

        Pass `gain=None` for a recruiter feature at the top of the gain chain;
        gain multiplies drop out and the witness norm bound reduces to the
        unit-gain case `‖w‖² ≤ 1`. The witness map's input must then have been
        sized with `gain_features=0`.
        """
        gain_observation = None if gain is None else gain.observation
        gain_instrument = None if gain is None else gain.instrument
        moments = self._moments(
            raw_observation, local_variance, predictor.observation, gain_observation
        )
        # Loss weights are LearnableParameters now, but their adaptive rule
        # lives in the (not-yet-implemented) multi-loss gradient transform.
        # Stop-gradient the readout so single-loss Adam doesn't collapse them
        # to zero. Remove the stop_gradient when the multi-loss transform lands.
        r_weight = jax.nn.softplus(jax.lax.stop_gradient(self.reconstruction_cost_weight.value))
        c_weight = jax.nn.softplus(jax.lax.stop_gradient(self.confounding_cost_weight.value))

        reconstruction_cost, mismatch_message = _reconstruction_and_mismatch(moments, r_weight)
        witness_residuals = _witness_residuals(moments)
        confounding_witness = self._confounding_witness(
            predictor.instrument, gain_instrument, streams=streams, inference=inference
        )
        confounding_cost = jnp.sum(
            c_weight
            * _scale_rows(witness_residuals, moments.gain)
            * jax.lax.stop_gradient(confounding_witness)
        )
        witness_loss, multiplier_loss = self._witness_training(
            moments.gain, witness_residuals, confounding_witness
        )

        return MismatchOutputs(
            mismatch=mismatch_message,
            reconstruction_cost=reconstruction_cost,
            confounding_cost=confounding_cost,
            witness_loss=witness_loss,
            dissonance=reconstruction_cost + confounding_cost,
            disutility=reconstruction_cost - confounding_cost,
            loss=reconstruction_cost + confounding_cost + witness_loss + multiplier_loss,
        )

    def _moments(
        self,
        raw_observation: NormalNP,
        local_variance: JaxRealArray,
        predictor_observations: NormalNP,
        gain_observation: NormalNP | None,
    ) -> _Moments:
        raw_mean, raw_precision = mean_and_precision(raw_observation)
        pred_mean, pred_precision = mean_and_precision(
            self.prediction_map.infer(*mean_and_precision(predictor_observations))
        )
        gain = None if gain_observation is None else mean_and_precision(gain_observation)[0]
        return _Moments(
            target_mean=raw_mean,
            target_variance=1 / raw_precision + local_variance,
            pred_mean=pred_mean,
            pred_variance=1 / pred_precision,
            gain=gain,
        )

    def _confounding_witness(
        self,
        predictor_instruments: NormalNP,
        gain_instrument: NormalNP | None,
        *,
        streams: Mapping[str, RngStream],
        inference: bool,
    ) -> JaxRealArray:
        parts = [
            predictor_instruments.mean_times_precision,
            predictor_instruments.negative_half_precision,
        ]
        if gain_instrument is not None:
            parts.extend(
                [gain_instrument.mean_times_precision, gain_instrument.negative_half_precision]
            )
        witness_input = jnp.concat(parts)
        return self.witness_map.infer(witness_input, streams=streams, inference=inference).reshape(
            -1, 2
        )

    def _witness_training(
        self,
        gain: JaxRealArray | None,
        witness_residuals: JaxRealArray,
        confounding_witness: JaxRealArray,
    ) -> tuple[JaxRealArray, JaxRealArray]:
        covariance = -jnp.sum(
            _scale_rows(jax.lax.stop_gradient(witness_residuals), gain) * confounding_witness
        )
        norm_bound = 1.0 if gain is None else jnp.mean(gain)
        constraint_violation = (
            jnp.sum(_scale_rows(jnp.square(confounding_witness), gain)) - norm_bound
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
    recon_density = 0.5 * (
        jnp.log(2.0 * jnp.pi * moments.pred_variance)
        + (moments.target_variance + jnp.square(mean_resid)) / moments.pred_variance
    )
    recon_error = recon_density if moments.gain is None else moments.gain * recon_density
    reconstruction_cost = jnp.sum(r_weight * recon_error)

    second_moment_diff = (
        jnp.square(moments.pred_mean)
        + moments.pred_variance
        - jnp.square(moments.target_mean)
        - moments.target_variance
    )
    mismatch_pair = jnp.stack([mean_resid, second_moment_diff], axis=-1)
    presence = r_weight if moments.gain is None else r_weight * moments.gain
    mismatch_flat = (presence[:, None] * mismatch_pair).reshape(-1)
    mismatch_message = normal_from_mean_and_presence(mismatch_flat, jnp.ones_like(mismatch_flat))
    return reconstruction_cost, mismatch_message


def _witness_residuals(moments: _Moments) -> JaxRealArray:
    mean_resid = moments.pred_mean - moments.target_mean
    var_resid = moments.pred_variance - moments.target_variance - jnp.square(mean_resid)
    return jnp.stack([mean_resid, var_resid], axis=-1)


def _scale_rows(x: JaxRealArray, gain: JaxRealArray | None) -> JaxRealArray:
    """Return `x` scaled per-channel by `gain` (shape (n,)), or unchanged if gain is None."""
    return x if gain is None else gain[:, None] * x
