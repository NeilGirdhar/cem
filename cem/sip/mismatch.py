"""SIP mismatch circuit: prediction map, witness, and purification losses."""

from collections.abc import Mapping
from typing import Self

import equinox as eqx
import jax
import jax.numpy as jnp
from efax import NormalNP
from tjax import JaxRealArray, RngStream

from cem.npn.gaussian import GaussianNPN, normal_from_mean_and_presence
from cem.sip.losses import SIPLossOutputs
from cem.sip.messages import (
    InstrumentedMessage,
    concat_messages,
    gaussian_cross_entropy,
    mean_and_precision,
)
from cem.sip.parameters import MismatchPredictionParameter, WitnessMultiplierParameter
from cem.sip.settings import MismatchSettings
from cem.sip.witness import WitnessMap, witness_loss


class MismatchOutputs(SIPLossOutputs):
    """The demand for a better explanation and the losses that train the prediction and witness."""

    mismatch: NormalNP  # 2n unit-presence components; demand for explanation
    reconstruction_cost: JaxRealArray  # scalar
    confounding_cost: JaxRealArray  # scalar
    witness_loss: JaxRealArray  # scalar


class _Moments(eqx.Module):
    """Target and prediction statistics with the gain that weights their mismatch."""

    target_mean: JaxRealArray
    target_variance: JaxRealArray
    pred_mean: JaxRealArray
    pred_variance: JaxRealArray
    gain: JaxRealArray | None


class _Reconstruction(eqx.Module):
    """The reconstruction cost and the outgoing demand for explanation."""

    cost: JaxRealArray
    mismatch: NormalNP


class Mismatch(eqx.Module):
    """Learn to predict a feature's observation and test for instrument-predictable errors.

    See `sip-mismatch-circuit` in the thesis.
    """

    prediction_map: GaussianNPN
    witness_map: WitnessMap
    witness_multiplier: WitnessMultiplierParameter  # scalar

    @classmethod
    def create(
        cls,
        n: int,
        *,
        predictor_features: int,
        gain_features: int,
        mismatch_settings: MismatchSettings,
        streams: Mapping[str, RngStream],
    ) -> Self:
        return cls(
            prediction_map=GaussianNPN.create(
                in_features=predictor_features,
                out_features=n,
                hidden_features=mismatch_settings.prediction_hidden_features,
                streams=streams,
                parameter_factory=MismatchPredictionParameter,
            ),
            witness_map=WitnessMap.create(
                n,
                instrument_features=predictor_features + gain_features,
                hidden_features=mismatch_settings.witness_hidden_features,
                streams=streams,
            ),
            witness_multiplier=WitnessMultiplierParameter(jnp.zeros(())),
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
        """Score the observation prediction and return its mismatch and training losses.

        The outgoing mismatch carries mean and raw second-moment errors. The witness instead
        tests centred residuals for error predictable from the instruments.

        When there is no gain parent, construct the witness without gain features. The circuit
        then uses unit gain for both the losses and the witness norm constraint.
        """
        gain_observation = None if gain is None else gain.observation
        gain_instrument = None if gain is None else gain.instrument
        moments = self._moments(
            raw_observation, local_variance, predictor.observation, gain_observation
        )
        reconstruction = _reconstruction_and_mismatch(moments)
        witness_residuals = _witness_residuals(moments)
        confounding_witness = self._confounding_witness(
            predictor.instrument, gain_instrument, streams=streams, inference=inference
        )
        confounding_cost = jnp.sum(
            _scale_rows(witness_residuals, moments.gain)
            * jax.lax.stop_gradient(confounding_witness)
        )
        loss = witness_loss(
            witness_residuals, confounding_witness, moments.gain, self.witness_multiplier.value
        )

        return MismatchOutputs(
            mismatch=reconstruction.mismatch,
            reconstruction_cost=reconstruction.cost,
            confounding_cost=confounding_cost,
            witness_loss=loss,
            balanced_losses={
                "reconstruction": reconstruction.cost,
                "confounding": confounding_cost,
            },
            fixed_weight_loss=loss,
        )

    def _moments(
        self,
        raw_observation: NormalNP,
        local_variance: JaxRealArray,
        predictor_observations: NormalNP,
        gain_observation: NormalNP | None,
    ) -> _Moments:
        """Include fresh local noise in the target variance while keeping the raw target mean.

        Hold the target fixed so that mismatch costs cannot move the observation or its local
        noise towards the prediction. Hold the predictor observations fixed so that the parents
        learn from their own goals rather than from these costs. Hold the gain fixed because
        differentiating a positive cost through it would favour reducing participation. See
        `sip-target-observation` and `sip-prediction-map` in the thesis.
        """
        raw_mean, raw_precision = mean_and_precision(raw_observation)
        pred_mean, pred_precision = mean_and_precision(
            self.prediction_map.infer(
                *mean_and_precision(jax.lax.stop_gradient(predictor_observations))
            )
        )
        gain = (
            None
            if gain_observation is None
            else jax.lax.stop_gradient(mean_and_precision(gain_observation)[0])
        )
        return _Moments(
            target_mean=jax.lax.stop_gradient(raw_mean),
            target_variance=jax.lax.stop_gradient(1 / raw_precision + local_variance),
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
        """Return the witness over the predictor and gain instruments jointly."""
        instruments = (
            predictor_instruments
            if gain_instrument is None
            else concat_messages(predictor_instruments, gain_instrument)
        )
        return self.witness_map.infer(instruments, streams=streams, inference=inference)


def _reconstruction_and_mismatch(moments: _Moments) -> _Reconstruction:
    mean_resid = moments.pred_mean - moments.target_mean
    recon_density = gaussian_cross_entropy(
        moments.target_mean, moments.target_variance, moments.pred_mean, moments.pred_variance
    )
    recon_error = recon_density if moments.gain is None else moments.gain * recon_density
    reconstruction_cost = jnp.sum(recon_error)

    second_moment_diff = (
        jnp.square(moments.pred_mean)
        + moments.pred_variance
        - jnp.square(moments.target_mean)
        - moments.target_variance
    )
    mismatch_pair = jnp.stack([mean_resid, second_moment_diff], axis=-1)
    mismatch_flat = _scale_rows(mismatch_pair, moments.gain).reshape(-1)
    mismatch_message = normal_from_mean_and_presence(mismatch_flat, jnp.ones_like(mismatch_flat))
    return _Reconstruction(cost=reconstruction_cost, mismatch=mismatch_message)


def _witness_residuals(moments: _Moments) -> JaxRealArray:
    mean_resid = moments.pred_mean - moments.target_mean
    var_resid = moments.pred_variance - moments.target_variance - jnp.square(mean_resid)
    return jnp.stack([mean_resid, var_resid], axis=-1)


def _scale_rows(x: JaxRealArray, gain: JaxRealArray | None) -> JaxRealArray:
    """Return `x` scaled per-channel by `gain` (shape (n,)), or unchanged if gain is None."""
    return x if gain is None else gain[:, None] * x
