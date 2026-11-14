"""SIP mismatch circuit: prediction map, integral and proportional witnesses, and their losses."""

from collections.abc import Mapping
from typing import Self

import equinox as eqx
import jax
import jax.numpy as jnp
from efax import NormalNP, NormalPP, NormalVP
from tjax import JaxRealArray, RngStream

from cem.framework.graph import FixedParameter
from cem.npn.gaussian import GaussianNeuralProjection
from cem.sip.losses import SIPLossOutputs
from cem.sip.messages import InstrumentedMessage, concat_messages
from cem.sip.parameters import WitnessMultiplierParameter
from cem.sip.settings import MismatchSettings
from cem.sip.witness import (
    IntegralWitnessMap,
    ProportionalWitnessMap,
    witness_losses,
    witness_residuals,
)


class MismatchOutputs(SIPLossOutputs):
    """The demand for explanation and the losses that train the prediction and witnesses."""

    mismatch: NormalNP  # 2n unit-presence components; demand for explanation
    reconstruction_cost: JaxRealArray  # scalar
    confounding_cost: JaxRealArray  # scalar
    integral_witness_loss: JaxRealArray  # scalar
    proportional_witness_loss: JaxRealArray  # scalar
    witness_norm_loss: JaxRealArray  # scalar


class _Moments(eqx.Module):
    """The target and prediction with the gain that weights their mismatch."""

    target: NormalVP
    prediction: NormalVP
    gain: JaxRealArray | None


class _Reconstruction(eqx.Module):
    """The reconstruction cost and the outgoing demand for explanation."""

    cost: JaxRealArray
    mismatch: NormalNP


class Mismatch(eqx.Module):
    """Learn to predict a feature's observation and test for instrument-predictable errors.

    See `sip-mismatch-circuit` in the thesis.
    """

    prediction_map: GaussianNeuralProjection
    integral_witness_map: IntegralWitnessMap
    proportional_witness_map: ProportionalWitnessMap
    witness_multiplier: WitnessMultiplierParameter  # scalar
    damping_weight: FixedParameter[JaxRealArray]  # scalar

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
            prediction_map=GaussianNeuralProjection.create(
                in_features=predictor_features,
                out_features=n,
                hidden_features=mismatch_settings.prediction_hidden_features,
                streams=streams,
            ),
            integral_witness_map=IntegralWitnessMap.create(
                n,
                instrument_features=predictor_features + gain_features,
                hidden_features=mismatch_settings.integral_witness_hidden_features,
                streams=streams,
            ),
            proportional_witness_map=ProportionalWitnessMap.create(
                n,
                instrument_features=predictor_features + gain_features,
                hidden_features=mismatch_settings.proportional_witness_hidden_features,
                streams=streams,
            ),
            witness_multiplier=WitnessMultiplierParameter(jnp.zeros(())),
            damping_weight=FixedParameter(
                jnp.asarray(mismatch_settings.damping_weight, dtype=jnp.float64)
            ),
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

        The outgoing mismatch carries, for each channel, the mean error and the deviation error:
        the predicted standard deviation minus the target's root-mean-square deviation about the
        predicted mean. Both scale linearly with the messages, so mismatch content does not
        compound powers of scale as it passes between features. The witnesses instead test centred
        residuals for error predictable from the instruments.

        When there is no gain parent, construct the witnesses without gain features. The circuit
        then uses unit gain for the losses and the witness norm constraint.
        """
        gain_observation = None if gain is None else gain.observation
        gain_instrument = None if gain is None else gain.instrument
        moments = self._moments(
            raw_observation, local_variance, predictor.observation, gain_observation
        )
        reconstruction = _reconstruction_and_mismatch(moments)
        residuals = witness_residuals(moments.target, moments.prediction)
        instruments = (
            predictor.instrument
            if gain_instrument is None
            else concat_messages(predictor.instrument, gain_instrument)
        )
        losses = witness_losses(
            residuals,
            integral_witness=self.integral_witness_map.infer(
                instruments, streams=streams, inference=inference
            ),
            proportional_witness=self.proportional_witness_map.infer(
                instruments, streams=streams, inference=inference
            ),
            gain=moments.gain,
            multiplier=self.witness_multiplier.value,
            damping_weight=self.damping_weight.value,
        )

        return MismatchOutputs(
            mismatch=reconstruction.mismatch,
            reconstruction_cost=reconstruction.cost,
            confounding_cost=losses.confounding_cost,
            integral_witness_loss=losses.integral_witness_loss,
            proportional_witness_loss=losses.proportional_witness_loss,
            witness_norm_loss=losses.witness_norm_loss,
            loss=reconstruction.cost + losses.loss,
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
        raw = raw_observation.to_variance_parametrization()
        target = NormalVP(raw.mean, raw.variance + local_variance)
        prediction = self.prediction_map.infer(jax.lax.stop_gradient(predictor_observations))
        gain = None if gain_observation is None else gain_observation.mean()
        return _Moments(
            target=jax.lax.stop_gradient(target),
            prediction=prediction.to_variance_parametrization(),
            gain=None if gain is None else jax.lax.stop_gradient(gain),
        )


def _reconstruction_and_mismatch(moments: _Moments) -> _Reconstruction:
    target, prediction = moments.target, moments.prediction
    mean_error = prediction.mean - target.mean
    reconstruction_density = target.cross_entropy(prediction)
    reconstruction_error = (
        reconstruction_density if moments.gain is None else moments.gain * reconstruction_density
    )
    reconstruction_cost = jnp.sum(reconstruction_error)

    # The target's root-mean-square deviation about the predicted mean.
    target_deviation = jnp.sqrt(target.variance + jnp.square(mean_error))
    deviation_error = jnp.sqrt(prediction.variance) - target_deviation
    mismatch_pair = jnp.stack([mean_error, deviation_error], axis=-1)
    mismatch_flat = _scale_rows(mismatch_pair, moments.gain).reshape(-1)
    mismatch_message = NormalPP(mismatch_flat, jnp.ones_like(mismatch_flat)).to_nat()
    return _Reconstruction(cost=reconstruction_cost, mismatch=mismatch_message)


def _scale_rows(x: JaxRealArray, gain: JaxRealArray | None) -> JaxRealArray:
    """Return `x` scaled per-channel by `gain` (shape (n,)), or unchanged if gain is None."""
    return x if gain is None else gain[:, None] * x
