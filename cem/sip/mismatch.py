"""SIP mismatch circuit: prediction map, integral and proportional witnesses, and objectives."""

from collections.abc import Mapping
from typing import Self

import equinox as eqx
import jax
import jax.numpy as jnp
from efax import NormalNP, NormalPP, NormalVP, parameter_concat
from tjax import JaxRealArray, RngStream

from cem.framework.graph import FixedParameter
from cem.npn.gaussian import GaussianNeuralProjection
from cem.sip.losses import SIPLossOutputs
from cem.sip.messages import InstrumentedMessage
from cem.sip.parameters import WitnessMultiplierParameter
from cem.sip.settings import MismatchSettings
from cem.sip.witness import (
    IntegralWitnessMap,
    ProportionalWitnessMap,
    witness_objectives,
    witness_residuals,
)


class MismatchOutputs(SIPLossOutputs):
    """The demand for explanation and the objectives that train the prediction and witnesses."""

    mismatch: NormalNP  # 2n unit-presence components; demand for explanation
    reconstruction_loss: JaxRealArray  # scalar
    confounding_objective: JaxRealArray  # scalar
    integral_witness_objective: JaxRealArray  # scalar
    proportional_witness_objective: JaxRealArray  # scalar
    witness_norm_objective: JaxRealArray  # scalar


class _Moments(eqx.Module):
    """The target and prediction with the gain that weights their mismatch."""

    target: NormalVP
    prediction: NormalVP
    gain: JaxRealArray | None


class _Reconstruction(eqx.Module):
    """The reconstruction loss and the outgoing demand for explanation."""

    loss: JaxRealArray
    mismatch: NormalNP


class Mismatch(eqx.Module):
    """Learn to predict a feature's observation and test for instrument-predictable residuals.

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
        """Score the observation prediction and return its mismatch and training objectives.

        The outgoing mismatch carries, for each channel, the target-minus-prediction mean residual
        and the deviation residual: the target's root-mean-square deviation about the predicted
        mean minus the predicted standard deviation. Both scale linearly with the messages, so
        mismatch content does not compound powers of scale as it passes between features. The
        witnesses instead test centred residuals for patterns predictable from the instruments.

        When there is no gain parent, construct the witnesses without gain features. The circuit
        then uses unit gain for the objectives and the witness norm constraint.
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
            else parameter_concat([predictor.instrument, gain_instrument])
        )
        objectives = witness_objectives(
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
            reconstruction_loss=reconstruction.loss,
            confounding_objective=objectives.confounding_objective,
            integral_witness_objective=objectives.integral_witness_objective,
            proportional_witness_objective=objectives.proportional_witness_objective,
            witness_norm_objective=objectives.witness_norm_objective,
            loss=reconstruction.loss + objectives.loss,
        )

    def _moments(
        self,
        raw_observation: NormalNP,
        local_variance: JaxRealArray,
        predictor_observations: NormalNP,
        gain_observation: NormalNP | None,
    ) -> _Moments:
        """Include fresh local noise in the target variance while keeping the raw target mean.

        Hold the target fixed so that the reconstruction loss and confounding objective cannot move
        the observation or its local noise towards the prediction. Hold the predictor observations
        fixed so that the parents learn from their own goals rather than from these signals. Hold
        the gain fixed because differentiating a positive loss through it would favour reducing
        participation. See
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
    mean_residual = target.mean - prediction.mean
    reconstruction_density = target.cross_entropy(prediction)
    channel_reconstruction_losses = (
        reconstruction_density if moments.gain is None else moments.gain * reconstruction_density
    )
    reconstruction_loss = jnp.sum(channel_reconstruction_losses)

    # The target's root-mean-square deviation about the predicted mean.
    target_deviation = jnp.sqrt(target.variance + jnp.square(mean_residual))
    deviation_residual = target_deviation - jnp.sqrt(prediction.variance)
    mismatch_pair = jnp.stack([mean_residual, deviation_residual], axis=-1)
    mismatch_flat = _scale_rows(mismatch_pair, moments.gain).reshape(-1)
    mismatch_message = NormalPP(mismatch_flat, jnp.ones_like(mismatch_flat)).to_nat()
    return _Reconstruction(loss=reconstruction_loss, mismatch=mismatch_message)


def _scale_rows(x: JaxRealArray, gain: JaxRealArray | None) -> JaxRealArray:
    """Return `x` scaled per-channel by `gain` (shape (n,)), or unchanged if gain is None."""
    return x if gain is None else gain[:, None] * x
