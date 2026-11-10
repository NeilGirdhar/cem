"""SIP mismatch circuit: prediction map, witness, and purification losses."""

from collections.abc import Mapping
from typing import Self

import equinox as eqx
import jax
import jax.numpy as jnp
from efax import NormalNP, NormalVP
from tjax import JaxRealArray, RngStream

from cem.npn.gaussian import GaussianNPN, normal_from_mean_and_presence
from cem.sip.losses import SIPLossOutputs
from cem.sip.messages import InstrumentedMessage, concat_messages, mean_and_precision
from cem.sip.parameters import MismatchPredictionParameter, WitnessMultiplierParameter
from cem.sip.settings import MismatchSettings
from cem.sip.witness import WitnessMap, witness_loss, witness_residuals


class MismatchOutputs(SIPLossOutputs):
    """The demand for a better explanation and the losses that train the prediction and witness."""

    mismatch: NormalNP  # 2n unit-presence components; demand for explanation
    reconstruction_cost: JaxRealArray  # scalar
    confounding_cost: JaxRealArray  # scalar
    witness_loss: JaxRealArray  # scalar


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
        residuals = witness_residuals(moments.target, moments.prediction)
        confounding_witness = self._confounding_witness(
            predictor.instrument, gain_instrument, streams=streams, inference=inference
        )
        confounding_cost = jnp.sum(
            _scale_rows(residuals, moments.gain) * jax.lax.stop_gradient(confounding_witness)
        )
        loss = witness_loss(
            residuals, confounding_witness, moments.gain, self.witness_multiplier.value
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
        raw = raw_observation.to_variance_parametrization()
        target = NormalVP(raw.mean, raw.variance + local_variance)
        prediction = self.prediction_map.infer(
            *mean_and_precision(jax.lax.stop_gradient(predictor_observations))
        )
        gain = None if gain_observation is None else gain_observation.mean()
        return _Moments(
            target=jax.lax.stop_gradient(target),
            prediction=prediction.to_variance_parametrization(),
            gain=None if gain is None else jax.lax.stop_gradient(gain),
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
    target, prediction = moments.target, moments.prediction
    mean_residual = prediction.mean - target.mean
    reconstruction_density = target.cross_entropy(prediction)
    reconstruction_error = (
        reconstruction_density if moments.gain is None else moments.gain * reconstruction_density
    )
    reconstruction_cost = jnp.sum(reconstruction_error)

    second_moment_difference = prediction.to_exp().second_moment - target.to_exp().second_moment
    mismatch_pair = jnp.stack([mean_residual, second_moment_difference], axis=-1)
    mismatch_flat = _scale_rows(mismatch_pair, moments.gain).reshape(-1)
    mismatch_message = normal_from_mean_and_presence(mismatch_flat, jnp.ones_like(mismatch_flat))
    return _Reconstruction(cost=reconstruction_cost, mismatch=mismatch_message)


def _scale_rows(x: JaxRealArray, gain: JaxRealArray | None) -> JaxRealArray:
    """Return `x` scaled per-channel by `gain` (shape (n,)), or unchanged if gain is None."""
    return x if gain is None else gain[:, None] * x
