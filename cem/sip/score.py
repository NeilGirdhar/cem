"""Real-valued score circuit for self-instrumental purification."""

from collections.abc import Mapping
from typing import Self

import equinox as eqx
import jax.numpy as jnp
from tjax import JaxRealArray, RngStream

from cem.sip.predictor_witness import PredictorWitnessPair, feature_vector


class ScoreOutput(eqx.Module):
    """Prediction, reconstruction score, and instrument-conditioned witness."""

    prediction: JaxRealArray
    observation_score: JaxRealArray
    reconstruction_loss: JaxRealArray
    witness: JaxRealArray
    confounding_error: JaxRealArray


class SIPScore(eqx.Module):
    """Score real-valued observations against predictions.

    The prediction map consumes predictor observations. The witness map consumes
    predictor instruments through a separate path. Gain scales both outputs. For
    squared reconstruction loss, the observation score is the cotangent with
    respect to the prediction: ``prediction - observation``.

    This is the first score-circuit layer. It exposes the confounding error but
    leaves adversarial witness optimization and precision constraints to the next
    SIP component.
    """

    predictor: PredictorWitnessPair
    observation_features: int = eqx.field(static=True)
    predictor_observation_features: int = eqx.field(static=True)
    predictor_instrument_features: int = eqx.field(static=True)

    @classmethod
    def create(
        cls,
        predictor_observation_features: int,
        predictor_instrument_features: int,
        observation_features: int,
        *,
        hidden_features: int | tuple[int, ...] = (),
        streams: Mapping[str, RngStream],
    ) -> Self:
        predictor = PredictorWitnessPair.create(
            predictor_observation_features,
            predictor_instrument_features,
            observation_features,
            hidden_features=hidden_features,
            streams=streams,
        )
        return cls(
            predictor=predictor,
            observation_features=observation_features,
            predictor_observation_features=predictor_observation_features,
            predictor_instrument_features=predictor_instrument_features,
        )

    def infer(
        self,
        observation: JaxRealArray,
        predictor_observations: JaxRealArray,
        predictor_instruments: JaxRealArray,
        gain: JaxRealArray,
        *,
        streams: Mapping[str, RngStream],
        inference: bool,
    ) -> ScoreOutput:
        """Score one observation or a batch of observations."""
        observation = feature_vector(observation, self.observation_features)
        prediction = self.predictor.prediction(
            predictor_observations,
            gain,
            streams=streams,
            inference=inference,
        )
        witness = self.predictor.witness(
            predictor_instruments,
            gain,
            streams=streams,
            inference=inference,
        )
        observation_score = prediction - observation
        reconstruction_loss = 0.5 * jnp.sum(jnp.square(observation_score), axis=-1)
        confounding_error = jnp.sum(observation_score * witness, axis=-1)
        return ScoreOutput(
            prediction=prediction,
            observation_score=observation_score,
            reconstruction_loss=reconstruction_loss,
            witness=witness,
            confounding_error=confounding_error,
        )
