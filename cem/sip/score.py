"""Real-valued score circuit for self-instrumental purification."""

from collections.abc import Mapping
from typing import Self

import equinox as eqx
import jax.numpy as jnp
from tjax import JaxRealArray, RngStream

from cem.perceptron.mlp import MLP


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

    prediction_map: MLP
    witness_map: MLP
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
        dimensions = (
            predictor_observation_features,
            predictor_instrument_features,
            observation_features,
        )
        if any(dimension < 1 for dimension in dimensions):
            msg = "all score feature dimensions must be positive"
            raise ValueError(msg)
        return cls(
            prediction_map=MLP.create(
                predictor_observation_features,
                observation_features,
                hidden_features=hidden_features,
                streams=streams,
            ),
            witness_map=MLP.create(
                predictor_instrument_features,
                observation_features,
                hidden_features=hidden_features,
                streams=streams,
            ),
            observation_features=observation_features,
            predictor_observation_features=predictor_observation_features,
            predictor_instrument_features=predictor_instrument_features,
        )

    @staticmethod
    def _feature_vector(value: JaxRealArray, expected_features: int) -> JaxRealArray:
        if value.ndim == 0:
            return value[jnp.newaxis]
        if value.ndim == 1 and value.shape[-1] != expected_features:
            return value[..., jnp.newaxis]
        return value

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
        observation = self._feature_vector(observation, self.observation_features)
        predictor_observations = self._feature_vector(
            predictor_observations,
            self.predictor_observation_features,
        )
        predictor_instruments = self._feature_vector(
            predictor_instruments,
            self.predictor_instrument_features,
        )
        gain = self._feature_vector(gain, 1)
        if gain.shape[-1] != 1:
            msg = f"gain must have one feature, got {gain.shape[-1]}"
            raise ValueError(msg)

        raw_prediction = self.prediction_map.infer(
            predictor_observations,
            streams=streams,
            inference=inference,
        )
        raw_witness = self.witness_map.infer(
            predictor_instruments,
            streams=streams,
            inference=inference,
        )
        witness_norm = jnp.sqrt(jnp.mean(jnp.square(raw_witness), axis=-1, keepdims=True) + 1e-8)
        normalized_witness = raw_witness / witness_norm
        gain = jnp.broadcast_to(gain, (*raw_prediction.shape[:-1], 1))
        prediction = gain * raw_prediction
        witness = gain * normalized_witness
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
