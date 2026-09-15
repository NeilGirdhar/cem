"""Shared prediction and confounding-witness maps for SIP score circuits."""

from collections.abc import Mapping
from typing import Self

import equinox as eqx
import jax.numpy as jnp
from tjax import JaxRealArray, RngStream

from cem.perceptron.mlp import MLP


def feature_vector(value: JaxRealArray, expected_features: int) -> JaxRealArray:
    """Promote a scalar or a bare batch of scalars to an explicit feature axis."""
    if value.ndim == 0:
        return value[jnp.newaxis]
    if value.ndim == 1 and value.shape[-1] != expected_features:
        return value[..., jnp.newaxis]
    return value


def _gain_vector(gain: JaxRealArray, batch_shape: tuple[int, ...]) -> JaxRealArray:
    gain = feature_vector(gain, 1)
    if gain.shape[-1] != 1:
        msg = f"gain must have one feature, got {gain.shape[-1]}"
        raise ValueError(msg)
    return jnp.broadcast_to(gain, (*batch_shape, 1))


class PredictorWitnessPair(eqx.Module):
    """A gain-scaled prediction map paired with a gain-scaled confounding witness.

    SIPScore and SIPTDError both purify a prediction against a confounding witness
    built from predictor instruments; this pair factors their shared machinery. The
    witness is normalized to unit mean-square magnitude before gain, so witness
    learning cannot inflate confounding error merely by growing witness magnitude.
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
            msg = "all predictor feature dimensions must be positive"
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

    def prediction(
        self,
        predictor_observations: JaxRealArray,
        gain: JaxRealArray,
        *,
        streams: Mapping[str, RngStream],
        inference: bool,
    ) -> JaxRealArray:
        """Gain-scaled prediction from predictor observations."""
        predictor_observations = feature_vector(
            predictor_observations,
            self.predictor_observation_features,
        )
        raw_prediction = self.prediction_map.infer(
            predictor_observations,
            streams=streams,
            inference=inference,
        )
        return _gain_vector(gain, raw_prediction.shape[:-1]) * raw_prediction

    def witness(
        self,
        predictor_instruments: JaxRealArray,
        gain: JaxRealArray,
        *,
        streams: Mapping[str, RngStream],
        inference: bool,
    ) -> JaxRealArray:
        """Gain-scaled, unit-normalized confounding witness from predictor instruments."""
        predictor_instruments = feature_vector(
            predictor_instruments,
            self.predictor_instrument_features,
        )
        raw_witness = self.witness_map.infer(
            predictor_instruments,
            streams=streams,
            inference=inference,
        )
        witness_norm = jnp.sqrt(jnp.mean(jnp.square(raw_witness), axis=-1, keepdims=True) + 1e-8)
        normalized_witness = raw_witness / witness_norm
        return _gain_vector(gain, normalized_witness.shape[:-1]) * normalized_witness
