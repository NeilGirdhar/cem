"""Real-valued emitter for self-instrumental purification."""

from collections.abc import Mapping
from typing import Self

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
from tjax import JaxRealArray, RngStream

from cem.perceptron.mlp import MLP
from cem.structure.graph import LearnableParameter


class _Linear(eqx.Module):
    """Bias-free real linear map used for inherited instruments."""

    weight: LearnableParameter[JaxRealArray]

    @classmethod
    def create(
        cls,
        in_features: int,
        out_features: int,
        *,
        stream: RngStream,
    ) -> Self:
        if in_features < 1 or out_features < 1:
            msg = "linear dimensions must be positive"
            raise ValueError(msg)
        scale = 1.0 / jnp.sqrt(jnp.asarray(in_features, dtype=jnp.float64))
        weight = scale * jr.normal(stream.key(), (out_features, in_features), dtype=jnp.float64)
        return cls(weight=LearnableParameter(weight))

    def project(self, x: JaxRealArray) -> JaxRealArray:
        """Apply the bias-free map to the final feature axis."""
        return x @ self.weight.value.T


class EmitterOutput(eqx.Module):
    """Observation and instrument emitted by one SIP feature."""

    observation: JaxRealArray
    instrument: JaxRealArray
    raw_observation: JaxRealArray
    inherited_instrument: JaxRealArray
    injected_noise: JaxRealArray


class SIPEmitter(eqx.Module):
    """Construct a real-valued observation and its self-instrument.

    The observation map uses innovation and goal features. A separate bias-free map
    propagates instruments inherited from causal parents. During training, independent
    Gaussian perturbations are added to both paths. Gain scales both emitted channels,
    so the instrument remains the exogenous component of the emitted observation.

    Noise magnitudes are represented by unconstrained logits and transformed with
    ``softplus``. Inference disables local experimentation, matching the usual
    train/evaluation distinction for stochastic interventions.
    """

    observation_map: MLP
    instrument_map: _Linear
    noise_logits: LearnableParameter[JaxRealArray]
    observation_features: int = eqx.field(static=True)
    goal_features: int = eqx.field(static=True)
    predictor_instrument_features: int = eqx.field(static=True)

    @classmethod
    def create(
        cls,
        innovation_features: int,
        goal_features: int,
        predictor_instrument_features: int,
        observation_features: int,
        *,
        hidden_features: int | tuple[int, ...] = (),
        initial_noise: float = 1e-3,
        streams: Mapping[str, RngStream],
    ) -> Self:
        dimensions = (
            innovation_features,
            goal_features,
            predictor_instrument_features,
            observation_features,
        )
        if any(dimension < 1 for dimension in dimensions):
            msg = "all emitter feature dimensions must be positive"
            raise ValueError(msg)
        if initial_noise <= 0.0:
            msg = "initial_noise must be positive"
            raise ValueError(msg)
        stream = streams["parameters"]
        inverse_softplus = jnp.log(jnp.expm1(jnp.asarray(initial_noise, dtype=jnp.float64)))
        return cls(
            observation_map=MLP.create(
                innovation_features + goal_features,
                observation_features,
                hidden_features=hidden_features,
                streams=streams,
            ),
            instrument_map=_Linear.create(
                predictor_instrument_features,
                observation_features,
                stream=stream,
            ),
            noise_logits=LearnableParameter(
                jnp.full(observation_features, inverse_softplus, dtype=jnp.float64)
            ),
            observation_features=observation_features,
            goal_features=goal_features,
            predictor_instrument_features=predictor_instrument_features,
        )

    @property
    def noise_magnitudes(self) -> JaxRealArray:
        """Learned nonnegative intervention scale per channel."""
        return jax.nn.softplus(self.noise_logits.value)

    @staticmethod
    def _feature_vector(value: JaxRealArray, expected_features: int) -> JaxRealArray:
        if value.ndim == 0:
            return value[jnp.newaxis]
        if value.ndim == 1 and value.shape[-1] != expected_features:
            return value[..., jnp.newaxis]
        return value

    def infer(
        self,
        innovation: JaxRealArray,
        goal: JaxRealArray,
        gain: JaxRealArray,
        predictor_instruments: JaxRealArray,
        *,
        streams: Mapping[str, RngStream],
        inference: bool,
    ) -> EmitterOutput:
        """Emit an observation and instrument for one batch or one example."""
        goal = self._feature_vector(goal, self.goal_features)
        predictor_instruments = self._feature_vector(
            predictor_instruments,
            self.predictor_instrument_features,
        )
        gain = self._feature_vector(gain, 1)
        if gain.shape[-1] != 1:
            msg = f"gain must have one feature, got {gain.shape[-1]}"
            raise ValueError(msg)
        expected_inputs = self.observation_map.layers[0].weight.value.shape[1]
        if innovation.shape[-1] + goal.shape[-1] != expected_inputs:
            msg = "innovation and goal dimensions do not match this emitter"
            raise ValueError(msg)

        raw_observation = self.observation_map.infer(
            jnp.concatenate((innovation, goal), axis=-1),
            streams=streams,
            inference=inference,
        )
        inherited_instrument = self.instrument_map.project(predictor_instruments)
        noise = jnp.zeros_like(raw_observation)
        if not inference:
            noise_magnitudes = jnp.reshape(
                self.noise_magnitudes,
                (1,) * (raw_observation.ndim - 1) + (-1,),
            )
            noise = noise_magnitudes * jr.normal(
                streams["inference"].key(),
                shape=raw_observation.shape,
                dtype=raw_observation.dtype,
            )
        gain = jnp.broadcast_to(gain, (*raw_observation.shape[:-1], 1))
        return EmitterOutput(
            observation=gain * (raw_observation + noise),
            instrument=gain * (inherited_instrument + noise),
            raw_observation=raw_observation,
            inherited_instrument=inherited_instrument,
            injected_noise=noise,
        )
