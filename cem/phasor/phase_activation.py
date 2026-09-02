from collections.abc import Mapping
from typing import Self

import equinox as eqx
import jax.numpy as jnp
from tjax import JaxRealArray, RngStream

from cem.phasor.elementwise_rotation import FrequencyElementwiseRotation
from cem.phasor.message import JaxComplexArray
from cem.structure.graph import LearnableParameter


class FrequencyPhaseActivation(eqx.Module):
    """Rotate coherent phasor features and attenuate them by phase alone."""

    rotation: FrequencyElementwiseRotation
    log_sharpnesses: LearnableParameter[JaxRealArray]

    @classmethod
    def create(
        cls,
        features: int,
        frequencies: JaxRealArray,
        *,
        streams: Mapping[str, RngStream],
    ) -> Self:
        return cls(
            rotation=FrequencyElementwiseRotation.create(features, frequencies, streams=streams),
            log_sharpnesses=LearnableParameter(jnp.zeros(features, dtype=jnp.float64)),
        )

    def activate(self, x: JaxComplexArray) -> JaxComplexArray:
        """Apply the circular-exponential phase activation."""
        rotated = self.rotation.rotate(x)
        n_frequencies = self.rotation.frequencies.value.shape[0]
        features = self.log_sharpnesses.value.shape[0]
        grouped = rotated.reshape(*rotated.shape[:-1], features, n_frequencies)
        presence = jnp.abs(grouped)
        unit = grouped / jnp.where(presence > 0, presence, 1)
        sharpnesses = jnp.exp(self.log_sharpnesses.value)
        sharpness_shape = (1,) * (grouped.ndim - 2) + (features, 1)
        sharpnesses = jnp.reshape(sharpnesses, sharpness_shape)
        gate = jnp.exp(sharpnesses * (jnp.real(unit) - 1))
        return (gate * grouped).reshape(*rotated.shape[:-1], features * n_frequencies)
