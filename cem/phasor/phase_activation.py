from collections.abc import Mapping
from typing import Self

import equinox as eqx
import jax.numpy as jnp
from tjax import JaxRealArray, RngStream

from cem.phasor.elementwise_rotation import ElementwiseRotation
from cem.phasor.message import JaxComplexArray
from cem.structure.graph import LearnableParameter


class PhaseActivation(eqx.Module):
    """Rotate phasors and attenuate their presence according to phase alone.

    The circular-exponential gate equals one at the learned preferred phase and
    remains positive elsewhere.  Input presence does not affect the attenuation.

    Attributes:
        rotation: Learned preferred phase for each feature.
        log_sharpnesses: Learned logarithms of positive attenuation sharpnesses.
    """

    rotation: ElementwiseRotation
    log_sharpnesses: LearnableParameter[JaxRealArray]

    @classmethod
    def create(
        cls,
        features: int,
        *,
        streams: Mapping[str, RngStream],
    ) -> Self:
        return cls(
            rotation=ElementwiseRotation.create(features, streams=streams),
            log_sharpnesses=LearnableParameter(jnp.zeros(features, dtype=jnp.float64)),
        )

    def activate(self, x: JaxComplexArray) -> JaxComplexArray:
        """Apply learned phase-only attenuation."""
        rotated = self.rotation.rotate(x)
        presence = jnp.abs(rotated)
        unit = rotated / jnp.where(presence > 0, presence, 1)
        sharpnesses = jnp.exp(self.log_sharpnesses.value)
        sharpnesses = jnp.reshape(sharpnesses, (1,) * (x.ndim - 1) + (-1,))
        gate = jnp.exp(sharpnesses * (jnp.real(unit) - 1))
        return gate * rotated
