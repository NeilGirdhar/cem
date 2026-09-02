from collections.abc import Mapping
from typing import Self

import equinox as eqx
import jax.numpy as jnp
from tjax import JaxRealArray, RngStream

from cem.phasor.message import JaxComplexArray
from cem.structure.graph import LearnableParameter


class ElementwiseRotation(eqx.Module):
    """Rotate each phasor independently without changing its evidence magnitude."""

    displacements: LearnableParameter[JaxRealArray]

    @classmethod
    def create(
        cls,
        features: int,
        *,
        streams: Mapping[str, RngStream],
    ) -> Self:
        del streams
        return cls(displacements=LearnableParameter(jnp.zeros(features, dtype=jnp.float64)))

    def rotate(self, x: JaxComplexArray) -> JaxComplexArray:
        """Apply the learned phase rotations."""
        if x.shape[-1] != self.displacements.value.shape[0]:
            msg = (
                f"expected final input dimension {self.displacements.value.shape[0]}, "
                f"got {x.shape[-1]}"
            )
            raise ValueError(msg)
        rotation_shape = (1,) * (x.ndim - 1) + self.displacements.value.shape
        rotations = jnp.exp(1j * jnp.reshape(self.displacements.value, rotation_shape))
        return x * rotations
