from collections.abc import Mapping
from typing import Self

import equinox as eqx
import jax.numpy as jnp
from tjax import JaxRealArray, RngStream

from cem.phasor.message import JaxComplexArray
from cem.structure.graph import FixedParameter, LearnableParameter


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


class FrequencyElementwiseRotation(eqx.Module):
    """Translate each represented input value without mixing input features.

    Each feature learns one value-space displacement. At angular frequency
    ``omega``, that displacement becomes the unit rotation
    ``exp(1j * omega * displacement)``. The operation therefore preserves
    evidence magnitude and keeps the frequency bank coherent.
    """

    displacements: LearnableParameter[JaxRealArray]
    frequencies: FixedParameter[JaxRealArray]

    @classmethod
    def create(
        cls,
        features: int,
        frequencies: JaxRealArray,
        *,
        streams: Mapping[str, RngStream],
    ) -> Self:
        del streams
        if frequencies.ndim != 1:
            msg = f"frequencies must have rank 1, got shape {frequencies.shape}"
            raise ValueError(msg)
        return cls(
            displacements=LearnableParameter(jnp.zeros(features, dtype=jnp.float64)),
            frequencies=FixedParameter(frequencies),
        )

    def rotate(self, x: JaxComplexArray) -> JaxComplexArray:
        """Apply the elementwise value-space displacements."""
        n_frequencies = self.frequencies.value.shape[0]
        features = self.displacements.value.shape[0]
        expected = features * n_frequencies
        if x.shape[-1] != expected:
            msg = f"expected final input dimension {expected}, got {x.shape[-1]}"
            raise ValueError(msg)
        grouped = x.reshape(*x.shape[:-1], features, n_frequencies)
        rotations = jnp.exp(
            1j * self.displacements.value[:, jnp.newaxis] * self.frequencies.value[jnp.newaxis, :]
        )
        rotations = jnp.reshape(
            rotations,
            (1,) * (x.ndim - 1) + (features, n_frequencies),
        )
        return (grouped * rotations).reshape(*x.shape[:-1], expected)
