from __future__ import annotations

from collections.abc import Mapping
from typing import Self

import equinox as eqx
import jax.numpy as jnp
from jax.nn.initializers import variance_scaling
from tjax import JaxArray, RngStream

from cem.structure.graph import LearnableParameter

# Each real/imaginary component uses Lecun variance (0.5 * 1/fan_in), giving correct
# complex Lecun initialization when the two components are combined.
_complex_lecun = variance_scaling(0.5, "fan_in", "truncated_normal")


class Linear(eqx.Module):
    """Complex-valued affine transform: z W^T + b.

    Attributes:
        weight: Complex weight matrix, shape (out_features, in_features).
        bias: Complex bias vector, shape (out_features,).
    """

    weight: LearnableParameter[JaxArray]
    bias: LearnableParameter[JaxArray]

    @classmethod
    def create(
        cls,
        in_features: int,
        out_features: int,
        *,
        streams: Mapping[str, RngStream],
    ) -> Self:
        stream = streams["parameters"]
        shape = (out_features, in_features)
        w_re = _complex_lecun(stream.key(), shape, jnp.float64)
        w_im = _complex_lecun(stream.key(), shape, jnp.float64)
        return cls(
            weight=LearnableParameter(w_re + 1j * w_im),
            bias=LearnableParameter(jnp.zeros(out_features, dtype=jnp.complex128)),
        )

    def infer(self, z: JaxArray) -> JaxArray:
        """Apply linear transform.

        Args:
            z: Input phasors, shape (..., in_features).

        Returns:
            Output phasors, shape (..., out_features).
        """
        y = z @ self.weight.value.T
        broadcasted_bias = jnp.reshape(self.bias.value, (1,) * (y.ndim - 1) + (-1,))
        return y + broadcasted_bias
