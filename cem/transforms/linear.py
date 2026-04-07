from __future__ import annotations

from collections.abc import Mapping
from typing import Self

import equinox as eqx
import jax.numpy as jnp
from jax.nn.initializers import variance_scaling
from tjax import JaxArray, JaxRealArray, RngStream

from cem.phasor_calculus.message import PhasorMessage
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
        """Apply affine transform.

        Args:
            z: Input phasors, shape (..., in_features).

        Returns:
            Output phasors, shape (..., out_features).
        """
        y = z @ self.weight.value.T
        broadcasted_bias = jnp.reshape(self.bias.value, (1,) * (y.ndim - 1) + (-1,))
        return y + broadcasted_bias


class LinearWithDropout(Linear):
    """Complex-valued affine transform with phasor dropout.

    Dropout is applied to the output after the affine transform.  The default rate of 0.1
    provides light regularization; set it to 0.0 to disable.  Call
    ``eqx.nn.inference_mode(model, True)`` to disable dropout without recompilation.

    Attributes:
        dropout_rate: Scalar probability in [0, 1) of zeroing each output phasor.
        inference: When True, dropout is skipped entirely.
    """

    dropout_rate: JaxRealArray
    inference: bool

    @classmethod
    def create(  # type: ignore[override]
        cls,
        in_features: int,
        out_features: int,
        *,
        dropout_rate: float = 0.1,
        streams: Mapping[str, RngStream],
    ) -> Self:
        stream = streams["parameters"]
        shape = (out_features, in_features)
        w_re = _complex_lecun(stream.key(), shape, jnp.float64)
        w_im = _complex_lecun(stream.key(), shape, jnp.float64)
        return cls(
            weight=LearnableParameter(w_re + 1j * w_im),
            bias=LearnableParameter(jnp.zeros(out_features, dtype=jnp.complex128)),
            dropout_rate=jnp.asarray(dropout_rate),
            inference=False,
        )

    def infer(self, z: JaxArray, *, streams: Mapping[str, RngStream]) -> JaxArray:  # type: ignore[override]  # ty: ignore[invalid-method-override]
        """Apply affine transform followed by phasor dropout.

        Args:
            z: Input phasors, shape (..., in_features).
            streams: RNG streams; the ``"inference"`` stream is used for dropout.

        Returns:
            Output phasors, shape (..., out_features).
        """
        result = super().infer(z)
        dropped = PhasorMessage(result).dropout(streams["inference"].key(), self.dropout_rate).data
        return jnp.where(self.inference, result, dropped)
