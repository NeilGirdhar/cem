from __future__ import annotations

from collections.abc import Mapping
from typing import Self

import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from jax.nn.initializers import variance_scaling
from tjax import JaxArray, JaxRealArray, RngStream

from cem.structure.graph import FixedParameter, LearnableParameter

_real_lecun = variance_scaling(1.0, "fan_in", "truncated_normal")

# Each real/imaginary component uses Lecun variance (0.5 * 1/fan_in), giving correct
# complex Lecun initialization when the two components are combined.
_complex_lecun = variance_scaling(0.5, "fan_in", "truncated_normal")


def _init_weight(
    shape: tuple[int, int],
    *,
    complex_matrix: bool,
    stream: RngStream,
) -> JaxArray:
    if not complex_matrix:
        return _real_lecun(stream.key(), shape, jnp.float64)
    w_re = _complex_lecun(stream.key(), shape, jnp.float64)
    w_im = _complex_lecun(stream.key(), shape, jnp.float64)
    return w_re + 1j * w_im


def _bias_dtype(*, complex_matrix: bool) -> np.dtype:
    if complex_matrix:
        return jnp.complex128
    return jnp.float64


class Affine(eqx.Module):
    """Affine transform: x W^T + b.

    Attributes:
        weight: Weight matrix, shape (out_features, in_features).
        bias: Bias vector, shape (out_features,).
    """

    weight: LearnableParameter[JaxArray]
    bias: LearnableParameter[JaxArray]

    @classmethod
    def create(
        cls,
        in_features: int,
        out_features: int,
        *,
        complex_matrix: bool = False,
        streams: Mapping[str, RngStream],
    ) -> Self:
        stream = streams["parameters"]
        shape = (out_features, in_features)
        return cls(
            weight=LearnableParameter(
                _init_weight(shape, complex_matrix=complex_matrix, stream=stream)
            ),
            bias=LearnableParameter(
                jnp.zeros(out_features, dtype=_bias_dtype(complex_matrix=complex_matrix))
            ),
        )

    def infer(self, x: JaxArray) -> JaxArray:
        """Apply affine transform.

        Args:
            x: Input, shape (..., in_features).

        Returns:
            Output, shape (..., out_features).
        """
        y = x @ self.weight.value.T
        broadcasted_bias = jnp.reshape(self.bias.value, (1,) * (y.ndim - 1) + (-1,))
        return y + broadcasted_bias


class AffineWithDropout(Affine):
    """Affine transform with dropout.

    Dropout is applied to the output after the affine transform.  Pass ``inference=True``
    to :meth:`infer` to skip it at eval time.

    Attributes:
        dropout_rate: Scalar probability in [0, 1) of zeroing each output unit.
    """

    dropout_rate: FixedParameter[JaxRealArray]

    @classmethod
    def create(
        cls,
        in_features: int,
        out_features: int,
        *,
        complex_matrix: bool = False,
        dropout_rate: float = 0.1,
        streams: Mapping[str, RngStream],
    ) -> Self:
        base = Affine.create(
            in_features, out_features, complex_matrix=complex_matrix, streams=streams
        )
        return cls(
            weight=base.weight,
            bias=base.bias,
            dropout_rate=FixedParameter(jnp.asarray(dropout_rate)),
        )

    def infer(  # ty: ignore[invalid-method-override]
        self, x: JaxArray, *, streams: Mapping[str, RngStream], inference: bool
    ) -> JaxArray:
        """Apply affine transform followed by dropout.

        Args:
            x: Input, shape (..., in_features).
            streams: RNG streams; the ``"inference"`` stream is used for dropout.
            inference: When ``True``, dropout is skipped.

        Returns:
            Output, shape (..., out_features).
        """
        result = super().infer(x)
        if inference:
            return result
        key = streams["inference"].key()
        mask = jr.bernoulli(key, 1.0 - self.dropout_rate.value, shape=result.shape)
        return jnp.where(mask, result / (1.0 - self.dropout_rate.value), jnp.zeros_like(result))
