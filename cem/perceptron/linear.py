from __future__ import annotations

from collections.abc import Mapping
from typing import Self

import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
from jax.nn.initializers import variance_scaling
from tjax import JaxRealArray, RngStream

from cem.structure.graph import LearnableParameter

_lecun = variance_scaling(1.0, "fan_in", "truncated_normal")


class Linear(eqx.Module):
    """Real-valued affine transform: x W^T + b.

    Attributes:
        weight: Weight matrix, shape (out_features, in_features).
        bias: Bias vector, shape (out_features,).
    """

    weight: LearnableParameter[JaxRealArray]
    bias: LearnableParameter[JaxRealArray]

    @classmethod
    def create(
        cls,
        in_features: int,
        out_features: int,
        *,
        streams: Mapping[str, RngStream],
    ) -> Self:
        stream = streams["parameters"]
        w = _lecun(stream.key(), (out_features, in_features), jnp.float64)
        return cls(
            weight=LearnableParameter(w),
            bias=LearnableParameter(jnp.zeros(out_features, dtype=jnp.float64)),
        )

    def infer(self, x: JaxRealArray) -> JaxRealArray:
        """Apply affine transform.

        Args:
            x: Input, shape (..., in_features).

        Returns:
            Output, shape (..., out_features).
        """
        y = x @ self.weight.value.T
        broadcasted_bias = jnp.reshape(self.bias.value, (1,) * (y.ndim - 1) + (-1,))
        return y + broadcasted_bias


class LinearWithDropout(Linear):
    """Real-valued affine transform with dropout.

    Dropout is applied to the output after the affine transform.  Pass ``inference=True``
    to :meth:`infer` to skip it at eval time.

    Attributes:
        dropout_rate: Scalar probability in [0, 1) of zeroing each output unit.
    """

    dropout_rate: JaxRealArray

    @classmethod
    def create(
        cls,
        in_features: int,
        out_features: int,
        *,
        dropout_rate: float = 0.1,
        streams: Mapping[str, RngStream],
    ) -> Self:
        base = Linear.create(in_features, out_features, streams=streams)
        return cls(weight=base.weight, bias=base.bias, dropout_rate=jnp.asarray(dropout_rate))

    def infer(  # ty: ignore[invalid-method-override]
        self, x: JaxRealArray, *, streams: Mapping[str, RngStream], inference: bool
    ) -> JaxRealArray:
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
        mask = jr.bernoulli(key, 1.0 - self.dropout_rate, shape=result.shape)
        return jnp.where(mask, result / (1.0 - self.dropout_rate), jnp.zeros_like(result))
