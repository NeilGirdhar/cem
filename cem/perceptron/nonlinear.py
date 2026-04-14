from __future__ import annotations

from collections.abc import Mapping
from typing import Self

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
from tjax import JaxRealArray, RngStream

from cem.perceptron.linear import Linear
from cem.structure.graph import FixedParameter, LearnableParameter


class LayerNorm(eqx.Module):
    """Layer normalisation with learnable affine parameters.

    Normalises across the last axis, then applies an elementwise scale and bias.

    Attributes:
        scale: Learnable scale, shape (features,). Initialised to ones.
        bias: Learnable bias, shape (features,). Initialised to zeros.
        eps: Small constant for numerical stability.
    """

    scale: LearnableParameter[JaxRealArray]
    bias: LearnableParameter[JaxRealArray]
    eps: FixedParameter[float]

    @classmethod
    def create(cls, features: int, *, eps: float = 1e-5) -> Self:
        return cls(
            scale=LearnableParameter(jnp.ones(features, dtype=jnp.float64)),
            bias=LearnableParameter(jnp.zeros(features, dtype=jnp.float64)),
            eps=FixedParameter(eps),
        )

    def infer(self, x: JaxRealArray) -> JaxRealArray:
        """Normalise x over the last axis and apply affine transform.

        Args:
            x: Input, shape (..., features).

        Returns:
            Normalised output, same shape as x.
        """
        mean = jnp.mean(x, axis=-1, keepdims=True)
        var = jnp.var(x, axis=-1, keepdims=True)
        x_norm = (x - mean) / jnp.sqrt(var + self.eps.value)
        return self.scale.value * x_norm + self.bias.value


class Nonlinear(eqx.Module):
    """GLU-style nonlinear projection followed by layer normalisation, with optional dropout.

    h(x) = d(ln(f3(sigmoid(f1(x)) * f2(x))))

    where f1, f2, f3 are linear links, the sigmoid gate controls information flow,
    ln is layer normalisation, and d is an optional dropout.  Pass ``inference=True``
    to :meth:`infer` to skip dropout at eval time.

    Attributes:
        f1: Gate signal projection, in_features → mid_features.
        f2: Content projection, in_features → mid_features.
        f3: Output projection, mid_features → out_features.
        layer_norm: Layer normalisation applied to the output.
        dropout_rate: Fraction of outputs zeroed after layer normalisation. 0.0 disables.
    """

    f1: Linear
    f2: Linear
    f3: Linear
    layer_norm: LayerNorm
    dropout_rate: FixedParameter[JaxRealArray]

    @classmethod
    def create(
        cls,
        in_features: int,
        out_features: int,
        *,
        mid_features: int | None = None,
        dropout_rate: float = 0.0,
        streams: Mapping[str, RngStream],
    ) -> Self:
        if mid_features is None:
            mid_features = out_features
        return cls(
            f1=Linear.create(in_features, mid_features, streams=streams),
            f2=Linear.create(in_features, mid_features, streams=streams),
            f3=Linear.create(mid_features, out_features, streams=streams),
            layer_norm=LayerNorm.create(out_features),
            dropout_rate=FixedParameter(jnp.asarray(dropout_rate)),
        )

    def infer(
        self, x: JaxRealArray, *, streams: Mapping[str, RngStream], inference: bool
    ) -> JaxRealArray:
        """Apply GLU-style nonlinear transform with layer normalisation and optional dropout.

        Args:
            x: Input, shape (..., in_features).
            streams: RNG streams; the ``"inference"`` stream is used for dropout.
            inference: When ``True``, dropout is skipped.

        Returns:
            Output, shape (..., out_features).
        """
        gated = jax.nn.sigmoid(self.f1.infer(x)) * self.f2.infer(x)
        result = self.layer_norm.infer(self.f3.infer(gated))
        if inference:
            return result
        key = streams["inference"].key()
        mask = jr.bernoulli(key, 1.0 - self.dropout_rate.value, shape=result.shape)
        return jnp.where(mask, result / (1.0 - self.dropout_rate.value), jnp.zeros_like(result))
