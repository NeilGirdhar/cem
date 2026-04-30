from __future__ import annotations

import itertools as it
from collections.abc import Mapping
from typing import Self

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
from tjax import JaxRealArray, RngStream

from cem.structure.graph import FixedParameter
from cem.transforms import Affine


class MLP(eqx.Module):
    """Real-valued multi-layer perceptron with optional hidden-layer dropout.

    Attributes:
        layers: Affine projections from input through hidden features to output.
        dropout_rate: Fraction of hidden activations zeroed after GELU. 0.0 disables.
    """

    layers: tuple[Affine, ...]
    dropout_rate: FixedParameter[JaxRealArray]

    @classmethod
    def create(
        cls,
        in_features: int,
        out_features: int,
        *,
        hidden_features: int | tuple[int, ...] = (),
        dropout_rate: float = 0.0,
        streams: Mapping[str, RngStream],
    ) -> Self:
        if isinstance(hidden_features, int):
            hidden_features = (hidden_features,)
        feature_sizes = (in_features, *hidden_features, out_features)
        return cls(
            layers=tuple(
                Affine.create(n_in, n_out, streams=streams)
                for n_in, n_out in it.pairwise(feature_sizes)
            ),
            dropout_rate=FixedParameter(jnp.asarray(dropout_rate)),
        )

    def infer(
        self, x: JaxRealArray, *, streams: Mapping[str, RngStream], inference: bool
    ) -> JaxRealArray:
        """Apply affine hidden layers with GELU activations, then a linear output layer.

        Args:
            x: Input, shape (..., in_features).
            streams: RNG streams; the ``"inference"`` stream is used for hidden dropout.
            inference: When ``True``, hidden dropout is skipped.

        Returns:
            Output, shape (..., out_features).
        """
        result = x
        for layer in self.layers[:-1]:
            result = jax.nn.gelu(layer.infer(result))
            if not inference:
                key = streams["inference"].key()
                keep_prob = 1.0 - self.dropout_rate.value
                mask = jr.bernoulli(key, keep_prob, shape=result.shape)
                result = jnp.where(mask, result / keep_prob, jnp.zeros_like(result))
        return self.layers[-1].infer(result)
