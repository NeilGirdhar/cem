"""A real-valued feedforward network for point-valued feature vectors."""

import itertools as it
from collections.abc import Mapping
from typing import Self

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
from tjax import JaxRealArray, RngStream

from cem.structure.graph import (
    FixedParameter,
    LearnableParameterFactory,
    SummedGradientParameter,
)
from cem.transforms import Affine


class MLP(eqx.Module):
    """A multilayer perceptron with optional hidden-layer dropout."""

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
        parameter_factory: LearnableParameterFactory[JaxRealArray] = SummedGradientParameter,
    ) -> Self:
        """Build an MLP from the input, hidden, and output widths.

        An empty tuple omits hidden layers. A single integer specifies one hidden layer.
        The parameter factory wraps each layer's weights and biases.
        """
        if isinstance(hidden_features, int):
            hidden_features = (hidden_features,)
        feature_sizes = (in_features, *hidden_features, out_features)
        return cls(
            layers=tuple(
                Affine.create(
                    n_in,
                    n_out,
                    complex_matrix=False,
                    streams=streams,
                    parameter_factory=parameter_factory,
                )
                for n_in, n_out in it.pairwise(feature_sizes)
            ),
            dropout_rate=FixedParameter(jnp.asarray(dropout_rate)),
        )

    def infer(
        self, x: JaxRealArray, *, streams: Mapping[str, RngStream], inference: bool
    ) -> JaxRealArray:
        """Map the last input axis to the output width, preserving leading batch axes.

        Apply GELU to hidden layers and leave the output linear. In training mode,
        use the inference random stream for hidden dropout.
        """
        result = x
        for layer in self.layers[:-1]:
            result = jax.nn.gelu(layer.project(result))
            if not inference:
                key = streams["inference"].key()
                keep_prob = 1.0 - self.dropout_rate.value
                mask = jr.bernoulli(key, keep_prob, shape=result.shape)
                result = jnp.where(mask, result / keep_prob, jnp.zeros_like(result))
        return self.layers[-1].project(result)
