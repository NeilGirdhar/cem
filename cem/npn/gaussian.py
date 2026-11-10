"""Gaussian natural-parameter networks with moment-matched activations."""

import itertools as it
import math
from collections.abc import Mapping
from typing import Self

import equinox as eqx
import jax.numpy as jnp
from efax import NormalNP, NormalPP, NormalVP
from jax.nn import initializers, softplus
from jax.scipy.special import ndtr
from tjax import JaxRealArray, RngStream

from cem.structure.graph import (
    LearnableParameter,
    LearnableParameterFactory,
    SummedGradientParameter,
)

_MINIMUM_VARIANCE = 1e-8
_STANDARD_NORMAL_DENSITY_SCALE = 1 / jnp.sqrt(2 * jnp.pi)
_LECUN_NORMAL = initializers.variance_scaling(1.0, "fan_in", "truncated_normal")
_UNIT_SOFTPLUS_RAW = math.log(math.expm1(1.0))  # softplus of this is one


def gaussian_relu(distribution: NormalVP) -> NormalVP:
    """Approximate a rectified Gaussian by matching its first two moments.

    Floor the resulting variance so the output remains a proper Gaussian.
    """
    mean, variance = distribution.mean, distribution.variance
    deviation = jnp.sqrt(variance)
    standardized_mean = mean / deviation
    cdf = ndtr(standardized_mean)
    density = _STANDARD_NORMAL_DENSITY_SCALE * jnp.exp(-0.5 * standardized_mean**2)
    output_mean = mean * cdf + deviation * density
    output_second_moment = (mean**2 + variance) * cdf + mean * deviation * density
    output_variance = jnp.maximum(
        output_second_moment - output_mean**2,
        _MINIMUM_VARIANCE,
    )
    return NormalVP(output_mean, output_variance)


class GaussianNPNLinear(eqx.Module):
    """Propagate the marginal moments of a factorized Gaussian through a learned affine map.

    The weights are deterministic and the bias is Gaussian. The bias variance lets the layer
    predict spread that its inputs do not carry, including when it has no inputs. See
    `natural-parameter-networks` in the thesis.
    """

    weight: LearnableParameter[JaxRealArray]
    bias: LearnableParameter[JaxRealArray]
    bias_raw_variance: LearnableParameter[JaxRealArray]  # softplus gives the bias variance

    @classmethod
    def create(
        cls,
        in_features: int,
        out_features: int,
        *,
        streams: Mapping[str, RngStream],
        parameter_factory: LearnableParameterFactory[JaxRealArray] = SummedGradientParameter,
    ) -> Self:
        stream = streams["parameters"]
        weight = _LECUN_NORMAL(
            stream.key(),
            (out_features, in_features),
            jnp.float64,
        )
        return cls(
            weight=parameter_factory(weight),
            bias=parameter_factory(jnp.zeros(out_features, dtype=jnp.float64)),
            bias_raw_variance=parameter_factory(
                jnp.full(out_features, _UNIT_SOFTPLUS_RAW, dtype=jnp.float64)
            ),
        )

    def infer(self, inputs: NormalVP) -> NormalVP:
        """Return a factorized Gaussian with the affine output's marginal means and variances.

        Supply inputs with finite variance. The factorized output retains marginal uncertainty and
        drops cross-covariances. The bias variance adds to the propagated input variance.
        """
        output_mean = inputs.mean @ self.weight.value.T + self.bias.value
        output_variance = inputs.variance @ jnp.square(self.weight.value).T + softplus(
            self.bias_raw_variance.value
        )
        return NormalVP(output_mean, jnp.maximum(output_variance, _MINIMUM_VARIANCE))


class GaussianNPNInputPool(eqx.Module):
    """Combine inputs with learned background evidence so absent inputs have defined moments."""

    dummy_mean: LearnableParameter[JaxRealArray]
    dummy_raw_presence: LearnableParameter[JaxRealArray]

    @classmethod
    def create(
        cls,
        n_features: int,
        *,
        parameter_factory: LearnableParameterFactory[JaxRealArray] = SummedGradientParameter,
    ) -> Self:
        """Initialize the background evidence to a standard normal for each input."""
        return cls(
            dummy_mean=parameter_factory(jnp.zeros(n_features, dtype=jnp.float64)),
            dummy_raw_presence=parameter_factory(
                jnp.full(n_features, _UNIT_SOFTPLUS_RAW, dtype=jnp.float64)
            ),
        )

    def infer(self, inputs: NormalNP) -> NormalNP:
        """Pool evidence by adding the background and input natural parameters."""
        dummy = NormalPP(self.dummy_mean.value, softplus(self.dummy_raw_presence.value)).to_nat()
        return NormalNP(
            mean_times_precision=inputs.mean_times_precision + dummy.mean_times_precision,
            negative_half_precision=(
                inputs.negative_half_precision + dummy.negative_half_precision
            ),
        )


class GaussianNPN(eqx.Module):
    """A feedforward network that propagates Gaussian messages and accepts missing inputs."""

    input_pool: GaussianNPNInputPool
    layers: tuple[GaussianNPNLinear, ...]

    @classmethod
    def create(
        cls,
        in_features: int,
        out_features: int,
        *,
        hidden_features: int | tuple[int, ...] = (),
        streams: Mapping[str, RngStream],
        parameter_factory: LearnableParameterFactory[JaxRealArray] = SummedGradientParameter,
    ) -> Self:
        """Build a Gaussian network with one layer per hidden width and an affine output.

        An empty tuple omits hidden layers. A single integer specifies one hidden layer.
        The factory wraps the input pool's parameters and each layer's weights and biases.
        """
        if isinstance(hidden_features, int):
            hidden_features = (hidden_features,)
        feature_sizes = (in_features, *hidden_features, out_features)
        return cls(
            input_pool=GaussianNPNInputPool.create(
                in_features, parameter_factory=parameter_factory
            ),
            layers=tuple(
                GaussianNPNLinear.create(
                    n_in, n_out, streams=streams, parameter_factory=parameter_factory
                )
                for n_in, n_out in it.pairwise(feature_sizes)
            ),
        )

    def infer(self, inputs: NormalNP) -> NormalNP:
        """Propagate Gaussian input messages through the network.

        Zero precision marks an absent input. Background evidence gives each input positive
        precision before moment conversion. The layers propagate means and variances, so the
        network converts only at its edges.
        """
        moments = self.input_pool.infer(inputs).to_variance_parametrization()
        for layer in self.layers[:-1]:
            moments = gaussian_relu(layer.infer(moments))
        return self.layers[-1].infer(moments).to_nat()
