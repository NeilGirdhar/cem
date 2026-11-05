"""Gaussian natural-parameter networks with moment-matched activations."""

import itertools as it
import math
from collections.abc import Mapping
from typing import Self

import equinox as eqx
import jax.numpy as jnp
from efax import NormalNP, NormalVP
from jax.nn import initializers, softplus
from jax.scipy.special import ndtr
from tjax import JaxRealArray, RngStream

from cem.structure.graph import LearnableParameter

_MINIMUM_VARIANCE = 1e-8
_STANDARD_NORMAL_DENSITY_SCALE = 1 / jnp.sqrt(2 * jnp.pi)
_LECUN_NORMAL = initializers.variance_scaling(1.0, "fan_in", "truncated_normal")
_UNIT_PRESENCE_RAW = math.log(math.expm1(1.0))


def normal_from_mean_and_presence(
    mean: JaxRealArray,
    presence: JaxRealArray,
) -> NormalNP:
    """Encode a Gaussian using its mean and nonnegative precision.

    A finite placeholder mean with zero precision gives an absent message.
    """
    return NormalNP(
        mean_times_precision=mean * presence,
        negative_half_precision=-0.5 * presence,
    )


def _mean_and_variance(distribution: NormalNP) -> tuple[JaxRealArray, JaxRealArray]:
    presence = -2 * distribution.negative_half_precision
    mean = distribution.mean_times_precision / presence
    variance = 1 / presence
    return mean, variance


def gaussian_relu(distribution: NormalNP) -> NormalNP:
    """Approximate a rectified Gaussian by matching its first two moments.

    Floor the resulting variance so the output remains a proper Gaussian.
    """
    mean, variance = _mean_and_variance(distribution)
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
    return NormalVP(output_mean, output_variance).to_nat()


class GaussianNPNLinear(eqx.Module):
    """Propagate the marginal moments of a factorized Gaussian through a learned affine map."""

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
        weight = _LECUN_NORMAL(
            stream.key(),
            (out_features, in_features),
            jnp.float64,
        )
        return cls(
            weight=LearnableParameter(weight),
            bias=LearnableParameter(jnp.zeros(out_features, dtype=jnp.float64)),
        )

    def infer(self, inputs: NormalNP) -> NormalNP:
        """Return a factorized Gaussian with the affine output's marginal means and variances.

        Supply inputs with positive precision so their moments are defined. The
        factorized output retains marginal uncertainty and drops cross-covariances.
        """
        input_mean, input_variance = _mean_and_variance(inputs)
        output_mean = input_mean @ self.weight.value.T + self.bias.value
        output_variance = input_variance @ jnp.square(self.weight.value).T
        return NormalVP(
            output_mean,
            jnp.maximum(output_variance, _MINIMUM_VARIANCE),
        ).to_nat()


class GaussianNPNInputPool(eqx.Module):
    """Combine inputs with learned background evidence so absent inputs have defined moments."""

    dummy_mean: LearnableParameter[JaxRealArray]
    dummy_raw_presence: LearnableParameter[JaxRealArray]

    @classmethod
    def create(cls, n_features: int) -> Self:
        """Initialize the background evidence to a standard normal for each input."""
        return cls(
            dummy_mean=LearnableParameter(jnp.zeros(n_features, dtype=jnp.float64)),
            dummy_raw_presence=LearnableParameter(
                jnp.full(n_features, _UNIT_PRESENCE_RAW, dtype=jnp.float64)
            ),
        )

    def infer(self, inputs: NormalNP) -> NormalNP:
        """Pool evidence by adding the background and input natural parameters."""
        dummy = normal_from_mean_and_presence(
            self.dummy_mean.value,
            softplus(self.dummy_raw_presence.value),
        )
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
    ) -> Self:
        """Build a Gaussian network with one layer per hidden width and an affine output.

        An empty tuple omits hidden layers. A single integer specifies one hidden layer.
        """
        if isinstance(hidden_features, int):
            hidden_features = (hidden_features,)
        feature_sizes = (in_features, *hidden_features, out_features)
        return cls(
            input_pool=GaussianNPNInputPool.create(in_features),
            layers=tuple(
                GaussianNPNLinear.create(n_in, n_out, streams=streams)
                for n_in, n_out in it.pairwise(feature_sizes)
            ),
        )

    def infer(
        self,
        values: JaxRealArray,
        presences: JaxRealArray | None = None,
    ) -> NormalNP:
        """Propagate Gaussian messages from feature values and their precisions.

        Omit presences for unit precision. Zero presence marks an absent input.
        Background evidence gives each input positive precision before moment conversion.
        """
        if presences is None:
            presences = jnp.ones_like(values)
        result = self.input_pool.infer(normal_from_mean_and_presence(values, presences))
        for layer in self.layers[:-1]:
            result = gaussian_relu(layer.infer(result))
        return self.layers[-1].infer(result)
