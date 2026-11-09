"""SIP confounding witness map: a bounded test of residuals against instrument context."""

from collections.abc import Mapping
from typing import Self

import equinox as eqx
import jax.numpy as jnp
from efax import NormalNP
from tjax import JaxRealArray, RngStream

from cem.perceptron.mlp import MLP


class WitnessMap(eqx.Module):
    """Map instrument messages to one bounded mean and one variance coefficient per channel.

    The witness must stay finite and small for certain instruments with tiny displacements, treat
    every missing instrument alike regardless of its mean, and stay neutral for an instrument
    context that supplies neither displacement nor spread. See `sip-confounding-witness` in the
    thesis.

    Each Gaussian instrument coordinate enters as the pair (u, r) with r = 1 / (1 + σ²) and
    u = r m. The perceptron output is centred at the all-certain-zero context, whose pairs all equal
    (0, 1), and passed through tanh.
    """

    perceptron: MLP

    @classmethod
    def create(
        cls,
        n: int,
        *,
        instrument_features: int,
        hidden_features: int | tuple[int, ...],
        streams: Mapping[str, RngStream],
    ) -> Self:
        """Build the witness map for `n` channels from `instrument_features` instrument inputs."""
        return cls(
            perceptron=MLP.create(
                in_features=2 * instrument_features,
                out_features=2 * n,
                hidden_features=hidden_features,
                streams=streams,
            )
        )

    def infer(
        self,
        instruments: NormalNP,
        *,
        streams: Mapping[str, RngStream],
        inference: bool,
    ) -> JaxRealArray:
        """Return the witness, shape (n, 2), for the concatenated instrument messages.

        The perceptron has no dropout, so subtracting its output at the reference is exact.
        """
        encoded = _encode(instruments)
        reference = jnp.concat([jnp.zeros_like(encoded[0]), jnp.ones_like(encoded[1])])
        batch = jnp.stack([jnp.concat(encoded), reference])
        output, reference_output = self.perceptron.infer(
            batch, streams=streams, inference=inference
        )
        return jnp.tanh(output - reference_output).reshape(-1, 2)


def _encode(instruments: NormalNP) -> tuple[JaxRealArray, JaxRealArray]:
    """Return (u, r) from natural parameters, giving (0, 0) exactly at zero precision.

    With precision κ, r = κ / (1 + κ) and u = (m κ) / (1 + κ). Both stay bounded as κ grows.
    """
    precision = -2.0 * instruments.negative_half_precision
    scale = 1.0 / (1.0 + precision)
    return scale * instruments.mean_times_precision, scale * precision
