from collections.abc import Mapping
from typing import Self

import equinox as eqx
import jax.numpy as jnp
from jax.nn.initializers import variance_scaling
from tjax import JaxArray, JaxRealArray, RngStream

from cem.structure.graph import FixedParameter, LearnableParameter
from cem.transforms.dropout import apply_dropout_if_training

# Each real/imaginary component uses Lecun variance (0.5 * 1/fan_in), giving correct
# complex Lecun initialization when the two components are combined.
_complex_lecun = variance_scaling(0.5, "fan_in", "truncated_normal")


def _init_weight(shape: tuple[int, int], *, stream: RngStream) -> JaxArray:
    w_re = _complex_lecun(stream.key(), shape, jnp.float64)
    w_im = _complex_lecun(stream.key(), shape, jnp.float64)
    return w_re + 1j * w_im


class EvidencePooling(eqx.Module):
    """Evidence pooling link: a bias-free linear projection mapping a source phasor vector.

    Maps into a target phasor vector, f(z) = W z. Each complex weight rotates and scales one
    source component into a target component, and contributions from different sources add as
    natural parameters. With no bias, absent input produces absent output.

    Attributes:
        weight: Weight matrix, shape (out_features, in_features).
    """

    weight: LearnableParameter[JaxArray]

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
        return cls(weight=LearnableParameter(_init_weight(shape, stream=stream)))

    def project(self, x: JaxArray) -> JaxArray:
        """Apply the evidence pooling link.

        Args:
            x: Input, shape (..., in_features).

        Returns:
            Output, shape (..., out_features).
        """
        return x @ self.weight.value.T


class EvidencePoolingWithDropout(EvidencePooling):
    """Evidence pooling link with dropout.

    Dropout is applied to the output after the projection.  Pass ``inference=True``
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
        dropout_rate: float = 0.1,
        streams: Mapping[str, RngStream],
    ) -> Self:
        base = EvidencePooling.create(in_features, out_features, streams=streams)
        return cls(
            weight=base.weight,
            dropout_rate=FixedParameter(jnp.asarray(dropout_rate)),
        )

    def infer(self, x: JaxArray, *, streams: Mapping[str, RngStream], inference: bool) -> JaxArray:
        """Apply the evidence pooling link followed by dropout.

        Args:
            x: Input, shape (..., in_features).
            streams: RNG streams; the ``"inference"`` stream is used for dropout.
            inference: When ``True``, dropout is skipped.

        Returns:
            Output, shape (..., out_features).
        """
        result = self.project(x)
        return apply_dropout_if_training(
            result, streams=streams, inference=inference, dropout_rate=self.dropout_rate.value
        )
