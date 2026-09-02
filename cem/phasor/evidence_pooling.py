from collections.abc import Mapping
from typing import Self

import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
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


class FrequencyAdaptedEvidencePooling(eqx.Module):
    """Pool corresponding features coherently across a frequency grid.

    Each edge learns a positive evidence gain and a displacement in value space.
    At angular frequency ``omega``, the displacement becomes the rotation
    ``exp(1j * omega * displacement)``. Thus the complex weights vary across
    frequencies while retaining one value-space meaning.

    Inputs and outputs use feature-major flattened layouts: the final axis is
    ``features * frequencies``, with all frequencies for one feature adjacent.

    Attributes:
        log_gains: Log evidence gains, shape (out_features, in_features).
        displacements: Value-space displacements, same shape.
        frequencies: Angular frequencies, shape (n_frequencies,).
    """

    log_gains: LearnableParameter[JaxRealArray]
    displacements: LearnableParameter[JaxRealArray]
    frequencies: FixedParameter[JaxRealArray]

    @classmethod
    def create(
        cls,
        in_features: int,
        out_features: int,
        frequencies: JaxRealArray,
        *,
        streams: Mapping[str, RngStream],
    ) -> Self:
        if frequencies.ndim != 1:
            msg = f"frequencies must have rank 1, got shape {frequencies.shape}"
            raise ValueError(msg)
        shape = (out_features, in_features)
        stream = streams["parameters"]
        max_frequency = jnp.max(jnp.abs(frequencies))
        displacement_scale = jnp.pi / jnp.where(max_frequency > 0, max_frequency, 1)
        return cls(
            log_gains=LearnableParameter(
                -jnp.log(in_features) + 0.05 * jr.normal(stream.key(), shape, dtype=jnp.float64)
            ),
            displacements=LearnableParameter(
                jr.uniform(
                    stream.key(),
                    shape,
                    dtype=jnp.float64,
                    minval=-displacement_scale,
                    maxval=displacement_scale,
                )
            ),
            frequencies=FixedParameter(frequencies),
        )

    def project(self, x: JaxArray) -> JaxArray:
        """Apply frequency-adapted evidence pooling."""
        n_frequencies = self.frequencies.value.shape[0]
        in_features = self.log_gains.value.shape[1]
        expected = in_features * n_frequencies
        if x.shape[-1] != expected:
            msg = f"expected final input dimension {expected}, got {x.shape[-1]}"
            raise ValueError(msg)
        grouped = x.reshape(*x.shape[:-1], in_features, n_frequencies)
        weights = jnp.exp(self.log_gains.value)[..., jnp.newaxis] * jnp.exp(
            1j
            * self.displacements.value[..., jnp.newaxis]
            * self.frequencies.value[jnp.newaxis, jnp.newaxis, :]
        )
        result = jnp.einsum("...if,oif->...of", grouped, weights)
        return result.reshape(*result.shape[:-2], -1)


class LowRankFrequencyAdaptedEvidencePooling(eqx.Module):
    """Frequency-adapted evidence pooling with low-rank edge parameters.

    The dense log-gain and displacement matrices are generated from separate
    low-rank factors. This retains the value-space interpretation of each edge
    while matching the parameterization used by low-rank Möbius banks.
    """

    log_gain_output: LearnableParameter[JaxRealArray]
    log_gain_input: LearnableParameter[JaxRealArray]
    displacement_output: LearnableParameter[JaxRealArray]
    displacement_input: LearnableParameter[JaxRealArray]
    frequencies: FixedParameter[JaxRealArray]

    @classmethod
    def create(
        cls,
        in_features: int,
        out_features: int,
        rank: int,
        frequencies: JaxRealArray,
        *,
        streams: Mapping[str, RngStream],
    ) -> Self:
        if frequencies.ndim != 1:
            msg = f"frequencies must have rank 1, got shape {frequencies.shape}"
            raise ValueError(msg)
        if rank < 1 or rank > min(in_features, out_features):
            msg = f"rank must lie in [1, {min(in_features, out_features)}], got {rank}"
            raise ValueError(msg)

        stream = streams["parameters"]
        factor_noise = 0.05
        log_gain_output = factor_noise * jr.normal(
            stream.key(), (out_features, rank), dtype=jnp.float64
        )
        log_gain_input = factor_noise * jr.normal(
            stream.key(), (rank, in_features), dtype=jnp.float64
        )
        log_gain_output = log_gain_output.at[:, 0].add(1)
        log_gain_input = log_gain_input.at[0, :].add(-jnp.log(in_features))

        max_frequency = jnp.max(jnp.abs(frequencies))
        displacement_scale = jnp.pi / jnp.where(max_frequency > 0, max_frequency, 1)
        displacement_output = factor_noise * jr.normal(
            stream.key(), (out_features, rank), dtype=jnp.float64
        )
        displacement_input = factor_noise * jr.normal(
            stream.key(), (rank, in_features), dtype=jnp.float64
        )
        displacement_output = displacement_output.at[:, 0].add(1)
        displacement_input = displacement_input.at[0, :].add(
            jr.uniform(
                stream.key(),
                (in_features,),
                dtype=jnp.float64,
                minval=-displacement_scale,
                maxval=displacement_scale,
            )
        )
        return cls(
            log_gain_output=LearnableParameter(log_gain_output),
            log_gain_input=LearnableParameter(log_gain_input),
            displacement_output=LearnableParameter(displacement_output),
            displacement_input=LearnableParameter(displacement_input),
            frequencies=FixedParameter(frequencies),
        )

    def project(self, x: JaxArray) -> JaxArray:
        """Apply the generated frequency-adapted pooling matrix."""
        log_gains = self.log_gain_output.value @ self.log_gain_input.value
        displacements = self.displacement_output.value @ self.displacement_input.value
        n_frequencies = self.frequencies.value.shape[0]
        in_features = log_gains.shape[1]
        expected = in_features * n_frequencies
        if x.shape[-1] != expected:
            msg = f"expected final input dimension {expected}, got {x.shape[-1]}"
            raise ValueError(msg)
        grouped = x.reshape(*x.shape[:-1], in_features, n_frequencies)
        weights = jnp.exp(log_gains)[..., jnp.newaxis] * jnp.exp(
            1j
            * displacements[..., jnp.newaxis]
            * self.frequencies.value[jnp.newaxis, jnp.newaxis, :]
        )
        result = jnp.einsum("...if,oif->...of", grouped, weights)
        return result.reshape(*result.shape[:-2], -1)


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
