from __future__ import annotations

from collections.abc import Mapping
from typing import Self

import equinox as eqx
import jax.numpy as jnp
from jax.nn.initializers import variance_scaling
from tjax import JaxArray, JaxRealArray, RngStream

from cem.phasor.message import PhasorMessage
from cem.structure.graph import FixedParameter, LearnableParameter

# Each real/imaginary component uses Lecun variance (0.5 * 1/fan_in), giving correct
# complex Lecun initialization when the two components are combined.
_complex_lecun = variance_scaling(0.5, "fan_in", "truncated_normal")


class Linear(eqx.Module):
    """Log-domain linear transform followed by per-channel phase scaling.

    Encodes each input phasor as (log|z|, arg(z)), applies a complex linear mix,
    then decodes and scales the phase.  Output phase is a linear combination of
    input phases scaled by ``phase_scales``; output log-amplitude is a linear
    combination of input log-amplitudes.  This gives a multiplicative inductive
    bias (products of powers) rather than an additive one.

    Output amplitude is clipped to [0, 1] (log-amplitude <= 0), consistent with
    characteristic-function magnitudes.  The phase scaling corrects per-channel
    phase without disturbing the amplitude set by the log-domain mix.

    Attributes:
        weight: Complex weight matrix, shape (out_features, in_features).
        phase_scales: Real per-channel phase-scale factors, shape (out_features,).
    """

    weight: LearnableParameter[JaxArray]
    phase_scales: LearnableParameter[JaxRealArray]

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
        w_re = _complex_lecun(stream.key(), shape, jnp.float64)
        w_im = _complex_lecun(stream.key(), shape, jnp.float64)
        return cls(
            weight=LearnableParameter(w_re + 1j * w_im),
            phase_scales=LearnableParameter(jnp.ones(out_features, dtype=jnp.float64)),
        )

    def infer(
        self,
        z: JaxArray,
        *,
        eps: float = 1e-6,
        min_log_amp: float = -8.0,
    ) -> JaxArray:
        """Apply log-domain linear mix then phase scaling.

        Args:
            z: Input phasors, shape (..., in_features).
            eps: Added to amplitude before log to avoid log(0).
            min_log_amp: Floor on log-amplitude, applied to both input and output.

        Returns:
            Output phasors, shape (..., out_features).
        """
        log_amp = jnp.maximum(jnp.log(jnp.abs(z) + eps), min_log_amp)
        eta = log_amp + 1j * jnp.angle(z)
        u = eta @ self.weight.value.T
        out_amp = jnp.exp(jnp.clip(u.real, min_log_amp, 0.0))
        phase_scales = jnp.reshape(self.phase_scales.value, (1,) * (u.ndim - 1) + (-1,))
        return out_amp * jnp.exp(1j * phase_scales * u.imag)


class LinearWithDropout(Linear):
    """Log-domain linear transform with phase scaling and phasor dropout.

    Dropout is applied to the output after the transform and phase scaling.
    The default rate of 0.1 provides light regularization; set it to 0.0 to disable
    dropout entirely, or pass ``inference=True`` to :meth:`infer` to skip it at eval time.

    Attributes:
        dropout_rate: Scalar probability in [0, 1) of zeroing each output phasor.
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
        base = Linear.create(in_features, out_features, streams=streams)
        return cls(
            weight=base.weight,
            phase_scales=base.phase_scales,
            dropout_rate=FixedParameter(jnp.asarray(dropout_rate)),
        )

    def infer(self, z: JaxArray, *, streams: Mapping[str, RngStream], inference: bool) -> JaxArray:  # ty: ignore[invalid-method-override]
        """Apply log-domain linear mix, phase scaling, then phasor dropout.

        Args:
            z: Input phasors, shape (..., in_features).
            streams: RNG streams; the ``"inference"`` stream is used for dropout.
            inference: When ``True``, dropout is skipped.

        Returns:
            Output phasors, shape (..., out_features).
        """
        result = super().infer(z)
        if inference:
            return result
        return (
            PhasorMessage(result).dropout(streams["inference"].key(), self.dropout_rate.value).data
        )
