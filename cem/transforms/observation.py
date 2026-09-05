from typing import Self

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from efax import Flattener, UnitVarianceNormalNP
from jax.lax import stop_gradient
from tjax import JaxArray, JaxRealArray

from cem.structure.graph import FixedParameter, MetaParameter, Parameter

_SEMICIRCLE_LIMIT = jnp.pi / 2


class ArctangentPhaseMap(eqx.Module):
    """Map each real feature to the open semicircle with a positive scale.

    The map ``atan(x / scale)`` compresses the real line polynomially near the
    phase boundaries. Each feature has its own scale, initialized to one.

    Attributes:
        log_scales: Logarithms of the positive feature scales.
    """

    log_scales: Parameter[JaxRealArray]

    @classmethod
    def create_learned(cls, features: int) -> Self:
        """Create a learnable, initially unit-scaled phase map."""
        return cls(log_scales=MetaParameter(jnp.zeros(features, dtype=jnp.float64)))

    @classmethod
    def create_fixed(cls, features: int) -> Self:
        """Create a fixed unit-scaled phase map."""
        return cls(log_scales=FixedParameter(jnp.zeros(features, dtype=jnp.float64)))

    def phase(self, values: JaxRealArray) -> JaxRealArray:
        """Map real feature values into phases in ``(-pi / 2, pi / 2)``."""
        return jnp.atan(values / jnp.exp(self.log_scales.value))

    def fisher_equalization_loss(self, values: JaxRealArray) -> JaxRealArray:
        """Measure variation in the phase map's local Fisher metric.

        The metric is the squared derivative of phase with respect to the input. A
        small scale penalty prevents the learned global scale from drifting while
        the metric is equalized.
        """
        scale = jnp.exp(self.log_scales.value)
        derivative = scale / (jnp.square(scale) + jnp.square(values))
        log_metric = jnp.log(jnp.square(derivative) + jnp.finfo(values.dtype).eps)
        equalization = jnp.mean(
            jnp.square(log_metric - jnp.mean(log_metric, axis=0, keepdims=True))
        )
        scale_penalty = 0.01 * jnp.mean(jnp.square(self.log_scales.value))
        return equalization + scale_penalty

    def encode(
        self,
        presences: JaxRealArray,
        values: JaxRealArray,
    ) -> JaxArray:
        """Encode feature presences and values as evidence phasors."""
        return presences * jnp.exp(1j * self.phase(values))

    def encode_with_reversed_phase_gradient(
        self,
        presences: JaxRealArray,
        values: JaxRealArray,
    ) -> JaxArray:
        """Encode phasors while reversing gradients into the phase map."""
        phase = self.phase(values)
        reversed_phase = 2 * stop_gradient(phase) - phase
        return presences * jnp.exp(1j * reversed_phase)

    def decode(self, phasors: JaxArray) -> JaxRealArray:
        """Decode phasors whose phases lie in the open right semicircle."""
        epsilon = jnp.finfo(phasors.real.dtype).eps
        phases = jnp.clip(
            jnp.angle(phasors),
            -_SEMICIRCLE_LIMIT + epsilon,
            _SEMICIRCLE_LIMIT - epsilon,
        )
        return jnp.exp(self.log_scales.value) * jnp.tan(phases)


def semicircle_observation_phase(values: JaxRealArray) -> JaxRealArray:
    """Map real observations monotonically into the open phase semicircle."""
    return _SEMICIRCLE_LIMIT * jnp.tanh(values)


def inverse_semicircle_observation_phase(phases: JaxRealArray) -> JaxRealArray:
    """Invert :func:`semicircle_observation_phase` within its open range."""
    epsilon = jnp.finfo(phases.dtype).eps
    normalized = jnp.clip(phases / _SEMICIRCLE_LIMIT, -1 + epsilon, 1 - epsilon)
    return jnp.arctanh(normalized)


def encode_observation_phasors(
    presences: JaxRealArray,
    values: JaxRealArray,
) -> JaxArray:
    """Encode scalar presences and values with the temporary semicircle map."""
    return presences * jnp.exp(1j * semicircle_observation_phase(values))


def decode_observation_phasors(phasors: JaxArray) -> JaxRealArray:
    """Decode values from phasor phases under the temporary semicircle map."""
    return inverse_semicircle_observation_phase(jnp.angle(phasors))


def encode_flat(values: JaxRealArray) -> JaxRealArray:
    """Encode a real vector as flat natural params of UnitVarianceNormalNP.

    Each component x_i is encoded as UnitVarianceNormalNP(x_i) (unit variance),
    then flattened with ``mapped_to_plane=True``.  The resulting array has shape
    ``(n,)`` for input of shape ``(n,)``.

    Args:
        values: Shape ``(n,)`` real vector.

    Returns:
        Shape ``(n,)`` flat encoding.
    """
    assert values.ndim == 1
    dist = UnitVarianceNormalNP(values)
    _, flat = Flattener.flatten(dist, mapped_to_plane=True)
    return flat.reshape(-1)


def standardize_columns(values: np.ndarray) -> np.ndarray:
    """Center and scale each numeric column to unit variance."""
    values = np.asarray(values, dtype=np.float64)
    mean = values.mean(axis=0)
    std = values.std(axis=0)
    std = np.where(std == 0.0, 1.0, std)
    return ((values - mean) / std).astype(np.float64)
