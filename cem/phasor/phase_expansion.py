from typing import Self

import equinox as eqx
import jax.numpy as jnp

from cem.phasor.message import JaxComplexArray
from cem.structure.graph import LearnableParameter


class PhaseExpansion(eqx.Module):
    """Expand each phasor into copies with learned Möbius phase stretches.

    Each output copy preserves the source presence while a Möbius transformation
    stretches its phase towards a learned direction. The expansion introduces
    nonlinear value features without pooling evidence from different inputs or
    introducing a discontinuity at the phase branch cut.

    Attributes:
        phase_stretches: Unconstrained complex stretch parameters, shape
            (in_features, expansion_factor). Each parameter's direction selects the
            attracting phase. Its magnitude controls the stretch after mapping the
            parameter into the open unit disk. Zero applies the identity transform.
    """

    phase_stretches: LearnableParameter[JaxComplexArray]

    @classmethod
    def create(cls, in_features: int, expansion_factor: int = 2) -> Self:
        """Create a phase expansion initialized to the identity transform."""
        if expansion_factor < 1:
            msg = f"expansion_factor must be positive, got {expansion_factor}"
            raise ValueError(msg)
        return cls(
            phase_stretches=LearnableParameter(
                jnp.zeros((in_features, expansion_factor), dtype=jnp.complex128)
            )
        )

    def expand(self, x: JaxComplexArray) -> JaxComplexArray:
        """Return the flattened phase-stretched copies of ``x``."""
        magnitude = jnp.abs(x)
        unit = x / jnp.where(magnitude > 0, magnitude, 1)
        parameter_shape = (1,) * (unit.ndim - 1) + self.phase_stretches.value.shape
        raw_stretch = jnp.reshape(self.phase_stretches.value, parameter_shape)
        stretch = raw_stretch / jnp.sqrt(1 + jnp.abs(raw_stretch) ** 2)
        unit = unit[..., :, jnp.newaxis]
        transformed = (unit + stretch) / (1 + jnp.conj(stretch) * unit)
        expanded = magnitude[..., :, jnp.newaxis] * transformed
        return expanded.reshape((*expanded.shape[:-2], -1))
