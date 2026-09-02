from collections.abc import Mapping
from typing import Self

import equinox as eqx
import jax
import jax.numpy as jnp
from tjax import RngStream

from cem.perceptron import MLP
from cem.phasor.message import JaxComplexArray, phasor_to_real


class ValueProjection(eqx.Module):
    """Nonlinearly project phasor values while preserving their presences.

    A real MLP reads all real and imaginary components and proposes a new Cartesian
    direction for each phasor. The result is normalized per phasor and multiplied by
    the corresponding input magnitude. This lets values interact nonlinearly without
    creating or destroying evidence.

    Attributes:
        network: Real MLP from concatenated Cartesian input to Cartesian directions.
    """

    network: MLP

    @classmethod
    def create(
        cls,
        features: int,
        *,
        hidden_features: int | None = None,
        streams: Mapping[str, RngStream],
    ) -> Self:
        """Create a value projection over ``features`` phasors."""
        if hidden_features is None:
            hidden_features = features
        return cls(
            network=MLP.create(
                2 * features,
                2 * features,
                hidden_features=hidden_features,
                streams=streams,
            )
        )

    def infer(
        self, z: JaxComplexArray, *, streams: Mapping[str, RngStream], inference: bool
    ) -> JaxComplexArray:
        """Project phasor directions and restore each input magnitude."""
        cartesian = self.network.infer(phasor_to_real(z), streams=streams, inference=inference)
        real, imaginary = jnp.split(cartesian, 2, axis=-1)
        direction = jax.lax.complex(real, imaginary)
        direction_magnitude = jnp.abs(direction)
        safe_magnitude = jnp.where(
            direction_magnitude > 0,
            direction_magnitude,
            jnp.ones_like(direction_magnitude),
        )
        unit_direction = direction / safe_magnitude
        unit_direction = jnp.where(
            direction_magnitude > 0,
            unit_direction,
            jnp.ones_like(unit_direction),
        )
        return jnp.abs(z) * unit_direction
