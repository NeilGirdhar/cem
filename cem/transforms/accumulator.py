from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
from tjax import JaxArray

from .gate import phasor_gate


class Accumulator(eqx.Module):
    """Persistent phasor state updated by gated decay and evidence accumulation.

    z' = gate(d, z) + u

    where d is the decay coefficient, z is the current state, and u is the evidence increment.
    State is managed explicitly: call init() to get the initial state, then thread the returned
    state through successive calls.

    Attributes:
        features: Dimension of the accumulated phasor state.
    """

    features: int = eqx.field(static=True)

    def init(self) -> JaxArray:
        """Return initial zero state, shape (features,)."""
        return jnp.zeros(self.features, dtype=jnp.complex128)

    def infer(self, state: JaxArray, decay: JaxArray, increment: JaxArray) -> JaxArray:
        """Update accumulator state.

        Args:
            state: Current phasor state, shape (..., features).
            decay: Decay coefficient, shape (..., features). Re(decay) controls gate strength.
            increment: Evidence increment to add, shape (..., features).

        Returns:
            New state z' = gate(decay, state) + increment, shape (..., features).
        """
        return phasor_gate(decay, state) + increment
