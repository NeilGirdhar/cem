from __future__ import annotations

import jax.numpy as jnp
from jax.nn import sigmoid
from tjax import JaxArray


def phasor_gate(gate_signal: JaxArray, z: JaxArray) -> JaxArray:
    """Scale each component of z by logistic(Re(gate_signal)).

    g(a, z) = logistic(Re(a)) ⊙ z

    The gate suppresses signals in any direction — when gate_signal is produced by a complex linear
    transform, Re(gate_signal) captures alignment with whatever direction that transform rotates
    onto the real axis.

    Args:
        gate_signal: Complex gate signal, shape (..., m).
        z: Content phasors to gate, shape (..., m).

    Returns:
        Gated phasors, shape (..., m).
    """
    return sigmoid(jnp.real(gate_signal)) * z


def rotate_by_location(z: JaxArray, location: JaxArray) -> JaxArray:
    """Rotate z by the phase of location, preserving z's evidence strength.

    z' = z ⊙ l / |l|

    Each component of location is normalized to unit magnitude before multiplication, so the
    operation shifts phases without altering presence.  Zero-magnitude location components leave
    z unchanged.

    Args:
        z: Source phasors, shape (..., n).
        location: Location signal, shape (..., n).

    Returns:
        Rotated phasors with the same presence as z, shape (..., n).
    """
    magnitude = jnp.abs(location)
    unit_location = jnp.where(magnitude > 0, location / magnitude, jnp.ones_like(location))
    return z * unit_location
