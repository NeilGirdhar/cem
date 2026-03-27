from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
from tjax import JaxArray


class PhasorMessage(eqx.Module):
    """A vector of evidence phasors encoding feature presence and value.

    Each phasor z_j = presence_j * exp(i * value_j) encodes:
    - presence_j = |z_j| >= 0: how strongly feature j is supported.
    - value_j = arg(z_j) in (-pi, pi]: which value of feature j is supported.

    Under the von Mises interpretation, phasors are the natural parameters of von Mises
    distributions.  Independent evidence combines by complex addition, and is forgotten by real
    scaling.

    Attributes:
        data: Complex array of phasors, shape (..., m).
    """

    data: JaxArray  # complex128

    @property
    def shape(self) -> tuple[int, ...]:
        return self.data.shape

    @property
    def ndim(self) -> int:
        return self.data.ndim

    @property
    def presence(self) -> JaxArray:
        """Magnitude of each phasor — evidence strength."""
        return jnp.abs(self.data)

    @property
    def value(self) -> JaxArray:
        """Phase of each phasor in (-pi, pi] — supported feature value."""
        return jnp.angle(self.data)

    # Construction ---------------------------------------------------------------------------------

    @classmethod
    def zeros(cls, features: int) -> PhasorMessage:
        """Zero phasor vector — no evidence."""
        return cls(jnp.zeros(features, dtype=jnp.complex128))

    def zeros_like(self) -> PhasorMessage:
        return type(self)(jnp.zeros_like(self.data))

    @classmethod
    def from_polar(cls, presence: JaxArray, value: JaxArray) -> PhasorMessage:
        """Construct from presence (magnitude) and value (phase).

        Args:
            presence: Evidence strength, shape (..., m).
            value: Supported feature value as phase, shape (..., m).

        Returns:
            PhasorMessage with data = presence * exp(i * value).
        """
        return cls(presence * jnp.exp(1j * value))

    @classmethod
    def encode_scalar(
        cls,
        x: JaxArray,
        presence: JaxArray,
        frequencies: JaxArray,
    ) -> PhasorMessage:
        """Encode a scalar observation as a vector of phasors.

        Each phasor_j = presence * exp(i * k_j * x), where the k_j are geometrically spaced
        frequencies giving multi-scale coverage of x.  This is the same scheme used by rotary
        positional embeddings (RoPE).

        Args:
            x: Scalar value to encode, shape (...).
            presence: Evidence weight (e.g. shutter time or 1.0), shape (...).
            frequencies: Frequencies k_j, shape (m,). Use geometric_frequencies() for the standard
                geometrically spaced choice.

        Returns:
            PhasorMessage with shape (..., m).
        """
        phases = x[..., jnp.newaxis] * jnp.reshape(frequencies, (1,) * x.ndim + (-1,))  # (..., m)
        return cls(presence[..., jnp.newaxis] * jnp.exp(1j * phases))

    # Operations -----------------------------------------------------------------------------------

    def combined(self, other: PhasorMessage, /) -> PhasorMessage:
        """Combine independent evidence by complex addition (natural parameter addition)."""
        return PhasorMessage(self.data + other.data)

    def scaled(self, scale: JaxArray) -> PhasorMessage:
        """Scale evidence strength by a real factor, preserving phase."""
        return PhasorMessage(self.data * scale)

    def rotated(self, rotation: JaxArray) -> PhasorMessage:
        """Rotate phasors by a complex value, shifting their phases.

        Multiplication by a unit-magnitude complex adds a constant offset to each phase while
        leaving presence unchanged.  Because rotations compose additively, a receiver that knows
        the rotation can apply the inverse to recover the original value.

        Args:
            rotation: Complex rotation factor, shape broadcastable with data. Use unit magnitude to
                preserve evidence strength.

        Returns:
            PhasorMessage with each phasor multiplied by rotation.
        """
        return PhasorMessage(self.data * rotation)

    def concordance(self, other: PhasorMessage) -> JaxArray:
        """Measure agreement between two phasors.

        Computes Re(z_a * conj(z_b)) = presence_a * presence_b * cos(value_a - value_b), the
        natural inner product on von Mises natural parameters.  Used in attention to implement
        Bayesian evidence weighting: selection weights proportional to the likelihood of each key
        under the query distribution.

        Args:
            other: Phasor to measure agreement with.

        Returns:
            Real array of elementwise concordance values, same shape as data.
        """
        return jnp.real(self.data * jnp.conj(other.data))

    def dropout(self, key: JaxArray, rate: float) -> PhasorMessage:
        """Apply phasor dropout: zero with probability rate, scale by 1/(1-rate) otherwise.

        Preserves expected phasor value while corrupting only along the ray (not by rotation),
        analogous to standard real-valued dropout.  Rotations are excluded because they would
        create evidence for an orthogonal feature.

        Args:
            key: JAX random key.
            rate: Dropout probability in [0, 1).

        Returns:
            Post-dropout PhasorMessage with the same expected value.
        """
        mask = jr.bernoulli(key, 1.0 - rate, shape=self.data.shape)
        scaled = self.data / (1.0 - rate)
        return PhasorMessage(jnp.where(mask, scaled, jnp.zeros_like(self.data)))

    def to_real(self) -> JaxArray:
        """Return (Re, Im) parts concatenated as a float array for neural network input.

        The concatenated (Re(z), Im(z)) vector is exactly the natural parameter representation
        of the von Mises distribution.
        """
        return jnp.concat([jnp.real(self.data), jnp.imag(self.data)], axis=-1)


def geometric_frequencies(num_features: int, base: float = 2.0 * jnp.pi) -> JaxArray:
    """Generate geometrically spaced frequencies k_j = base * 2^j for scalar encoding.

    Args:
        num_features: Number of frequency components m.
        base: Base frequency. Default is 2*pi as in the thesis.

    Returns:
        Float array of shape (m,) with k_j = base * 2^j for j = 0, ..., m-1.
    """
    return base * jnp.pow(2.0, jnp.arange(num_features, dtype=jnp.float64))
