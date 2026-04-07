from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
from efax import (
    ExpectationParametrization,
    Flattener,
    NaturalParametrization,
    expectation_parameters_from_characteristic_function,
)
from tjax import JaxArray, JaxComplexArray, JaxRealArray

from cem.phasor.frequency import make_frequency_grid


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

    @classmethod
    def from_distribution(
        cls,
        dist: NaturalParametrization,  # type: ignore[type-arg]
        frequencies: JaxRealArray,
    ) -> PhasorMessage:
        """Encode a belief distribution as a matrix of phasors via the characteristic function.

        For each frequency f_j and each sufficient-statistic component k, computes

            Z[j, k] = E[exp(i * f_j * T(x)_k)]
                     = exp(A(η + i * f_j * e_k) − A(η))

        where e_k is the k-th standard basis vector in natural-parameter space.  This is the
        distributional encoding from @eqn-distributional-encoding in the thesis.

        Args:
            dist: Exponential family belief, shape (*s).  The batch dimensions *s are preserved
                in the output.
            frequencies: Geometric frequency grid, shape (m,).  Typically produced by
                ``geometric_frequencies(m, base)``.

        Returns:
            PhasorMessage with data of shape (*s, m * d), where d = final_dimension_size() is
            the number of natural parameters (= number of sufficient-statistic components).
            The phasor at flat index ``j * d + k`` encodes sufficient statistic T(x)_k at
            frequency ``frequencies[j]``, for j in 0..m-1 and k in 0..d-1.
        """
        assert frequencies.ndim == 1
        flattener, _ = Flattener.flatten(dist, mapped_to_plane=False)
        t = make_frequency_grid(flattener, frequencies)

        # vmap over all batch dims of dist so each scalar element sees t of shape (m*d,),
        # producing output shape (*s, m*d).
        cf_fn = lambda d: d.characteristic_function(t)  # noqa: E731
        for _ in dist.shape:
            cf_fn = jax.vmap(cf_fn)
        cf = cf_fn(dist)  # shape (*s, m * d)
        return cls(cf)

    def to_distribution(
        self,
        t: NaturalParametrization,  # type: ignore[type-arg]
    ) -> ExpectationParametrization:  # type: ignore[type-arg]
        """Recover a belief distribution from phasors via OLS on the characteristic function.

        Inverts ``from_distribution`` by solving the overdetermined linear system

            Im(log Z[j, k]) ≈ f_j * E[T(x)_k]

        for the expectation parameters E[T(x)].  The estimate is exact for Normal
        distributions and a first-order approximation for other families.

        Args:
            t: The frequency grid used to produce this PhasorMessage, i.e. the same ``t``
                built internally by ``from_distribution``.  Shape (m * d,), where m is the
                number of frequencies and d is the natural-parameter dimension.  Each leaf
                has shape (m * d, *field_shape).

        Returns:
            ExpectationParametrization with shape (*s,), where *s are the batch dimensions
            of this PhasorMessage's data (shape (*s, m * d)).
        """
        cf_values: JaxComplexArray = self.data
        return expectation_parameters_from_characteristic_function(t, cf_values)

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
        assert frequencies.ndim == 1
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

    def dropout(self, key: JaxArray, rate: float | JaxRealArray) -> PhasorMessage:
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
