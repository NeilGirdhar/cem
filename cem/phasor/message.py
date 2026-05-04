from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
from efax import (
    Distribution,
    ExpectationParametrization,
    Flattener,
    HasConjugatePrior,
    NaturalParametrization,
    ScalarSupport,
    expectation_parameters_from_characteristic_function,
    parameters,
)
from tjax import JaxArray, JaxRealArray

from cem.phasor.frequency import make_frequency_grid


def has_single_scalar_parameter(p: Distribution) -> bool:
    params = parameters(p, fixed=False, support=True)
    return len(params) == 1 and all(
        isinstance(support, ScalarSupport) for _value, support in params.values()
    )


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
        dist: NaturalParametrization,
        frequencies: JaxRealArray,
        *,
        presences: JaxRealArray | None = None,
        raveled: bool = False,
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
            presences: Optional per-sample evidence weight, shape (*s,) matching ``dist.shape``.
                Scales each phasor's magnitude by the corresponding presence.  When None,
                presences default to 1.
            raveled: If False (default), returns data of shape (*s, m * d).  If True, all
                dimensions are raveled into a single flat vector of shape (prod(*s) * m * d,).

        Returns:
            PhasorMessage with data of shape (*s, m * d) when ``raveled=False``, or shape
            (prod(*s) * m * d,) when ``raveled=True``.  Here d = final_dimension_size() is
            the number of natural parameters (= number of sufficient-statistic components).
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
        if presences is not None:
            cf *= presences[..., jnp.newaxis]
        return cls(cf.reshape(-1) if raveled else cf)

    def to_distribution(
        self,
        t: NaturalParametrization,
    ) -> ExpectationParametrization:
        """Recover the expectation parametrization from phasors via the characteristic function.

        Works for any exponential family (any d).  Does not recover presence.
        Use :meth:`to_conjugate_prior` for d=1 distributions when presence is also needed.

        Args:
            t: Frequency grid built by ``make_frequency_grid`` for this distribution family.

        Returns:
            The recovered expectation parametrization, shape (*s,).
        """
        return expectation_parameters_from_characteristic_function(t, self.data)

    def to_conjugate_prior(
        self,
        t: NaturalParametrization,
    ) -> NaturalParametrization:
        """Recover a d=1 HasConjugatePrior distribution and presence from phasors via OLS.

        Requires a scalar exponential family.

        Two OLS passes on log Z_j:
        - Expectation parameter: Im(log Z_j) ≈ f_j · μ.  Solved by OLS on the imaginary
          part.  Exact for Normal distributions.
        - Presence: Re(log Z_j) ≈ log p − f_j²/(2p).  Solved by OLS of the real part on
          f_j².  The intercept gives log p.  Exact for Normal distributions.

        Args:
            t: Frequency grid of shape (m,), built by ``make_frequency_grid`` for this
               distribution family.

        Returns:
            The conjugate prior natural parametrization encoding the recovered mean and
            presence, shape (*s,).
        """
        ep_cls = type(t).expectation_parametrization_cls()
        assert issubclass(ep_cls, HasConjugatePrior), (
            f"{ep_cls.__name__} does not implement HasConjugatePrior"
        )
        assert has_single_scalar_parameter(t), f"{ep_cls.__name__} is non-scalar."

        ep = expectation_parameters_from_characteristic_function(t, self.data)
        assert isinstance(ep, HasConjugatePrior)

        # Presence from Re OLS: Re(log Z_j) = log p − f_j²/(2p)
        # b̂ = Cov_j(f_j², Re(log Z_j)) / Var_j(f_j²)
        # log p = E_j[Re(log Z_j)] − b̂ · E_j[f_j²]
        f: JaxRealArray = jax.tree_util.tree_leaves(t)[0]  # frequencies, shape (m,)
        f_sq = f**2  # (m,)
        re_log_z: JaxRealArray = jnp.real(jnp.log(self.data))  # (..., m)

        mean_f_sq = jnp.mean(f_sq)
        var_f_sq = jnp.mean(f_sq**2) - mean_f_sq**2
        mean_re = jnp.mean(re_log_z, axis=-1)  # (...,)
        cov = jnp.mean(f_sq * re_log_z, axis=-1) - mean_f_sq * mean_re  # (...,)
        b_hat = cov / var_f_sq
        presence: JaxRealArray = jnp.exp(mean_re - b_hat * mean_f_sq)  # (...,)

        return ep.conjugate_prior_distribution(presence)

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

    def split_frequencies(self, n_frequencies: int) -> PhasorMessage:
        """Reshape a raveled phasor vector into (n_components, n_frequencies).

        Inverse of raveling: splits a flat ``(n_components * n_frequencies,)`` phasor back
        into the structured ``(n_components, n_frequencies)`` form where each row is one
        component's frequency encoding.

        Args:
            n_frequencies: Number of frequencies per component.

        Returns:
            PhasorMessage with data of shape (n_components, n_frequencies).
        """
        return PhasorMessage(self.data.reshape(-1, n_frequencies))

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
