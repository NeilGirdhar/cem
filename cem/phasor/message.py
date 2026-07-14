import jax
import jax.numpy as jnp
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

type JaxComplexArray = JaxArray


def has_single_scalar_parameter(p: Distribution) -> bool:
    params = parameters(p, fixed=False, support=True)
    return len(params) == 1 and all(
        isinstance(support, ScalarSupport) for _value, support in params.values()
    )


def phasor_from_distribution(
    dist: NaturalParametrization,
    frequencies: JaxRealArray,
    *,
    presences: JaxRealArray | None = None,
    raveled: bool = False,
) -> JaxComplexArray:
    """Encode a belief distribution as characteristic-function phasors.

    For each frequency f_j and each sufficient-statistic component k, computes

        Z[j, k] = E[exp(i * f_j * T(x)_k)]
                 = exp(A(η + i * f_j * e_k) - A(η)).

    The result has shape (*s, m * d) when ``raveled=False`` or
    (prod(*s) * m * d,) when ``raveled=True``. Here *s is ``dist.shape`` and d is
    the number of natural parameters.
    """
    assert frequencies.ndim == 1
    flattener, _ = Flattener.flatten(dist, mapped_to_plane=False)
    t = make_frequency_grid(flattener, frequencies)

    cf_fn = lambda d: d.characteristic_function(t)  # noqa: E731
    for _ in dist.shape:
        cf_fn = jax.vmap(cf_fn)
    cf = cf_fn(dist)
    if presences is not None:
        cf *= presences[..., jnp.newaxis]
    return cf.reshape(-1) if raveled else cf


def phasor_to_distribution(
    z: JaxComplexArray,
    t: NaturalParametrization,
) -> ExpectationParametrization:
    """Recover expectation parameters from phasors via the characteristic function."""
    return expectation_parameters_from_characteristic_function(t, z)


def phasor_to_conjugate_prior(
    z: JaxComplexArray,
    t: NaturalParametrization,
) -> NaturalParametrization:
    """Recover a d=1 HasConjugatePrior distribution and presence from phasors via OLS."""
    ep_cls = type(t).expectation_parametrization_cls()
    assert issubclass(ep_cls, HasConjugatePrior), (
        f"{ep_cls.__name__} does not implement HasConjugatePrior"
    )
    assert has_single_scalar_parameter(t), f"{ep_cls.__name__} is non-scalar."

    ep = expectation_parameters_from_characteristic_function(t, z)
    assert isinstance(ep, HasConjugatePrior)

    f: JaxRealArray = jax.tree_util.tree_leaves(t)[0]
    f_sq = f**2
    re_log_z: JaxRealArray = jnp.real(jnp.log(z))

    mean_f_sq = jnp.mean(f_sq)
    var_f_sq = jnp.mean(f_sq**2) - mean_f_sq**2
    mean_re = jnp.mean(re_log_z, axis=-1)
    cov = jnp.mean(f_sq * re_log_z, axis=-1) - mean_f_sq * mean_re
    b_hat = cov / var_f_sq
    presence: JaxRealArray = jnp.exp(mean_re - b_hat * mean_f_sq)

    return ep.conjugate_prior_distribution(presence)


def encode_scalar_phasors(
    x: JaxArray,
    presence: JaxArray,
    frequencies: JaxArray,
) -> JaxComplexArray:
    """Encode scalar observations as multi-frequency phasors."""
    assert frequencies.ndim == 1
    phases = x[..., jnp.newaxis] * jnp.reshape(frequencies, (1,) * x.ndim + (-1,))
    return presence[..., jnp.newaxis] * jnp.exp(1j * phases)


def phasor_concordance(left: JaxComplexArray, right: JaxComplexArray) -> JaxRealArray:
    """Elementwise agreement Re(left * conj(right))."""
    return jnp.real(left * jnp.conj(right))


def phasor_to_real(z: JaxComplexArray) -> JaxArray:
    """Return (Re, Im) parts concatenated as a real array for neural network input."""
    return jnp.concat([jnp.real(z), jnp.imag(z)], axis=-1)
