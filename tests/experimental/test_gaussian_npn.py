from collections.abc import Mapping

import jax.numpy as jnp
from efax import NormalVP
from tjax import RngStream

from cem.experimental.gaussian_npn import (
    GaussianNPN,
    GaussianNPNInputPool,
    GaussianNPNLinear,
    gaussian_relu,
    normal_from_mean_and_presence,
)
from cem.structure.graph import LearnableParameter, count_real_learnable_parameters


def test_normal_natural_parameters_store_mean_times_presence() -> None:
    mean = jnp.array([-2.0, 3.0])
    presence = jnp.array([0.5, 4.0])

    distribution = normal_from_mean_and_presence(mean, presence)
    recovered = distribution.to_variance_parametrization()

    assert jnp.allclose(distribution.mean_times_precision, mean * presence)
    assert jnp.allclose(-2 * distribution.negative_half_precision, presence)
    assert jnp.allclose(recovered.mean, mean)
    assert jnp.allclose(recovered.variance, 1 / presence)


def test_gaussian_npn_linear_propagates_marginal_moments() -> None:
    layer = GaussianNPNLinear(
        weight=LearnableParameter(jnp.array([[2.0, -1.0], [0.5, 3.0]])),
        bias=LearnableParameter(jnp.array([0.25, -0.5])),
    )
    inputs = NormalVP(
        mean=jnp.array([1.0, -2.0]),
        variance=jnp.array([0.5, 4.0]),
    ).to_nat()

    output = layer.infer(inputs).to_variance_parametrization()

    assert jnp.allclose(output.mean, jnp.array([4.25, -6.0]))
    assert jnp.allclose(output.variance, jnp.array([6.0, 36.125]))


def test_gaussian_npn_input_pool_adds_dummy_evidence() -> None:
    """The learned dummy makes missing inputs proper before moment conversion."""
    pool = GaussianNPNInputPool(
        dummy_mean=LearnableParameter(jnp.zeros(2)),
        dummy_raw_presence=LearnableParameter(jnp.full(2, jnp.log(jnp.expm1(1.0)))),
    )
    inputs = normal_from_mean_and_presence(
        mean=jnp.array([99.0, 2.0]),
        presence=jnp.array([0.0, 4.0]),
    )

    output = pool.infer(inputs).to_variance_parametrization()

    assert jnp.allclose(output.mean, jnp.array([0.0, 1.6]))
    assert jnp.allclose(output.variance, jnp.array([1.0, 0.2]))


def test_gaussian_relu_matches_standard_normal_moments() -> None:
    standard_normal = NormalVP(jnp.zeros(()), jnp.ones(())).to_nat()

    rectified = gaussian_relu(standard_normal).to_variance_parametrization()

    assert jnp.allclose(rectified.mean, 1 / jnp.sqrt(2 * jnp.pi))
    assert jnp.allclose(rectified.variance, 0.5 - 1 / (2 * jnp.pi))


def test_gaussian_npn_matches_perceptron_parameter_count(
    streams: Mapping[str, RngStream],
) -> None:
    network = GaussianNPN.create(4, 2, hidden_features=8, streams=streams)

    assert count_real_learnable_parameters(network) == 2 * 4 + 4 * 8 + 8 + 8 * 2 + 2
