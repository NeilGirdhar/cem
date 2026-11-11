"""Check Gaussian message encodings, moment propagation, and parameter counts."""

import math
from collections.abc import Mapping

import equinox as eqx
import jax.numpy as jnp
from efax import NormalPP, NormalVP
from jax import tree
from tjax import RngStream

from cem.npn.gaussian import (
    GaussianNeuralProjection,
    GaussianNeuralProjectionInputPool,
    GaussianNeuralProjectionLayer,
    gaussian_relu,
)
from cem.structure.graph import (
    LossBalancedParameter,
    SummedGradientParameter,
    count_real_learnable_parameters,
    is_parameter,
)


def test_gaussian_neural_projection_layer_propagates_marginal_moments() -> None:
    layer = GaussianNeuralProjectionLayer(
        weight=SummedGradientParameter(jnp.asarray([[2.0, -1.0], [0.5, 3.0]])),
        bias=SummedGradientParameter(jnp.asarray([0.25, -0.5])),
        bias_raw_variance=SummedGradientParameter(jnp.full(2, math.log(math.expm1(0.5)))),
    )
    inputs = NormalVP(mean=jnp.asarray([1.0, -2.0]), variance=jnp.asarray([0.5, 4.0]))

    output = layer.infer(inputs)

    assert jnp.allclose(output.mean, jnp.asarray([4.25, -6.0]))
    assert jnp.allclose(output.variance, jnp.asarray([6.5, 36.625]))


def test_gaussian_neural_projection_input_pool_adds_dummy_evidence() -> None:
    """Check that background evidence makes an absent input usable for moment conversion."""
    pool = GaussianNeuralProjectionInputPool(
        dummy_mean=SummedGradientParameter(jnp.zeros(2)),
        dummy_raw_presence=SummedGradientParameter(jnp.full(2, jnp.log(jnp.expm1(1.0)))),
    )
    inputs = NormalPP(mean=jnp.asarray([99.0, 2.0]), precision=jnp.asarray([0.0, 4.0])).to_nat()

    output = pool.infer(inputs).to_variance_parametrization()

    assert jnp.allclose(output.mean, jnp.asarray([0.0, 1.6]))
    assert jnp.allclose(output.variance, jnp.asarray([1.0, 0.2]))


def test_gaussian_relu_matches_standard_normal_moments() -> None:
    rectified = gaussian_relu(NormalVP(jnp.zeros(()), jnp.ones(())))

    assert jnp.allclose(rectified.mean, 1 / jnp.sqrt(2 * jnp.pi))
    assert jnp.allclose(rectified.variance, 0.5 - 1 / (2 * jnp.pi))


def test_gaussian_neural_projection_factory_wraps_input_pool_and_all_layers(
    streams: Mapping[str, RngStream],
) -> None:
    model = GaussianNeuralProjection.create(
        2,
        1,
        hidden_features=(3,),
        streams=streams,
        parameter_factory=LossBalancedParameter,
    )
    parameters = tree.leaves(model, is_leaf=is_parameter)
    assert len(parameters) == 2 + 3 * len(model.layers)
    for parameter in parameters:
        assert isinstance(parameter, LossBalancedParameter)


def test_gaussian_neural_projection_parameter_count(streams: Mapping[str, RngStream]) -> None:
    """The input pool has two parameters per input; each layer has weights and a bias pair."""
    network = GaussianNeuralProjection.create(4, 2, hidden_features=8, streams=streams)

    assert count_real_learnable_parameters(network) == 2 * 4 + (4 * 8 + 2 * 8) + (8 * 2 + 2 * 2)


def test_gaussian_neural_projection_without_inputs_is_a_learned_prior(
    streams: Mapping[str, RngStream],
) -> None:
    """With no inputs, the output is the last layer's Gaussian bias, which training can reach."""
    network = GaussianNeuralProjection.create(0, 2, streams=streams)
    empty = NormalPP(jnp.zeros(0), jnp.zeros(0)).to_nat()
    output = network.infer(empty).to_variance_parametrization()
    assert jnp.allclose(output.mean, network.layers[-1].bias.value)
    assert jnp.allclose(output.variance, 1.0)

    target = NormalVP(jnp.zeros(2), jnp.full(2, 4.0))
    gradient = eqx.filter_grad(
        lambda n: jnp.sum(target.cross_entropy(n.infer(empty).to_variance_parametrization()))
    )(network)
    assert jnp.all(gradient.layers[-1].bias_raw_variance.value < 0)
