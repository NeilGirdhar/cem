"""Check perceptron shapes, affine behaviour without hidden layers, and dropout."""

from collections.abc import Mapping

import jax.numpy as jnp
from tjax import RngStream

from cem.perceptron import MLP
from cem.structure.graph import LossBalancedParameter


def test_mlp_parameter_factory_wraps_each_affine_layer(
    streams: Mapping[str, RngStream],
) -> None:
    mlp = MLP.create(
        2,
        1,
        hidden_features=(3,),
        streams=streams,
        parameter_factory=LossBalancedParameter,
    )
    for layer in mlp.layers:
        for parameter in (layer.weight, layer.bias):
            assert isinstance(parameter, LossBalancedParameter)


def test_mlp_output_shape(streams: Mapping[str, RngStream]) -> None:
    f = MLP.create(4, 2, hidden_features=8, streams=streams)

    assert f.infer(jnp.ones(4, dtype=jnp.float64), streams=streams, inference=True).shape == (2,)


def test_mlp_supports_multiple_hidden_sizes(streams: Mapping[str, RngStream]) -> None:
    f = MLP.create(4, 2, hidden_features=(8, 6, 5), streams=streams)

    assert [layer.weight.value.shape for layer in f.layers] == [
        (8, 4),
        (6, 8),
        (5, 6),
        (2, 5),
    ]


def test_mlp_without_hidden_layers_is_affine(streams: Mapping[str, RngStream]) -> None:
    f = MLP.create(4, 2, streams=streams)
    x = jnp.ones((3, 4), dtype=jnp.float64)

    assert len(f.layers) == 1
    assert f.infer(x, streams=streams, inference=True).shape == (3, 2)


def test_mlp_dropout_applies_when_inference_false(streams: Mapping[str, RngStream]) -> None:
    f = MLP.create(4, 20, hidden_features=(20,), dropout_rate=0.5, streams=streams)
    x = jnp.ones(4, dtype=jnp.float64)

    assert not jnp.allclose(
        f.infer(x, streams=streams, inference=False),
        f.infer(x, streams=streams, inference=True),
    )
