"""Check perceptron shapes, affine behaviour without hidden layers, and dropout."""

from collections.abc import Mapping

import jax.numpy as jnp
from tjax import RngStream

from cem.perceptron import MLP


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


def test_mlp_zero_dropout_rate_is_identity(streams: Mapping[str, RngStream]) -> None:
    f = MLP.create(4, 3, hidden_features=(6,), streams=streams)
    x = jnp.linspace(-1.0, 1.0, 4, dtype=jnp.float64)

    assert jnp.allclose(
        f.infer(x, streams=streams, inference=False),
        f.infer(x, streams=streams, inference=True),
    )


def test_mlp_dropout_preserves_expected_output(streams: Mapping[str, RngStream]) -> None:
    """Inverted dropout keeps the expected hidden activations, and the output layer is affine.

    With one hidden layer, the mean output in training mode therefore equals the inference output.
    """
    f = MLP.create(4, 3, hidden_features=(6,), dropout_rate=0.3, streams=streams)
    x = jnp.linspace(-1.0, 1.0, 4, dtype=jnp.float64)

    samples = jnp.stack([f.infer(x, streams=streams, inference=False) for _ in range(2000)])
    assert jnp.allclose(
        jnp.mean(samples, axis=0), f.infer(x, streams=streams, inference=True), atol=0.05
    )
