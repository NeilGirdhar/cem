from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import pytest
from efax import Flattener, NormalEP, NormalNP
from jax import tree
from tjax import frozendict

from cem.perceptron.nonlinear import LayerNorm
from cem.perceptron.target_node import PerceptronTargetConfiguration, PerceptronTargetNode
from cem.phasor.frequency import geometric_frequencies
from cem.phasor.loss import LossAndScore, reconstruction_loss, reconstruction_loss_and_score
from cem.phasor.message import PhasorMessage
from cem.phasor.target_node import PhasorTargetConfiguration, PhasorTargetNode
from cem.structure.graph import LearnableParameter, ParameterType

_M = 8
_BASE = 1e-4
_PRIOR = NormalNP(jnp.array(0.0), jnp.array(0.0))


@pytest.fixture
def freqs() -> jnp.ndarray:
    return geometric_frequencies(_M, base=_BASE)


@pytest.fixture
def target_node(freqs: jnp.ndarray) -> PhasorTargetNode:
    return PhasorTargetNode.create({"obs": _PRIOR}, freqs)


def infer_target_node(
    target_node: PhasorTargetNode,
    observed: dict[str, NormalNP],
    predicted: dict[str, PhasorMessage],
) -> PhasorTargetConfiguration:
    flat_observed = frozendict(
        {
            field: Flattener.flatten(dist, mapped_to_plane=True)[1]
            for field, dist in observed.items()
        }
    )
    z_hat_combined = PhasorMessage(
        jnp.concatenate([predicted[f].data for f in target_node.field_sizes], axis=-1)
    )
    return target_node.infer(flat_observed, z_hat_combined)


# ── reconstruction_loss_and_score ─────────────────────────────────────────────


def test_reconstruction_loss_and_score_returns_loss_and_score() -> None:
    z = PhasorMessage(jnp.array([1 + 0j, 0 + 1j]))
    assert isinstance(reconstruction_loss_and_score(z, z), LossAndScore)


def test_reconstruction_loss_and_score_score_is_phasor_message() -> None:
    z = PhasorMessage(jnp.array([1 + 0j, 0.5 - 0.5j]))
    z_hat = PhasorMessage(jnp.array([0.5 + 0.5j, 1 + 0j]))
    assert isinstance(reconstruction_loss_and_score(z, z_hat).score, PhasorMessage)


def test_reconstruction_loss_and_score_score_shape() -> None:
    z = PhasorMessage(jnp.array([1 + 0j, 0.5 - 0.5j, -1 + 0j]))
    z_hat = PhasorMessage(jnp.array([0.5 + 0.5j, 1 + 0j, 0 + 1j]))
    out = reconstruction_loss_and_score(z, z_hat)
    assert out.score.data.shape == z_hat.data.shape


def test_reconstruction_loss_and_score_loss_shape() -> None:
    z = PhasorMessage(jnp.array([1 + 0j, 0.5 - 0.5j]))
    z_hat = PhasorMessage(jnp.array([0.5 + 0.5j, 1 + 0j]))
    out = reconstruction_loss_and_score(z, z_hat)
    assert out.loss.shape == z_hat.data.shape


def test_reconstruction_loss_and_score_loss_matches_reconstruction_loss() -> None:
    observed = PhasorMessage(jnp.array([1 + 0j, 0.5 + 0.5j]))
    z_hat = PhasorMessage(jnp.array([0.8 + 0.2j, 0.3 - 0.3j]))
    out = reconstruction_loss_and_score(observed, z_hat)
    assert jnp.allclose(out.loss, reconstruction_loss(observed.data, z_hat.data))


def test_reconstruction_loss_and_score_total_loss_is_scalar() -> None:
    z = PhasorMessage(jnp.array([1 + 0j, 0 + 1j, 0.5 + 0.5j]))
    assert reconstruction_loss_and_score(z, z).total_loss().shape == ()


def test_reconstruction_loss_and_score_total_loss_is_sum_of_loss() -> None:
    observed = PhasorMessage(jnp.array([1 + 0j, 0.5 + 0.5j]))
    z_hat = PhasorMessage(jnp.array([0.8 + 0.2j, 0.3 - 0.3j]))
    out = reconstruction_loss_and_score(observed, z_hat)
    assert jnp.allclose(out.total_loss(), jnp.sum(out.loss))


def test_reconstruction_loss_and_score_self_score_is_zero() -> None:
    # At the minimum z_hat = observed, the gradient of the loss is zero.
    z = PhasorMessage(jnp.array([1 + 0j, 0 + 1j, 0.5 - 0.5j]))
    assert jnp.allclose(reconstruction_loss_and_score(z, z).score.data, 0.0, atol=1e-6)


def test_reconstruction_loss_and_score_score_equals_gradient() -> None:
    # score.data must equal jax.grad of the summed reconstruction loss w.r.t. z_hat.
    observed = PhasorMessage(jnp.array([1 + 0j, 0.5 + 0.5j]))
    z_hat = PhasorMessage(jnp.array([0.8 + 0.2j, 0.3 - 0.3j]))
    out = reconstruction_loss_and_score(observed, z_hat)
    expected = jax.grad(lambda z: jnp.sum(reconstruction_loss(observed.data, z.data)))(z_hat)
    assert jnp.allclose(out.score.data, expected.data)


def test_reconstruction_loss_and_score_batched_shapes() -> None:
    observed = PhasorMessage(jnp.ones((3, 4), dtype=jnp.complex128))
    z_hat = PhasorMessage(jnp.ones((3, 4), dtype=jnp.complex128) * (0.5 + 0.5j))
    out = reconstruction_loss_and_score(observed, z_hat)
    assert out.score.data.shape == (3, 4)
    assert out.loss.shape == (3, 4)
    assert out.total_loss().shape == ()


# ── PhasorTargetNode ──────────────────────────────────────────────────────────


def test_phasor_target_node_field_names(target_node: PhasorTargetNode) -> None:
    assert tuple(target_node.field_sizes) == ("obs",)


def test_phasor_target_node_has_frequency_grid_per_field(target_node: PhasorTargetNode) -> None:
    assert set(target_node.frequency_grids) == {"obs"}


def test_phasor_target_node_frequency_grid_shape(
    target_node: PhasorTargetNode, freqs: jnp.ndarray
) -> None:
    # NormalNP has d=2 sufficient statistics; m=8 frequencies → shape (m*d,) = (16,)
    grid = target_node.frequency_grids["obs"]
    assert grid.shape == (_M * 2,)


def test_phasor_target_node_multi_field(freqs: jnp.ndarray) -> None:
    node = PhasorTargetNode.create(
        {
            "x": NormalNP(jnp.array(0.0), jnp.array(0.0)),
            "y": NormalNP(jnp.array(1.0), jnp.array(-0.5)),
        },
        freqs,
    )
    assert set(node.field_sizes) == {"x", "y"}
    assert set(node.frequency_grids) == {"x", "y"}


def test_phasor_target_configuration_total_loss_is_zero(
    target_node: PhasorTargetNode, freqs: jnp.ndarray
) -> None:
    dist = NormalNP(jnp.array(0.5), jnp.array(-0.5))
    phasor = PhasorMessage.from_distribution(dist, freqs)
    config = PhasorTargetConfiguration(
        values=frozendict({"obs": phasor}),
        observed_distributions=frozendict({"obs": dist.to_exp()}),
        score=phasor.zeros_like(),
        loss=frozendict({"obs": jnp.zeros(phasor.shape)}),
        predicted_distributions=frozendict({"obs": dist.to_exp()}),
    )
    assert jnp.allclose(config.total_loss(), 0.0)


def test_phasor_target_node_returns_configuration(
    target_node: PhasorTargetNode, freqs: jnp.ndarray
) -> None:
    z_hat = PhasorMessage.from_distribution(_PRIOR, freqs)
    out = infer_target_node(target_node, {"obs": _PRIOR}, {"obs": z_hat})
    assert isinstance(out, PhasorTargetConfiguration)


def test_phasor_target_node_score_is_phasor_message(
    target_node: PhasorTargetNode, freqs: jnp.ndarray
) -> None:
    z_hat = PhasorMessage.from_distribution(_PRIOR, freqs)
    out = infer_target_node(target_node, {"obs": _PRIOR}, {"obs": z_hat})
    assert isinstance(out.score, PhasorMessage)
    assert out.score.data.shape == z_hat.data.shape


def test_phasor_target_node_predicted_distribution_recovers_mean(
    target_node: PhasorTargetNode, freqs: jnp.ndarray
) -> None:
    mu, sigma2 = 0.5, 1.0
    dist = NormalNP(jnp.array(mu / sigma2), jnp.array(-0.5 / sigma2))
    z_hat = PhasorMessage.from_distribution(dist, freqs)
    out = infer_target_node(target_node, {"obs": dist}, {"obs": z_hat})
    assert jnp.allclose(out.predicted_distributions["obs"].mean, jnp.array(mu), atol=1e-2)  # ty: ignore[unresolved-attribute]


def test_phasor_target_node_total_loss_is_sum_of_field_losses(
    freqs: jnp.ndarray,
) -> None:
    node = PhasorTargetNode.create(
        {
            "x": NormalNP(jnp.array(0.0), jnp.array(0.0)),
            "y": NormalNP(jnp.array(1.0), jnp.array(-0.5)),
        },
        freqs,
    )
    x_dist = NormalNP(jnp.array(0.1), jnp.array(-0.5))
    y_dist = NormalNP(jnp.array(0.7), jnp.array(-0.25))
    out = infer_target_node(
        node,
        {"x": x_dist, "y": y_dist},
        {
            "x": PhasorMessage.from_distribution(x_dist, freqs),
            "y": PhasorMessage.from_distribution(y_dist, freqs),
        },
    )
    assert jnp.allclose(out.total_loss(), jnp.sum(out.loss["x"]) + jnp.sum(out.loss["y"]))


def test_phasor_target_node_multi_field_score_is_joined_on_last_dimension(
    freqs: jnp.ndarray,
) -> None:
    node = PhasorTargetNode.create(
        {
            "x": NormalNP(jnp.array(0.0), jnp.array(0.0)),
            "y": NormalNP(jnp.array(1.0), jnp.array(-0.5)),
        },
        freqs,
    )
    x_phasor = PhasorMessage.from_distribution(NormalNP(jnp.array(0.1), jnp.array(-0.5)), freqs)
    y_phasor = PhasorMessage.from_distribution(NormalNP(jnp.array(0.7), jnp.array(-0.25)), freqs)
    out = infer_target_node(
        node,
        {
            "x": NormalNP(jnp.array(0.1), jnp.array(-0.5)),
            "y": NormalNP(jnp.array(0.7), jnp.array(-0.25)),
        },
        {"x": x_phasor, "y": y_phasor},
    )
    assert out.score.data.shape == (x_phasor.data.shape[-1] + y_phasor.data.shape[-1],)


def infer_perceptron_target_node(
    target_node: PerceptronTargetNode,
    observed: dict[str, NormalNP],
    predicted: dict[str, jnp.ndarray],
) -> PerceptronTargetConfiguration:
    flat_observed = frozendict(
        {
            field: Flattener.flatten(dist, mapped_to_plane=True)[1]
            for field, dist in observed.items()
        }
    )
    concat_prediction = jnp.concatenate([predicted[f] for f in target_node.field_sizes], axis=-1)
    return target_node.infer(flat_observed, concat_prediction)


def test_perceptron_target_node_partition_round_trip_preserves_behavior() -> None:
    dist = NormalNP(jnp.asarray(0.25), jnp.asarray(-0.5))
    _, y_hat = Flattener.flatten(dist, mapped_to_plane=True)
    node = PerceptronTargetNode.create({"obs": dist})

    extracted, remainder = eqx.partition(node, eqx.is_array)
    round_tripped = eqx.combine(extracted, remainder)

    expected = infer_perceptron_target_node(node, {"obs": dist}, {"obs": y_hat})
    out = infer_perceptron_target_node(round_tripped, {"obs": dist}, {"obs": y_hat})
    assert jnp.allclose(out.total_loss(), expected.total_loss())
    predicted_d = out.predicted_distributions["obs"]
    expected_d = expected.predicted_distributions["obs"]
    assert isinstance(predicted_d, NormalEP)
    assert isinstance(expected_d, NormalEP)
    assert jnp.allclose(predicted_d.mean, expected_d.mean)


def test_layer_norm_partition_round_trip_preserves_eps() -> None:
    layer_norm = LayerNorm.create(2, eps=1e-3)
    extracted, remainder = eqx.partition(layer_norm, lambda x: isinstance(x, float))
    round_tripped = eqx.combine(extracted, remainder)

    assert tree.leaves(extracted) == [1e-3]

    x = jnp.asarray([1.0, 3.0])
    assert jnp.allclose(round_tripped.infer(x), layer_norm.infer(x))


def test_parameter_type_partition_round_trip_preserves_type() -> None:
    parameter_type = ParameterType(LearnableParameter)
    extracted, remainder = eqx.partition(parameter_type, lambda x: isinstance(x, type))
    round_tripped = eqx.combine(extracted, remainder)

    # t is static, so it does not appear as a dynamic pytree leaf.
    assert tree.leaves(extracted) == []
    assert round_tripped.t is LearnableParameter
