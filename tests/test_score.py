import equinox as eqx
import jax
import jax.numpy as jnp
import pytest
from efax import (
    ComplexVonMisesNP,
    Flattener,
    NormalEP,
    NormalNP,
    UnitVarianceNormalEP,
    UnitVarianceNormalNP,
)
from jax import tree
from tjax import frozendict

from cem.perceptron.target_node import PerceptronTargetConfiguration, PerceptronTargetNode
from cem.phasor.frequency import geometric_frequencies
from cem.phasor.loss import (
    LossAndScore,
    spectral_reconstruction_loss_and_score,
)
from cem.phasor.message import JaxComplexArray, phasor_from_distribution
from cem.phasor.target_node import PhasorTargetConfiguration, PhasorTargetNode
from cem.structure.graph import LearnableParameter, ParameterType

_M = 8
_BASE = 1.0  # f* = 1/σ = 1 for UnitVarianceNormal
_PRIOR = UnitVarianceNormalNP(jnp.array(0.0))


@pytest.fixture
def freqs() -> jnp.ndarray:
    return geometric_frequencies(_M, base=_BASE)


@pytest.fixture
def target_node(freqs: jnp.ndarray) -> PhasorTargetNode:
    return PhasorTargetNode.create({"obs": _PRIOR}, freqs)


def infer_target_node(
    target_node: PhasorTargetNode,
    observed: dict[str, UnitVarianceNormalNP],
    predicted: dict[str, JaxComplexArray],
) -> PhasorTargetConfiguration:
    flat_observed = frozendict(
        {
            field: Flattener.flatten(dist, mapped_to_plane=True)[1]
            for field, dist in observed.items()
        }
    )
    z_hat_combined = jnp.concatenate([predicted[f] for f in target_node.field_sizes], axis=-1)
    return target_node.infer(flat_observed, z_hat_combined)


# ── reconstruction_loss_and_score ─────────────────────────────────────────────


def test_reconstruction_loss_and_score_returns_loss_and_score() -> None:
    z = jnp.array([1 + 0j, 0 + 1j])
    assert isinstance(spectral_reconstruction_loss_and_score(z, z), LossAndScore)


def test_reconstruction_loss_and_score_score_is_array() -> None:
    z = jnp.array([1 + 0j, 0.5 - 0.5j])
    z_hat = jnp.array([0.5 + 0.5j, 1 + 0j])
    assert spectral_reconstruction_loss_and_score(z, z_hat).score.shape == z_hat.shape


def test_reconstruction_loss_and_score_score_shape() -> None:
    z = jnp.array([1 + 0j, 0.5 - 0.5j, -1 + 0j])
    z_hat = jnp.array([0.5 + 0.5j, 1 + 0j, 0 + 1j])
    out = spectral_reconstruction_loss_and_score(z, z_hat)
    assert out.score.shape == z_hat.shape


def test_reconstruction_loss_and_score_loss_shape() -> None:
    z = jnp.array([1 + 0j, 0.5 - 0.5j])
    z_hat = jnp.array([0.5 + 0.5j, 1 + 0j])
    out = spectral_reconstruction_loss_and_score(z, z_hat)
    assert out.loss.shape == ()


def test_reconstruction_loss_and_score_loss_matches_reconstruction_loss() -> None:
    observed = jnp.array([1 + 0j, 0.5 + 0.5j])
    z_hat = jnp.array([0.8 + 0.2j, 0.3 - 0.3j])
    out = spectral_reconstruction_loss_and_score(observed, z_hat)
    observed_dist = ComplexVonMisesNP(observed)
    predicted_dist = ComplexVonMisesNP(z_hat)
    elementwise_loss = observed_dist.to_exp().kl_divergence(predicted_dist, self_nat=observed_dist)
    assert jnp.allclose(out.loss, jnp.mean(elementwise_loss))


def test_reconstruction_loss_and_score_total_loss_is_scalar() -> None:
    z = jnp.array([1 + 0j, 0 + 1j, 0.5 + 0.5j])
    assert spectral_reconstruction_loss_and_score(z, z).total_loss().shape == ()


def test_reconstruction_loss_and_score_total_loss_is_sum_of_loss() -> None:
    observed = jnp.array([1 + 0j, 0.5 + 0.5j])
    z_hat = jnp.array([0.8 + 0.2j, 0.3 - 0.3j])
    out = spectral_reconstruction_loss_and_score(observed, z_hat)
    assert jnp.allclose(out.total_loss(), jnp.sum(out.loss))


def test_reconstruction_loss_and_score_self_score_is_zero() -> None:
    # At the minimum z_hat = observed, the gradient of the loss is zero.
    z = jnp.array([1 + 0j, 0 + 1j, 0.5 - 0.5j])
    assert jnp.allclose(spectral_reconstruction_loss_and_score(z, z).score, 0.0, atol=1e-6)


def test_reconstruction_loss_and_score_score_equals_gradient() -> None:
    # score must equal jax.grad of the mean reconstruction loss w.r.t. z_hat.
    observed = jnp.array([1 + 0j, 0.5 + 0.5j])
    z_hat = jnp.array([0.8 + 0.2j, 0.3 - 0.3j])
    out = spectral_reconstruction_loss_and_score(observed, z_hat)

    def direct_loss(z: jnp.ndarray) -> jnp.ndarray:
        observed_dist = ComplexVonMisesNP(observed)
        predicted_dist = ComplexVonMisesNP(z)
        elementwise_loss = observed_dist.to_exp().kl_divergence(
            predicted_dist, self_nat=observed_dist
        )
        return jnp.mean(elementwise_loss)

    expected = jax.grad(
        direct_loss,
    )(z_hat)
    assert jnp.allclose(out.score, expected)


def test_reconstruction_loss_and_score_batched_shapes() -> None:
    observed = jnp.ones((3, 4), dtype=jnp.complex128)
    z_hat = jnp.ones((3, 4), dtype=jnp.complex128) * (0.5 + 0.5j)
    out = spectral_reconstruction_loss_and_score(observed, z_hat)
    assert out.score.shape == (3, 4)
    assert out.loss.shape == (3,)
    assert out.total_loss().shape == ()


# ── PhasorTargetNode ──────────────────────────────────────────────────────────


def test_phasor_target_node_field_names(target_node: PhasorTargetNode) -> None:
    assert tuple(target_node.field_sizes) == ("obs",)


def test_phasor_target_node_has_particle_bounds_per_field(target_node: PhasorTargetNode) -> None:
    assert set(target_node.particle_bounds.value) == {"obs"}


def test_phasor_target_node_particle_state_shape(target_node: PhasorTargetNode) -> None:
    state = target_node.initial_particle_state()
    assert state.positions["obs"].shape == (target_node.particle_inference.n_particles, 1)


def test_phasor_target_node_multi_field(freqs: jnp.ndarray) -> None:
    node = PhasorTargetNode.create(
        {
            "x": UnitVarianceNormalNP(jnp.array(0.0)),
            "y": UnitVarianceNormalNP(jnp.array(1.0)),
        },
        freqs,
    )
    assert set(node.field_sizes) == {"x", "y"}
    assert set(node.particle_bounds.value) == {"x", "y"}


def test_phasor_target_configuration_total_loss_is_zero(
    target_node: PhasorTargetNode, freqs: jnp.ndarray
) -> None:
    dist = UnitVarianceNormalNP(jnp.array(0.5))
    phasor = phasor_from_distribution(dist, freqs)
    config = PhasorTargetConfiguration(
        values=frozendict({"obs": phasor}),
        observed_distributions=frozendict({"obs": dist.to_exp()}),
        score=jnp.zeros_like(phasor),
        spectral_loss=frozendict({"obs": jnp.zeros(())}),
        loss=frozendict({"obs": jnp.zeros(())}),
        predicted_distributions=frozendict({"obs": dist.to_exp()}),
        particle_state=target_node.initial_particle_state(),
    )
    assert jnp.allclose(config.total_loss(), 0.0)


def test_phasor_target_node_returns_configuration(
    target_node: PhasorTargetNode, freqs: jnp.ndarray
) -> None:
    z_hat = phasor_from_distribution(_PRIOR, freqs)
    out = infer_target_node(target_node, {"obs": _PRIOR}, {"obs": z_hat})
    assert isinstance(out, PhasorTargetConfiguration)


def test_phasor_target_node_score_is_array(
    target_node: PhasorTargetNode, freqs: jnp.ndarray
) -> None:
    z_hat = phasor_from_distribution(_PRIOR, freqs)
    out = infer_target_node(target_node, {"obs": _PRIOR}, {"obs": z_hat})
    assert out.score.shape == z_hat.shape


def test_phasor_target_node_predicted_distribution_recovers_mean(
    target_node: PhasorTargetNode, freqs: jnp.ndarray
) -> None:
    mu = 0.5
    dist = UnitVarianceNormalNP(jnp.array(mu))
    z_hat = phasor_from_distribution(dist, freqs)
    out = infer_target_node(target_node, {"obs": dist}, {"obs": z_hat})
    pred = out.predicted_distributions["obs"]
    assert isinstance(pred, UnitVarianceNormalEP)
    assert jnp.allclose(pred.mean, jnp.array(mu), atol=1e-4)  # type: ignore[unresolved-attribute]


def test_phasor_target_node_total_loss_is_sum_of_field_losses(
    freqs: jnp.ndarray,
) -> None:
    node = PhasorTargetNode.create(
        {
            "x": UnitVarianceNormalNP(jnp.array(0.0)),
            "y": UnitVarianceNormalNP(jnp.array(1.0)),
        },
        freqs,
    )
    x_dist = UnitVarianceNormalNP(jnp.array(0.1))
    y_dist = UnitVarianceNormalNP(jnp.array(0.7))
    out = infer_target_node(
        node,
        {"x": x_dist, "y": y_dist},
        {
            "x": phasor_from_distribution(x_dist, freqs),
            "y": phasor_from_distribution(y_dist, freqs),
        },
    )
    assert jnp.allclose(out.total_loss(), jnp.sum(out.loss["x"]) + jnp.sum(out.loss["y"]))


def test_phasor_target_node_multi_field_score_is_joined_on_last_dimension(
    freqs: jnp.ndarray,
) -> None:
    node = PhasorTargetNode.create(
        {
            "x": UnitVarianceNormalNP(jnp.array(0.0)),
            "y": UnitVarianceNormalNP(jnp.array(1.0)),
        },
        freqs,
    )
    x_phasor = phasor_from_distribution(UnitVarianceNormalNP(jnp.array(0.1)), freqs)
    y_phasor = phasor_from_distribution(UnitVarianceNormalNP(jnp.array(0.7)), freqs)
    out = infer_target_node(
        node,
        {
            "x": UnitVarianceNormalNP(jnp.array(0.1)),
            "y": UnitVarianceNormalNP(jnp.array(0.7)),
        },
        {"x": x_phasor, "y": y_phasor},
    )
    assert out.score.shape == (x_phasor.shape[-1] + y_phasor.shape[-1],)


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


def test_parameter_type_partition_round_trip_preserves_type() -> None:
    parameter_type = ParameterType(LearnableParameter)
    extracted, remainder = eqx.partition(parameter_type, lambda x: isinstance(x, type))
    round_tripped = eqx.combine(extracted, remainder)

    # t is static, so it does not appear as a dynamic pytree leaf.
    assert tree.leaves(extracted) == []
    assert round_tripped.t is LearnableParameter
