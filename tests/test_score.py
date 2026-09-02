import equinox as eqx
import jax
import jax.numpy as jnp
from efax import ComplexVonMisesNP, Flattener, NormalEP, NormalNP
from jax import tree
from tjax import frozendict

from cem.perceptron.target_node import PerceptronTargetConfiguration, PerceptronTargetNode
from cem.phasor.loss import LossAndScore, phasor_reconstruction_loss_and_score
from cem.structure.graph import LearnableParameter, ParameterType


def test_reconstruction_loss_and_score_returns_loss_and_score() -> None:
    observed = jnp.array([1 + 0j, 0 + 1j])
    assert isinstance(
        phasor_reconstruction_loss_and_score(observed, observed),
        LossAndScore,
    )


def test_reconstruction_loss_and_score_shapes() -> None:
    observed = jnp.ones((3, 4), dtype=jnp.complex128)
    prediction = jnp.full((3, 4), 0.5 + 0.5j, dtype=jnp.complex128)
    result = phasor_reconstruction_loss_and_score(observed, prediction)
    assert result.score.shape == prediction.shape
    assert result.loss.shape == (3,)
    assert result.total_loss().shape == ()


def test_reconstruction_loss_matches_von_mises_kl() -> None:
    observed = jnp.array([1 + 0j, 0.5 + 0.5j])
    prediction = jnp.array([0.8 + 0.2j, 0.3 - 0.3j])
    result = phasor_reconstruction_loss_and_score(observed, prediction)
    observed_dist = ComplexVonMisesNP(observed)
    predicted_dist = ComplexVonMisesNP(prediction)
    elementwise_loss = observed_dist.to_exp().kl_divergence(
        predicted_dist,
        self_nat=observed_dist,
    )
    assert jnp.allclose(result.loss, jnp.mean(elementwise_loss))


def test_reconstruction_score_is_zero_at_observation() -> None:
    observed = jnp.array([1 + 0j, 0 + 1j, 0.5 - 0.5j])
    result = phasor_reconstruction_loss_and_score(observed, observed)
    assert jnp.allclose(result.score, 0.0, atol=1e-6)


def test_reconstruction_score_equals_gradient() -> None:
    observed = jnp.array([1 + 0j, 0.5 + 0.5j])
    prediction = jnp.array([0.8 + 0.2j, 0.3 - 0.3j])
    result = phasor_reconstruction_loss_and_score(observed, prediction)

    def direct_loss(candidate: jnp.ndarray) -> jnp.ndarray:
        observed_dist = ComplexVonMisesNP(observed)
        predicted_dist = ComplexVonMisesNP(candidate)
        elementwise_loss = observed_dist.to_exp().kl_divergence(
            predicted_dist,
            self_nat=observed_dist,
        )
        return jnp.mean(elementwise_loss)

    assert jnp.allclose(result.score, jax.grad(direct_loss)(prediction))


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
    concat_prediction = jnp.concatenate(
        [predicted[field] for field in target_node.field_sizes],
        axis=-1,
    )
    return target_node.infer(flat_observed, concat_prediction)


def test_perceptron_target_node_partition_round_trip_preserves_behavior() -> None:
    dist = NormalNP(jnp.asarray(0.25), jnp.asarray(-0.5))
    _, prediction = Flattener.flatten(dist, mapped_to_plane=True)
    node = PerceptronTargetNode.create({"obs": dist})

    extracted, remainder = eqx.partition(node, eqx.is_array)
    round_tripped = eqx.combine(extracted, remainder)

    expected = infer_perceptron_target_node(node, {"obs": dist}, {"obs": prediction})
    result = infer_perceptron_target_node(
        round_tripped,
        {"obs": dist},
        {"obs": prediction},
    )
    assert jnp.allclose(result.total_loss(), expected.total_loss())
    predicted_dist = result.predicted_distributions["obs"]
    expected_dist = expected.predicted_distributions["obs"]
    assert isinstance(predicted_dist, NormalEP)
    assert isinstance(expected_dist, NormalEP)
    assert jnp.allclose(predicted_dist.mean, expected_dist.mean)


def test_parameter_type_partition_round_trip_preserves_type() -> None:
    parameter_type = ParameterType(LearnableParameter)
    extracted, remainder = eqx.partition(parameter_type, lambda x: isinstance(x, type))
    round_tripped = eqx.combine(extracted, remainder)

    assert tree.leaves(extracted) == []
    assert round_tripped.t is LearnableParameter
