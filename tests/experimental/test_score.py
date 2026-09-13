import equinox as eqx
import jax.numpy as jnp
from efax import (
    Flattener,
    NormalEP,
    NormalNP,
)
from jax import tree
from tjax import frozendict

from cem.perceptron.target_node import PerceptronTargetConfiguration, PerceptronTargetNode
from cem.structure.graph import LearnableParameter, ParameterType


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
