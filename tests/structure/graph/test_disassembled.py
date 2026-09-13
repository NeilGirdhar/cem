import equinox as eqx
from jax import tree

from cem.structure.graph import LearnableParameter, ParameterType


def test_parameter_type_partition_round_trip_preserves_type() -> None:
    parameter_type = ParameterType(LearnableParameter)
    extracted, remainder = eqx.partition(parameter_type, lambda x: isinstance(x, type))
    round_tripped = eqx.combine(extracted, remainder)

    assert tree.leaves(extracted) == []
    assert round_tripped.t is LearnableParameter
