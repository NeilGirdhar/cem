import jax.numpy as jnp

from cem.structure.graph import LearnableParameter, MetaParameter, count_real_learnable_parameters

_EXPECTED_REAL_PARAMETER_COUNT = 7


def test_real_parameter_count_counts_complex_values_twice() -> None:
    parameters = {
        "real": LearnableParameter(jnp.zeros(3)),
        "complex": LearnableParameter(jnp.zeros(2, dtype=jnp.complex64)),
        "meta": MetaParameter(jnp.zeros(1)),
    }

    assert count_real_learnable_parameters(parameters) == _EXPECTED_REAL_PARAMETER_COUNT + 1
