"""Check learnable parameter metadata and counts in real degrees of freedom."""

import jax.numpy as jnp
from jax import tree

from cem.structure.graph import (
    LossBalancedParameter,
    SummedGradientParameter,
    count_real_learnable_parameters,
)

_EXPECTED_REAL_PARAMETER_COUNT = 8


def test_loss_balanced_parameter_keeps_canonical_names_as_static_metadata() -> None:
    parameter = LossBalancedParameter(
        jnp.zeros(2),
        loss_names=("reconstruction", "confounding", "reconstruction"),
    )
    assert parameter.loss_names == ("confounding", "reconstruction")
    (leaf,) = tree.leaves(parameter)
    assert leaf is parameter.value


def test_real_parameter_count_counts_complex_values_twice() -> None:
    parameters = {
        "real": SummedGradientParameter(jnp.zeros(3)),
        "complex": SummedGradientParameter(jnp.zeros(2, dtype=jnp.complex64)),
        "balanced": LossBalancedParameter(
            jnp.zeros(1), loss_names=("confounding", "reconstruction")
        ),
    }

    assert count_real_learnable_parameters(parameters) == _EXPECTED_REAL_PARAMETER_COUNT
