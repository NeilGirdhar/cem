from __future__ import annotations

from collections.abc import Mapping

import jax.numpy as jnp
import jax.random as jr
from tjax import RngStream

from cem.demos.supervised.problem import load_synthetic_regression
from cem.demos.supervised.solution import PhasorSupervisedModel
from cem.phasor import PhasorTargetConfiguration


def test_phasor_supervised_multi_target_infer_splits_target_fields(
    streams: Mapping[str, RngStream],
) -> None:
    problem = load_synthetic_regression()
    model = PhasorSupervisedModel.create(
        problem,
        n_frequencies=10,
        hidden_size=8,
        streams=streams,
    )
    observation = problem.create_data_source().initial_problem_state(jr.key(0))

    result = model.infer(observation, None, streams=streams, inference=False)
    config = result.configurations["target"]
    assert isinstance(config, PhasorTargetConfiguration)

    assert tuple(config.loss) == ("y_0", "y_1")
    assert config.score.data.shape == (problem.n_targets * 10,)
    assert jnp.isfinite(result.loss)
