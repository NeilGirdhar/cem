import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pandas as pd
import pytest

import cem.demos.supervised.problem as supervised_problem

from ._problems import small_supervised_problem

_EXPECTED_HF_TEST_FEATURES = 2
_EXPECTED_HF_SELECTED_ROWS = 5
_EXPECTED_HF_TRAINING_ROWS = 4
_EXPECTED_HF_INFERENCE_ROWS = 1


def test_hf_tabular_regression_dataframe_loader_is_deterministic() -> None:
    df = pd.DataFrame(
        {
            "feature_a": np.arange(10, dtype=np.float64),
            "feature_b": np.arange(10, dtype=np.float64) ** 2,
            "ignored": ["x"] * 10,
            "target": np.linspace(0.0, 1.0, 10),
        }
    )
    first = supervised_problem.problem_from_numeric_dataframe(
        df,
        max_rows=_EXPECTED_HF_SELECTED_ROWS,
        seed=7,
    )
    second = supervised_problem.problem_from_numeric_dataframe(
        df,
        max_rows=_EXPECTED_HF_SELECTED_ROWS,
        seed=7,
    )

    assert first.n_features == _EXPECTED_HF_TEST_FEATURES
    assert first.n_targets == 1
    assert first.training.x_flat.shape == (
        _EXPECTED_HF_TRAINING_ROWS,
        _EXPECTED_HF_TEST_FEATURES,
    )
    assert first.training.y_flat.shape == (_EXPECTED_HF_TRAINING_ROWS, 1)
    assert first.inference.x_flat.shape == (
        _EXPECTED_HF_INFERENCE_ROWS,
        _EXPECTED_HF_TEST_FEATURES,
    )
    assert first.inference.y_flat.shape == (_EXPECTED_HF_INFERENCE_ROWS, 1)
    assert jnp.all(jnp.isfinite(first.training.x_flat))
    assert jnp.all(jnp.isfinite(first.training.y_flat))
    assert jnp.all(jnp.isfinite(first.inference.x_flat))
    assert jnp.all(jnp.isfinite(first.inference.y_flat))
    assert jnp.allclose(first.training.x_flat, second.training.x_flat)
    assert jnp.allclose(first.training.y_flat, second.training.y_flat)
    assert jnp.allclose(first.inference.x_flat, second.inference.x_flat)
    assert jnp.allclose(first.inference.y_flat, second.inference.y_flat)
    assert not jnp.any(
        jnp.all(
            first.training.x_flat == first.inference.x_flat[0][jnp.newaxis, :],
            axis=-1,
        )
    )


def test_supervised_problem_selects_training_and_inference_sources() -> None:
    problem = small_supervised_problem()

    training = problem.create_data_source(inference=False)
    inference = problem.create_data_source(inference=True)

    assert training is problem.training
    assert inference is problem.inference


@pytest.mark.parametrize("source_name", ["training", "inference"])
def test_supervised_sources_reuse_rows_for_common_keys(source_name: str) -> None:
    first = getattr(small_supervised_problem(), source_name)
    second = getattr(small_supervised_problem(), source_name)
    keys = jr.split(jr.key(23), 16)

    first_states = [first.initial_problem_state(key) for key in keys]
    second_states = [second.initial_problem_state(key) for key in keys]

    for first_state, second_state in zip(first_states, second_states, strict=True):
        assert jnp.array_equal(first_state.x, second_state.x)
        assert jnp.array_equal(first_state.y, second_state.y)
