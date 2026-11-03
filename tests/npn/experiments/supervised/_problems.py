"""Shared small-problem factories for supervised experiment tests."""

import numpy as np
import pandas as pd

from cem.npn.experiments.supervised.datasets import (
    SupervisedProblem,
    problem_from_numeric_dataframe,
)


def small_supervised_problem() -> SupervisedProblem:
    rng = np.random.default_rng(0)
    x = rng.normal(size=(32, 4))
    y = 0.5 * x[:, 0] - 0.25 * x[:, 1] + 0.1 * rng.normal(size=32)
    df = pd.DataFrame({f"x_{i}": x[:, i] for i in range(x.shape[1])})
    df["target"] = y
    return problem_from_numeric_dataframe(df, max_rows=16, seed=0)


def small_multi_target_problem() -> SupervisedProblem:
    rng = np.random.default_rng(1)
    x = rng.normal(size=(32, 4))
    df = pd.DataFrame({f"x_{i}": x[:, i] for i in range(x.shape[1])})
    df["target_0"] = 0.5 * x[:, 0] - 0.25 * x[:, 1]
    df["target_1"] = -0.3 * x[:, 2] + 0.2 * x[:, 3]
    return problem_from_numeric_dataframe(
        df,
        max_rows=None,
        seed=0,
        n_targets=2,
    )
