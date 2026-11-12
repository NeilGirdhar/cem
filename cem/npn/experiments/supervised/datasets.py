"""Prepare numeric tables as training and held-out data for supervised models."""

from functools import cache
from typing import Any, cast, override

import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pandas as pd
from datasets import load_dataset
from efax import UnitVarianceNormalNP
from jax import vmap
from tjax import JaxRealArray, KeyArray

from cem.framework.problem import Problem
from cem.framework.problem.data_source import DataSource, ProblemState
from cem.transforms import encode_flat

_INFERENCE_FRACTION = 0.2
_MIN_PARTITIONED_ROWS = 2


class SupervisedProblemState(ProblemState):
    """One example with flat unit-variance Gaussian encodings for inputs and targets."""

    x: JaxRealArray
    y: JaxRealArray


class SupervisedDataSource(DataSource):
    """A partition of encoded examples that can be sampled with replacement."""

    x_flat: JaxRealArray  # (n_samples, n_features)
    y_flat: JaxRealArray  # (n_samples, n_targets)

    def initial_problem_state(self, example_key: KeyArray) -> SupervisedProblemState:
        """Select a row uniformly using the example key.

        Solvers using the same partition and key schedule see the same sampled rows.
        """
        n = self.x_flat.shape[0]
        idx = jr.randint(example_key, shape=(), minval=0, maxval=n)
        return SupervisedProblemState(x=self.x_flat[idx], y=self.y_flat[idx])


class SupervisedProblem(Problem):
    """Training and held-out partitions with priors and dimensions for configuring the model."""

    training: SupervisedDataSource
    inference: SupervisedDataSource
    x_prior: UnitVarianceNormalNP
    y_prior: UnitVarianceNormalNP
    n_features: int = eqx.field(static=True)
    n_targets: int = eqx.field(static=True)

    @override
    def create_data_source(self, *, inference: bool = False) -> SupervisedDataSource:
        return self.inference if inference else self.training


def _encode_dataset(
    training_x: np.ndarray,
    training_y: np.ndarray,
    inference_x: np.ndarray,
    inference_y: np.ndarray,
) -> tuple[
    SupervisedDataSource,
    SupervisedDataSource,
    UnitVarianceNormalNP,
    UnitVarianceNormalNP,
    int,
    int,
]:
    """Encode both partitions as unit-variance Gaussian observations.

    Use only training means and scales to standardize either partition. Return both
    data sources, the zero-mean priors, and the input and target widths.
    """
    if training_y.ndim == 1:
        training_y = training_y[:, np.newaxis]
    if inference_y.ndim == 1:
        inference_y = inference_y[:, np.newaxis]

    def standardize_from_training(
        training: np.ndarray,
        inference: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        mean = training.mean(axis=0)
        std = training.std(axis=0)
        std = np.where(std == 0.0, 1.0, std)
        return (training - mean) / std, (inference - mean) / std

    training_x, inference_x = standardize_from_training(training_x, inference_x)
    training_y, inference_y = standardize_from_training(training_y, inference_y)

    n_features = training_x.shape[1]
    n_targets = training_y.shape[1]

    training_x_jax = jnp.asarray(training_x)
    training_y_jax = jnp.asarray(training_y)
    inference_x_jax = jnp.asarray(inference_x)
    inference_y_jax = jnp.asarray(inference_y)

    training_x_flat = vmap(encode_flat)(training_x_jax)
    training_y_flat = vmap(encode_flat)(training_y_jax)
    inference_x_flat = vmap(encode_flat)(inference_x_jax)
    inference_y_flat = vmap(encode_flat)(inference_y_jax)

    x_prior = UnitVarianceNormalNP(jnp.zeros(n_features))
    y_prior = UnitVarianceNormalNP(jnp.zeros(n_targets))
    return (
        SupervisedDataSource(x_flat=training_x_flat, y_flat=training_y_flat),
        SupervisedDataSource(x_flat=inference_x_flat, y_flat=inference_y_flat),
        x_prior,
        y_prior,
        n_features,
        n_targets,
    )


def problem_from_numeric_dataframe(
    df: pd.DataFrame,
    *,
    max_rows: int | None,
    seed: int,
    n_targets: int = 1,
) -> SupervisedProblem:
    """Build reproducible training and held-out partitions from a numeric table.

    Drop nonnumeric columns and incomplete rows. Shuffle with the supplied seed
    before limiting the row count. Treat the last `n_targets` columns as targets.
    """
    numeric_df = df.select_dtypes(include=[np.number]).dropna()
    if n_targets < 1:
        msg = f"n_targets must be positive, got {n_targets}"
        raise ValueError(msg)
    if len(numeric_df.columns) <= n_targets:
        msg = "tabular regression data must contain at least one numeric feature and target"
        raise ValueError(msg)
    if len(numeric_df) < _MIN_PARTITIONED_ROWS:
        msg = "supervised data requires at least two rows"
        raise ValueError(msg)
    numeric_df = numeric_df.sample(frac=1.0, random_state=seed)
    if max_rows is not None:
        numeric_df = numeric_df.head(max_rows)
    if len(numeric_df) < _MIN_PARTITIONED_ROWS:
        msg = "supervised data requires at least two selected rows"
        raise ValueError(msg)
    inference_rows = max(1, round(len(numeric_df) * _INFERENCE_FRACTION))
    inference_df = numeric_df.iloc[:inference_rows]
    training_df = numeric_df.iloc[inference_rows:]
    training_x = training_df.iloc[:, :-n_targets].to_numpy(dtype=np.float64)
    training_y = training_df.iloc[:, -n_targets:].to_numpy(dtype=np.float64)
    inference_x = inference_df.iloc[:, :-n_targets].to_numpy(dtype=np.float64)
    inference_y = inference_df.iloc[:, -n_targets:].to_numpy(dtype=np.float64)
    (
        training,
        inference,
        x_prior,
        y_prior,
        n_features,
        n_targets,
    ) = _encode_dataset(training_x, training_y, inference_x, inference_y)
    return SupervisedProblem(
        training=training,
        inference=inference,
        x_prior=x_prior,
        y_prior=y_prior,
        n_features=n_features,
        n_targets=n_targets,
    )


@cache
def load_hf_tabular_regression(
    config_name: str,
    *,
    max_rows: int | None = 5000,
    seed: int = 0,
) -> SupervisedProblem:
    """Load a numeric regression configuration from the Hugging Face tabular benchmark."""
    ds = load_dataset("inria-soda/tabular-benchmark", config_name, split="train")
    df = cast("pd.DataFrame", ds.to_pandas())
    return problem_from_numeric_dataframe(df, max_rows=max_rows, seed=seed)


@cache
def load_iris() -> SupervisedProblem:
    """Load Iris as a regression problem with four measurements and a numeric species target.

    Assign consecutive integer labels to the species before splitting and standardizing.
    """
    ds = load_dataset(
        "scikit-learn/iris",
        split="train",
        revision="0bda0ce801be0fa2f464ff845a9d5ceae99aad7d",
    )
    df = cast("pd.DataFrame", ds.to_pandas())
    feature_cols = ["SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm"]
    x = df[feature_cols].to_numpy(dtype=np.float64)
    species_map: dict[Any, int] = {s: i for i, s in enumerate(df["Species"].unique())}
    y = df["Species"].map(species_map).to_numpy(dtype=np.float64)
    return problem_from_numeric_dataframe(
        pd.DataFrame(
            np.column_stack((x, y)),
            columns=[*feature_cols, "target"],
        ),
        max_rows=None,
        seed=0,
    )
