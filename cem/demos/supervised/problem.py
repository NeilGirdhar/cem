"""Supervised learning problem: data sources and problem state."""

from functools import cache
from typing import Any, cast, override

import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pandas as pd
from datasets import load_dataset
from efax import UnitVarianceNormalNP
from tjax import JaxRealArray, KeyArray

from cem.structure.problem import Problem
from cem.structure.problem.data_source import DataSource, ProblemState
from cem.transforms import encode_flat

_INFERENCE_FRACTION = 0.2
_MIN_PARTITIONED_ROWS = 2


class SupervisedProblemState(ProblemState):
    """One supervised example: flat natural-param encodings of x and y.

    ``x`` and ``y`` are flat real arrays in ``mapped_to_plane=True`` coordinates,
    shaped ``(2 * n_features,)`` and ``(2 * n_targets,)`` respectively.  Field
    names match input node field names so that ``as_shallow_dict`` routes them
    correctly.
    """

    x: JaxRealArray
    y: JaxRealArray


class SupervisedDataSource(DataSource):
    """Data source for a supervised learning problem.

    Stores pre-encoded flat arrays for all examples and draws one uniformly at
    random, with replacement, per call to :meth:`initial_problem_state`.  The
    same example key always selects the same row.  Consequently, solvers that
    share a partition and example-key schedule receive the same sequence of rows.

    Attributes:
        x_flat: Shape ``(n_samples, n_features)``.
        y_flat: Shape ``(n_samples, n_targets)``.
    """

    x_flat: JaxRealArray  # (n_samples, n_features)
    y_flat: JaxRealArray  # (n_samples, n_targets)

    def initial_problem_state(self, example_key: KeyArray) -> SupervisedProblemState:
        n = self.x_flat.shape[0]
        idx = jr.randint(example_key, shape=(), minval=0, maxval=n)
        return SupervisedProblemState(x=self.x_flat[idx], y=self.y_flat[idx])


class SupervisedProblem(Problem):
    """Complete supervised learning dataset with priors.

    Attributes:
        training: Encoded examples used for parameter updates.
        inference: Held-out encoded examples used for inference.
        x_prior: UnitVarianceNormalNP prior for the input features (used to configure the input
            node).
        y_prior: UnitVarianceNormalNP prior for the targets (used to configure the target node).
        n_features: Number of input features.
        n_targets: Number of target dimensions.
    """

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
    """Standardize both partitions with training statistics and encode them.

    Args:
        training_x: Training feature matrix.
        training_y: Training target matrix.
        inference_x: Held-out feature matrix.
        inference_y: Held-out target matrix.

    Returns:
        Training and inference data sources, priors, and feature counts.
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

    # Vectorise over samples.
    from jax import vmap  # ruff:ignore[import-outside-top-level]

    training_x_flat = vmap(encode_flat)(training_x_jax)
    training_y_flat = vmap(encode_flat)(training_y_jax)
    inference_x_flat = vmap(encode_flat)(inference_x_jax)
    inference_y_flat = vmap(encode_flat)(inference_y_jax)

    # Priors: zero-mean, unit variance.
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
    """Build deterministic training and inference partitions from a numeric table."""
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
    """Load a numeric Hugging Face tabular-regression config as a supervised problem."""
    ds = load_dataset("inria-soda/tabular-benchmark", config_name, split="train")
    df = cast("pd.DataFrame", ds.to_pandas())
    return problem_from_numeric_dataframe(df, max_rows=max_rows, seed=seed)


@cache
def load_iris() -> SupervisedProblem:
    """Load the Iris dataset from HuggingFace as a 4-feature -> 1-target problem.

    Features: sepal length, sepal width, petal length, petal width.
    Target: integer class label (0, 1, 2).

    Returns:
        A :class:`SupervisedProblem` with 150 samples, 4 features, 1 target.
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
