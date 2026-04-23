"""Supervised learning problem: data sources and problem state."""

from __future__ import annotations

from functools import cache
from typing import Any, cast

import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from datasets import load_dataset
from efax import Flattener, UnitVarianceNormalNP
from sklearn.datasets import make_classification, make_regression
from sklearn.preprocessing import StandardScaler
from tjax import JaxRealArray, KeyArray

from cem.structure.problem import Problem
from cem.structure.problem.data_source import DataSource, ProblemState


class SupervisedProblemState(ProblemState):
    """One supervised example: flat natural-param encodings of x and y.

    ``x`` and ``y`` are flat real arrays in ``mapped_to_plane=True`` coordinates,
    shaped ``(2 * n_features,)`` and ``(2 * n_targets,)`` respectively.  Field
    names match input node field names so that ``as_shallow_dict`` routes them
    correctly.
    """

    x: JaxRealArray
    y: JaxRealArray


def _encode_flat(values: JaxRealArray) -> JaxRealArray:
    """Encode a real vector as flat natural params of UnitVarianceNormalNP.

    Each component x_i is encoded as UnitVarianceNormalNP(x_i) (unit variance),
    then flattened with ``mapped_to_plane=True``.  The resulting array has shape
    ``(n,)`` for input of shape ``(n,)``.

    Args:
        values: Shape ``(n,)`` real vector.

    Returns:
        Shape ``(n,)`` flat encoding.
    """
    assert values.ndim == 1
    dist = UnitVarianceNormalNP(values)
    _, flat = Flattener.flatten(dist, mapped_to_plane=True)
    return flat.reshape(-1)


class SupervisedDataSource(DataSource):
    """Data source for a supervised learning problem.

    Stores pre-encoded flat arrays for all examples and draws one uniformly at
    random per call to :meth:`initial_problem_state`.

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
        x_flat: Pre-encoded feature matrix, shape ``(n_samples, n_features)``.
        y_flat: Pre-encoded target matrix, shape ``(n_samples, n_targets)``.
        x_prior: UnitVarianceNormalNP prior for the input features (used to configure the input
            node).
        y_prior: UnitVarianceNormalNP prior for the targets (used to configure the target node).
        n_features: Number of input features.
        n_targets: Number of target dimensions.
    """

    x_flat: JaxRealArray
    y_flat: JaxRealArray
    x_prior: UnitVarianceNormalNP
    y_prior: UnitVarianceNormalNP
    n_features: int = eqx.field(static=True)
    n_targets: int = eqx.field(static=True)

    def create_data_source(self) -> SupervisedDataSource:
        return SupervisedDataSource(x_flat=self.x_flat, y_flat=self.y_flat)


def _encode_dataset(
    x: np.ndarray,
    y: np.ndarray,
) -> tuple[JaxRealArray, JaxRealArray, UnitVarianceNormalNP, UnitVarianceNormalNP, int, int]:
    """Standardise, encode, and return all components for a SupervisedProblem.

    Args:
        x: Feature matrix, shape ``(n_samples, n_features)``.
        y: Target matrix, shape ``(n_samples, n_targets)`` or ``(n_samples,)``.

    Returns:
        Tuple of ``(x_flat, y_flat, x_prior, y_prior, n_features, n_targets)``.
    """
    if y.ndim == 1:
        y = y[:, np.newaxis]
    x = StandardScaler().fit_transform(x).astype(np.float64)
    y = StandardScaler().fit_transform(y).astype(np.float64)

    n_features = x.shape[1]
    n_targets = y.shape[1]

    x_jax = jnp.asarray(x)
    y_jax = jnp.asarray(y)

    # Vectorise over samples.
    from jax import vmap  # noqa: PLC0415

    x_flat = vmap(_encode_flat)(x_jax)  # (n_samples, n_features)
    y_flat = vmap(_encode_flat)(y_jax)  # (n_samples, n_targets)

    # Priors: zero-mean, unit variance.
    x_prior = UnitVarianceNormalNP(jnp.zeros(n_features))
    y_prior = UnitVarianceNormalNP(jnp.zeros(n_targets))
    return x_flat, y_flat, x_prior, y_prior, n_features, n_targets


@cache
def load_iris() -> SupervisedProblem:
    """Load the Iris dataset from HuggingFace as a 4-feature → 1-target problem.

    Features: sepal length, sepal width, petal length, petal width.
    Target: integer class label (0, 1, 2).

    Returns:
        A :class:`SupervisedProblem` with 150 samples, 4 features, 1 target.
    """
    import pandas as pd  # noqa: PLC0415

    ds = load_dataset("scikit-learn/iris", split="train")
    df = cast("pd.DataFrame", ds.to_pandas())
    feature_cols = ["SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm"]
    x = df[feature_cols].to_numpy(dtype=np.float64)
    species_map: dict[Any, int] = {s: i for i, s in enumerate(df["Species"].unique())}
    y = df["Species"].map(species_map).to_numpy(dtype=np.float64)
    x_flat, y_flat, x_prior, y_prior, n_features, n_targets = _encode_dataset(x, y)
    return SupervisedProblem(
        x_flat=x_flat,
        y_flat=y_flat,
        x_prior=x_prior,
        y_prior=y_prior,
        n_features=n_features,
        n_targets=n_targets,
    )


@cache
def load_synthetic_regression(
    n_samples: int = 500,
    n_features: int = 8,
    n_targets: int = 2,
    *,
    seed: int = 0,
) -> SupervisedProblem:
    """Generate a synthetic regression dataset.

    Args:
        n_samples: Number of examples.
        n_features: Number of input features.
        n_targets: Number of regression targets.
        seed: Random seed for reproducibility.

    Returns:
        A :class:`SupervisedProblem`.
    """
    x, y = make_regression(
        n_samples=n_samples,
        n_features=n_features,
        n_targets=n_targets,
        noise=0.1,
        random_state=seed,
    )
    x_flat, y_flat, x_prior, y_prior, n_feats, n_tgts = _encode_dataset(x, y)
    return SupervisedProblem(
        x_flat=x_flat,
        y_flat=y_flat,
        x_prior=x_prior,
        y_prior=y_prior,
        n_features=n_feats,
        n_targets=n_tgts,
    )


@cache
def load_synthetic_classification(
    n_samples: int = 500,
    n_features: int = 8,
    *,
    seed: int = 0,
) -> SupervisedProblem:
    """Generate a synthetic binary classification dataset.

    Args:
        n_samples: Number of examples.
        n_features: Number of input features.
        seed: Random seed for reproducibility.

    Returns:
        A :class:`SupervisedProblem` with 1 binary target.
    """
    x, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        random_state=seed,
    )
    x_flat, y_flat, x_prior, y_prior, n_feats, n_tgts = _encode_dataset(x, y)
    return SupervisedProblem(
        x_flat=x_flat,
        y_flat=y_flat,
        x_prior=x_prior,
        y_prior=y_prior,
        n_features=n_feats,
        n_targets=n_tgts,
    )
