"""Standardization and natural-parameter encoding for real-valued observations."""

import numpy as np
from efax import Flattener, UnitVarianceNormalNP
from tjax import JaxRealArray


def encode_flat(values: JaxRealArray) -> JaxRealArray:
    """Encode each entry of a real vector as a unit-variance Gaussian observation.

    Flatten the natural parameters in plane coordinates, preserving the vector's length.
    """
    assert values.ndim == 1
    dist = UnitVarianceNormalNP(values)
    _, flat = Flattener.flatten(dist, mapped_to_plane=True)
    return flat.reshape(-1)


def standardize_columns(values: np.ndarray) -> np.ndarray:
    """Centre numeric columns and scale nonconstant columns to unit variance."""
    values = np.asarray(values, dtype=np.float64)
    mean = values.mean(axis=0)
    std = values.std(axis=0)
    std = np.where(std == 0.0, 1.0, std)
    return ((values - mean) / std).astype(np.float64)
