from __future__ import annotations

import numpy as np
from efax import Flattener, UnitVarianceNormalNP
from tjax import JaxRealArray


def encode_flat(values: JaxRealArray) -> JaxRealArray:
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


def standardize_columns(values: np.ndarray) -> np.ndarray:
    """Center and scale each numeric column to unit variance."""
    values = np.asarray(values, dtype=np.float64)
    mean = values.mean(axis=0)
    std = values.std(axis=0)
    std = np.where(std == 0.0, 1.0, std)
    return ((values - mean) / std).astype(np.float64)
