import jax.numpy as jnp
import numpy as np
from efax import Flattener, NaturalParametrization, UnitVarianceNormalNP
from tjax import JaxArray, JaxRealArray

from cem.phasor.message import phasor_from_distribution


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


def encode_phasor(
    flat: JaxRealArray,
    flattener: Flattener[NaturalParametrization],
    frequencies: JaxRealArray,
) -> JaxArray:
    """Unflatten a flat UnitVarianceNormalNP encoding and return raveled phasors."""
    dist = flattener.unflatten(flat, raveled=True)
    return phasor_from_distribution(dist, frequencies, raveled=True)


def encode_semicircle_phasors(
    values: JaxRealArray,
    centres: JaxRealArray,
    angular_scales: JaxRealArray,
) -> JaxArray:
    """Encode real features as unit phasors in calibrated angular coordinates.

    ``centres`` and ``angular_scales`` define one affine phase coordinate per
    feature. Callers can keep a training domain inside the open right semicircle
    by choosing scales whose resulting phases lie in ``(-pi / 2, pi / 2)``.
    """
    if values.shape[-1] != centres.shape[0] or centres.shape != angular_scales.shape:
        msg = (
            "values, centres, and angular_scales must share their final feature "
            f"dimension, got {values.shape}, {centres.shape}, and {angular_scales.shape}"
        )
        raise ValueError(msg)
    return jnp.exp(1j * (values - centres) * angular_scales)
