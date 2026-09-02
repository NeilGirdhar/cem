import jax.numpy as jnp
import numpy as np
from efax import Flattener, UnitVarianceNormalNP
from tjax import JaxArray, JaxRealArray

_SEMICIRCLE_LIMIT = jnp.pi / 2


def semicircle_observation_phase(values: JaxRealArray) -> JaxRealArray:
    """Map real observations monotonically into the open phase semicircle."""
    return _SEMICIRCLE_LIMIT * jnp.tanh(values)


def inverse_semicircle_observation_phase(phases: JaxRealArray) -> JaxRealArray:
    """Invert :func:`semicircle_observation_phase` within its open range."""
    epsilon = jnp.finfo(phases.dtype).eps
    normalized = jnp.clip(phases / _SEMICIRCLE_LIMIT, -1 + epsilon, 1 - epsilon)
    return jnp.arctanh(normalized)


def encode_observation_phasors(
    presences: JaxRealArray,
    values: JaxRealArray,
) -> JaxArray:
    """Encode scalar presences and values with the temporary semicircle map."""
    return presences * jnp.exp(1j * semicircle_observation_phase(values))


def decode_observation_phasors(phasors: JaxArray) -> JaxRealArray:
    """Decode values from phasor phases under the temporary semicircle map."""
    return inverse_semicircle_observation_phase(jnp.angle(phasors))


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
