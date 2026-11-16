"""Natural-parameter encoding for real-valued observations."""

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
