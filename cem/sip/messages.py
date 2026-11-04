"""SIP-specific Gaussian message helpers."""

from efax import NormalNP
from tjax import JaxRealArray


def mean_and_precision(distribution: NormalNP) -> tuple[JaxRealArray, JaxRealArray]:
    """Extract the mean and precision from a Gaussian natural-parameter message."""
    precision = -2 * distribution.negative_half_precision
    mean = distribution.mean_times_precision / precision
    return mean, precision
