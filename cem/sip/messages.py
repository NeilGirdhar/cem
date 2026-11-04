"""SIP-specific Gaussian message helpers."""

from efax import NormalNP
from tjax import JaxRealArray


def mean_and_precision(distribution: NormalNP) -> tuple[JaxRealArray, JaxRealArray]:
    """Extract the mean and precision from a Gaussian natural-parameter message."""
    precision = -2 * distribution.negative_half_precision
    mean = distribution.mean_times_precision / precision
    return mean, precision


def evidence_scale(message: NormalNP, gain: JaxRealArray) -> NormalNP:
    """Scale a Gaussian message's evidence by `gain`.

    Multiplies both natural parameters by `gain`. The mean is preserved and the
    precision is scaled by `gain` (@evidence-scaling-link in `thesis.typ`).
    """
    return NormalNP(
        mean_times_precision=gain * message.mean_times_precision,
        negative_half_precision=gain * message.negative_half_precision,
    )
