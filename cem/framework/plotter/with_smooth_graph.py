"""Robust smoothing and magnitude percentiles for experiment plots."""

from dataclasses import KW_ONLY

import numpy as np
from scipy.stats import quantile
from tjax import NumpyArray, NumpyIntegralArray, NumpyRealArray

from .plotter import Plotter


def smooth_data(
    values: NumpyIntegralArray | NumpyRealArray,
    smoothing: float,
    *,
    log_space: bool = False,
) -> NumpyRealArray:
    """Smooth a one-dimensional series with robust locally weighted regression.

    The smoothing fraction sets the neighbourhood size. Nonpositive fractions and
    series shorter than three points return the original values as floating point.
    Use log-space smoothing only for positive series.
    """
    assert values.ndim == 1
    np_values = np.astype(values, np.float64)
    if smoothing <= 0.0 or values.shape[-1] < 3:  # ruff:ignore[magic-value-comparison]
        return np_values
    if log_space:
        if not np.all(np_values > 0.0):
            msg = "log-space smoothing requires positive values"
            raise ValueError(msg)
        return np.exp(_lowess(np.log(np_values), smoothing))
    return _lowess(np_values, smoothing)


def _lowess(
    values: NumpyRealArray,
    smoothing: float,
    *,
    robust_iterations: int = 2,
) -> NumpyRealArray:
    """Return robust locally linear LOWESS values for equally spaced samples."""
    n = values.shape[0]
    x = np.arange(n, dtype=np.float64)
    frac = float(np.clip(smoothing, 1.0 / n, 1.0))
    neighbors = max(2, int(np.ceil(frac * n)))
    robust_weights = np.ones(n, dtype=np.float64)
    fitted = np.empty_like(values)

    for iteration in range(robust_iterations + 1):
        for i, x_i in enumerate(x):
            distances = np.abs(x - x_i)
            bandwidth = np.partition(distances, neighbors - 1)[neighbors - 1]
            if bandwidth <= 0.0:
                bandwidth = np.max(distances)
            if bandwidth <= 0.0:
                fitted[i] = values[i]
                continue
            scaled = distances / bandwidth
            kernel_weights = np.where(scaled < 1.0, (1.0 - scaled**3) ** 3, 0.0)
            weights = kernel_weights * robust_weights
            sqrt_weights = np.sqrt(weights)
            design = np.column_stack((np.ones(n, dtype=np.float64), x - x_i))
            weighted_design = design * sqrt_weights[:, None]
            weighted_values = values * sqrt_weights
            fitted[i] = np.linalg.lstsq(weighted_design, weighted_values, rcond=None)[0][0]

        if iteration == robust_iterations:
            break
        residuals = values - fitted
        median_abs_residual = np.median(np.abs(residuals))
        if median_abs_residual <= 0.0:
            break
        scaled_residuals = residuals / (6.0 * median_abs_residual)
        robust_weights = np.where(
            np.abs(scaled_residuals) < 1.0,
            (1.0 - scaled_residuals**2) ** 2,
            0.0,
        )

    return fitted


def absolute_percentile(data: NumpyArray, percentile: float) -> float:
    """Return a percentile of the data's absolute values, on a scale from zero to one hundred."""
    return float(quantile(np.abs(data), percentile * 0.01))


class PlotterWithSmoothGraph(Plotter):
    """A plotter with a configurable smoothing fraction for its series."""

    _: KW_ONLY
    smoothing: float = 0.15
