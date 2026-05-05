from __future__ import annotations

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
    """Apply robust LOWESS smoothing to a 1-D array.

    Args:
        values: Time-series values to smooth, shape ``(steps,)``.
        smoothing: LOWESS neighborhood fraction.  ``0`` disables smoothing; larger values
            smooth more aggressively.  Useful plot values are usually in ``[0.05, 0.3]``.
        log_space: Smooth ``log(values)`` and exponentiate back.  Use only for positive series.

    Returns:
        Smoothed values with the same shape as ``values``.
    """
    assert values.ndim == 1
    np_values = np.astype(values, np.float64)
    if smoothing <= 0.0 or values.shape[-1] < 3:  # noqa: PLR2004
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
    """Return the percentile-th value in abs(data)."""
    return float(quantile(np.abs(data), percentile * 0.01))


class PlotterWithSmoothGraph(Plotter):
    _: KW_ONLY
    smoothing: float = 0.15
    clip_outlier_percentile: float = 100.0
