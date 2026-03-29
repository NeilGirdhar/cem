from __future__ import annotations

from dataclasses import KW_ONLY

import numpy as np
from scipy.signal import butter, sosfiltfilt
from scipy.stats import quantile
from tjax import NumpyArray, NumpyIntegralArray, NumpyRealArray

from .plotter import Plotter


def smooth_data(
    values: NumpyIntegralArray | NumpyRealArray,
    smoothing: float,
) -> NumpyRealArray:
    """Apply a Butterworth low-pass filter to a 1-D array."""
    assert values.ndim == 1
    np_values = np.astype(values, np.float64)
    if smoothing <= 0.0 or values.shape[-1] < 16:  # noqa: PLR2004
        return np_values
    np_cutoff_frequency = np.asarray(np.exp(-smoothing), dtype=np.float64)
    sos = butter(4, np_cutoff_frequency, output="sos")
    return sosfiltfilt(sos, np_values)


def absolute_percentile(data: NumpyArray, percentile: float) -> float:
    """Return the percentile-th value in abs(data)."""
    return float(quantile(np.abs(data), percentile * 0.01))


class PlotterWithSmoothGraph(Plotter):
    _: KW_ONLY
    smoothing: float = 1
    clip_outlier_percentile: float = 100.0
