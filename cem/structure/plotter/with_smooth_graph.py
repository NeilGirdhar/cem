from __future__ import annotations

from dataclasses import KW_ONLY

import numpy as np
from scipy.signal import butter, sosfiltfilt
from scipy.stats import quantile
from tjax import NumpyArray, NumpyIntegralArray, NumpyRealArray

from .plotter import Plotter


class PlotterWithSmoothGraph(Plotter):
    _: KW_ONLY
    smoothing: float = 1
    clip_outlier_percentile: float = 100.0

    def _absolute_percentile(self, data: NumpyArray) -> float:
        """Return the percentile'th value in abs(data)."""
        return float(quantile(np.abs(data), self.clip_outlier_percentile * 0.01))

    def _smooth_data(
        self,
        values: NumpyIntegralArray | NumpyRealArray,
    ) -> NumpyRealArray:
        """Apply a Butterworth filter."""
        assert values.ndim == 1
        np_values = np.astype(values, np.float64)
        if self.smoothing <= 0.0 or values.shape[-1] < 16:  # noqa: PLR2004
            return np_values
        np_cutoff_frequency = np.asarray(np.exp(-self.smoothing), dtype=np.float64)
        sos = butter(4, np_cutoff_frequency, output="sos")
        return sosfiltfilt(sos, np_values)
