import numpy as np
import pytest

from cem.structure.plotter.with_smooth_graph import smooth_data


def test_smooth_data_disabled_returns_values() -> None:
    values = np.array([1.0, 3.0, 2.0])
    assert np.allclose(smooth_data(values, 0.0), values)


def test_smooth_data_positive_values_remain_positive_in_log_space() -> None:
    values = np.array([10.0, 8.0, 9.0, 80.0, 7.0, 5.0, 4.0, 3.0, 2.0, 1.0])
    smoothed = smooth_data(values, 0.3, log_space=True)
    assert smoothed.shape == values.shape
    assert np.all(np.isfinite(smoothed))
    assert np.all(smoothed > 0.0)


def test_smooth_data_log_space_requires_positive_values() -> None:
    values = np.array([1.0, 0.0, 2.0])
    with pytest.raises(ValueError, match="positive"):
        smooth_data(values, 0.3, log_space=True)
