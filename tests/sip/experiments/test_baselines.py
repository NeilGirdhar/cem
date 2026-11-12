"""Check the reference estimators on data with known effects."""

import numpy as np

from cem.sip.experiments.baselines import ordinary_least_squares, two_stage_least_squares


def test_ordinary_least_squares_ignores_the_intercept() -> None:
    regressor = np.asarray([-1.0, 0.0, 2.0, 5.0])
    assert np.isclose(ordinary_least_squares(regressor, 3.0 * regressor + 7.0), 3.0)


def test_two_stage_least_squares_with_one_instrument_is_the_covariance_ratio() -> None:
    instrument = np.asarray([0.0, 1.0, 3.0, 4.0])
    endogenous = np.asarray([1.0, 2.0, 2.0, 5.0])
    outcome = np.asarray([2.0, -1.0, 4.0, 3.0])
    expected = np.cov(instrument, outcome)[0, 1] / np.cov(instrument, endogenous)[0, 1]
    assert np.isclose(two_stage_least_squares(instrument, endogenous, outcome), expected)


def test_two_stage_least_squares_removes_confounding(np_rng: np.random.Generator) -> None:
    samples = 100_000
    instruments = np_rng.normal(size=(samples, 2))
    confounder = np_rng.normal(size=samples)
    endogenous = instruments @ np.asarray([1.0, 0.5]) + confounder
    outcome = 2.0 * endogenous - 3.0 * confounder + 0.1 * np_rng.normal(size=samples)

    assert abs(ordinary_least_squares(endogenous, outcome) - 2.0) > 1.0
    assert np.isclose(two_stage_least_squares(instruments, endogenous, outcome), 2.0, atol=0.03)
