"""Reference estimators of causal effects: ordinary and two-stage least squares."""

import numpy as np


def _centred_columns(values: np.ndarray) -> np.ndarray:
    """Return the values as columns, one for each variable, with their means removed."""
    if values.ndim == 1:
        values = values[:, np.newaxis]
    return values - values.mean(axis=0)


def ordinary_least_squares_coefficients(regressors: np.ndarray, outcome: np.ndarray) -> np.ndarray:
    """Return the coefficients of the outcome on the regressors, with an intercept.

    Under confounding, each coefficient mixes a causal effect with confounded correlation.

    Args:
        regressors: The regressors, shape (samples,) or (samples, regressors).
        outcome: The outcome, shape (samples,).
    """
    coefficients, *_ = np.linalg.lstsq(_centred_columns(regressors), outcome - outcome.mean())
    return coefficients


def ordinary_least_squares(regressor: np.ndarray, outcome: np.ndarray) -> float:
    """Return the slope of the outcome on one regressor, with an intercept."""
    return float(ordinary_least_squares_coefficients(regressor, outcome)[0])


def two_stage_least_squares_coefficients(
    instruments: np.ndarray, regressors: np.ndarray, outcome: np.ndarray
) -> np.ndarray:
    """Return the two-stage least-squares effects of the regressors on the outcome.

    The first stage projects the regressors onto the instruments, with an intercept. The effects
    are the coefficients of the outcome on those projections. Each instrument must be independent
    of the outcome's unobserved causes and affect the outcome only through the regressors. The
    instruments must vary each regressor independently of the others.

    Args:
        instruments: The instruments, shape (samples,) or (samples, instruments).
        regressors: The regressors, shape (samples,) or (samples, regressors).
        outcome: The outcome, shape (samples,).
    """
    centred_instruments = _centred_columns(instruments)
    centred_regressors = _centred_columns(regressors)
    first_stage, *_ = np.linalg.lstsq(centred_instruments, centred_regressors)
    fitted = centred_instruments @ first_stage
    return np.linalg.solve(fitted.T @ centred_regressors, fitted.T @ (outcome - outcome.mean()))


def two_stage_least_squares(
    instruments: np.ndarray, endogenous: np.ndarray, outcome: np.ndarray
) -> float:
    """Return the two-stage least-squares effect of one endogenous variable on the outcome."""
    return float(two_stage_least_squares_coefficients(instruments, endogenous, outcome)[0])
