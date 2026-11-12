"""Reference estimators of a causal effect: ordinary and two-stage least squares."""

import numpy as np


def ordinary_least_squares(regressor: np.ndarray, outcome: np.ndarray) -> float:
    """Return the slope of the outcome on the regressor, with an intercept.

    Under confounding, this slope mixes the causal effect with the confounded correlation.
    """
    centred_regressor = regressor - regressor.mean()
    centred_outcome = outcome - outcome.mean()
    return float(
        np.vecdot(centred_regressor, centred_outcome)
        / np.vecdot(centred_regressor, centred_regressor)
    )


def two_stage_least_squares(
    instruments: np.ndarray, endogenous: np.ndarray, outcome: np.ndarray
) -> float:
    """Return the two-stage least-squares effect of the endogenous variable on the outcome.

    The first stage projects the endogenous variable onto the instruments, with an intercept. The
    effect is the slope of the outcome on that projection. Each instrument must be independent of
    the outcome's unobserved causes.

    Args:
        instruments: The instruments, shape (samples,) or (samples, instruments).
        endogenous: The endogenous variable, shape (samples,).
        outcome: The outcome, shape (samples,).
    """
    if instruments.ndim == 1:
        instruments = instruments[:, np.newaxis]
    centred_instruments = instruments - instruments.mean(axis=0)
    centred_endogenous = endogenous - endogenous.mean()
    centred_outcome = outcome - outcome.mean()
    coefficients, *_ = np.linalg.lstsq(centred_instruments, centred_endogenous)
    fitted = centred_instruments @ coefficients
    return float(np.vecdot(fitted, centred_outcome) / np.vecdot(fitted, centred_endogenous))
