"""Gaussian natural-parameter network primitives."""

from cem.npn.gaussian import (
    GaussianNPN,
    GaussianNPNInputPool,
    GaussianNPNLinear,
    gaussian_relu,
    normal_from_mean_and_presence,
)

__all__ = [
    "GaussianNPN",
    "GaussianNPNInputPool",
    "GaussianNPNLinear",
    "gaussian_relu",
    "normal_from_mean_and_presence",
]
