"""Retired phasor operators retained for reference and comparison tests."""

from cem.experimental.phasor.obsolete.accumulator import Accumulator
from cem.experimental.phasor.obsolete.attention import interpolate, select
from cem.experimental.phasor.obsolete.log_space_projection import (
    LogSpaceProjection,
    LogSpaceProjectionWithDropout,
)
from cem.experimental.phasor.obsolete.value_projection import ValueProjection

__all__ = [
    "Accumulator",
    "LogSpaceProjection",
    "LogSpaceProjectionWithDropout",
    "ValueProjection",
    "interpolate",
    "select",
]
