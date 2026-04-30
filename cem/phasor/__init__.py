"""Phasor-space primitives, transforms, losses, and graph nodes."""

from cem.phasor.accumulator import Accumulator
from cem.phasor.attention import interpolate, select
from cem.phasor.frequency import geometric_frequencies, make_frequency_grid
from cem.phasor.gate import phasor_gate, rotate_by_location
from cem.phasor.gated_projection import GatedProjection
from cem.phasor.input_node import PhasorInputConfiguration
from cem.phasor.log_space_projection import LogSpaceProjection, LogSpaceProjectionWithDropout
from cem.phasor.loss import (
    LossAndScore,
    centering_loss,
    decorrelation_loss,
    spectral_reconstruction_loss,
    spectral_reconstruction_loss_and_score,
    strength_loss,
)
from cem.phasor.message import PhasorMessage
from cem.phasor.rivalry import RivalryGroups, RivalryNorm
from cem.phasor.target_node import PhasorTargetConfiguration, PhasorTargetNode

__all__ = [
    "Accumulator",
    "GatedProjection",
    "LogSpaceProjection",
    "LogSpaceProjectionWithDropout",
    "LossAndScore",
    "PhasorInputConfiguration",
    "PhasorMessage",
    "PhasorTargetConfiguration",
    "PhasorTargetNode",
    "RivalryGroups",
    "RivalryNorm",
    "centering_loss",
    "decorrelation_loss",
    "geometric_frequencies",
    "interpolate",
    "make_frequency_grid",
    "phasor_gate",
    "rotate_by_location",
    "select",
    "spectral_reconstruction_loss",
    "spectral_reconstruction_loss_and_score",
    "strength_loss",
]
