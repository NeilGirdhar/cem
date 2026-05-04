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
from cem.phasor.message import (
    combine_phasors,
    encode_scalar_phasors,
    phasor_concordance,
    phasor_dropout,
    phasor_from_distribution,
    phasor_from_polar,
    phasor_presence,
    phasor_to_conjugate_prior,
    phasor_to_distribution,
    phasor_to_real,
    phasor_value,
    rotate_phasors,
    scale_phasors,
    split_phasor_frequencies,
    zero_phasors,
)
from cem.phasor.target_node import PhasorTargetConfiguration, PhasorTargetNode
from cem.phasor.telemetry import SpectralLossTelemetry

__all__ = [
    "Accumulator",
    "GatedProjection",
    "LogSpaceProjection",
    "LogSpaceProjectionWithDropout",
    "LossAndScore",
    "PhasorInputConfiguration",
    "PhasorTargetConfiguration",
    "PhasorTargetNode",
    "SpectralLossTelemetry",
    "centering_loss",
    "combine_phasors",
    "decorrelation_loss",
    "encode_scalar_phasors",
    "geometric_frequencies",
    "interpolate",
    "make_frequency_grid",
    "phasor_concordance",
    "phasor_dropout",
    "phasor_from_distribution",
    "phasor_from_polar",
    "phasor_gate",
    "phasor_presence",
    "phasor_to_conjugate_prior",
    "phasor_to_distribution",
    "phasor_to_real",
    "phasor_value",
    "rotate_by_location",
    "rotate_phasors",
    "scale_phasors",
    "select",
    "spectral_reconstruction_loss",
    "spectral_reconstruction_loss_and_score",
    "split_phasor_frequencies",
    "strength_loss",
    "zero_phasors",
]
