"""Phasor-space primitives, transforms, losses, and graph nodes."""

from cem.phasor.accumulator import Accumulator
from cem.phasor.attention import interpolate, select
from cem.phasor.evidence_pooling import EvidencePooling, EvidencePoolingWithDropout
from cem.phasor.frequency import geometric_frequencies, make_frequency_grid
from cem.phasor.gate import phasor_gate, rotate_by_location
from cem.phasor.gated_projection import GatedProjection
from cem.phasor.input_node import PhasorInputConfiguration
from cem.phasor.log_space_projection import LogSpaceProjection, LogSpaceProjectionWithDropout
from cem.phasor.loss import (
    LossAndScore,
    centering_loss,
    decorrelation_loss,
    spectral_reconstruction_loss_and_score,
    strength_loss,
)
from cem.phasor.message import (
    encode_scalar_phasors,
    phasor_concordance,
    phasor_from_distribution,
    phasor_to_conjugate_prior,
    phasor_to_distribution,
    phasor_to_real,
)
from cem.phasor.mobius_summation import MobiusSummation, mobius_sum
from cem.phasor.particle import (
    ObservationParticleState,
    ParticleInference,
    initialize_particles,
    observation_cross_entropy,
    refine_best_particle,
    update_particles,
)
from cem.phasor.target_node import PhasorTargetConfiguration, PhasorTargetNode
from cem.phasor.telemetry import SpectralLossTelemetry
from cem.phasor.value_projection import ValueProjection

__all__ = [
    "Accumulator",
    "EvidencePooling",
    "EvidencePoolingWithDropout",
    "GatedProjection",
    "LogSpaceProjection",
    "LogSpaceProjectionWithDropout",
    "LossAndScore",
    "MobiusSummation",
    "ObservationParticleState",
    "ParticleInference",
    "PhasorInputConfiguration",
    "PhasorTargetConfiguration",
    "PhasorTargetNode",
    "SpectralLossTelemetry",
    "ValueProjection",
    "centering_loss",
    "decorrelation_loss",
    "encode_scalar_phasors",
    "geometric_frequencies",
    "initialize_particles",
    "interpolate",
    "make_frequency_grid",
    "mobius_sum",
    "observation_cross_entropy",
    "phasor_concordance",
    "phasor_from_distribution",
    "phasor_gate",
    "phasor_to_conjugate_prior",
    "phasor_to_distribution",
    "phasor_to_real",
    "refine_best_particle",
    "rotate_by_location",
    "select",
    "spectral_reconstruction_loss_and_score",
    "strength_loss",
    "update_particles",
]
