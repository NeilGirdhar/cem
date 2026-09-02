"""Phasor-space primitives, transforms, losses, and graph nodes."""

from cem.phasor.accumulator import Accumulator
from cem.phasor.attention import interpolate, select
from cem.phasor.elementwise_rotation import ElementwiseRotation, FrequencyElementwiseRotation
from cem.phasor.evidence_pooling import (
    EvidencePooling,
    EvidencePoolingWithDropout,
    FrequencyAdaptedEvidencePooling,
    LowRankFrequencyAdaptedEvidencePooling,
)
from cem.phasor.frequency import geometric_frequencies, make_frequency_grid
from cem.phasor.frequency_gated_projection import FrequencyGatedProjection
from cem.phasor.frequency_mobius_projection import (
    FrequencyMobiusProjection,
    MobiusParameterization,
)
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
from cem.phasor.mobius_summation import (
    LowRankMobiusSummation,
    MobiusSummation,
    mobius_sum,
    phase_warp,
)
from cem.phasor.particle import (
    ObservationParticleState,
    ParticleInference,
    initialize_particles,
    observation_cross_entropy,
    refine_best_particle,
    update_particles,
)
from cem.phasor.phase_activation import FrequencyPhaseActivation
from cem.phasor.recurrent_phase_focusing import RecurrentPhaseFocusing
from cem.phasor.target_node import PhasorTargetConfiguration, PhasorTargetNode
from cem.phasor.telemetry import SpectralLossTelemetry
from cem.phasor.value_projection import ValueProjection

__all__ = [
    "Accumulator",
    "ElementwiseRotation",
    "EvidencePooling",
    "EvidencePoolingWithDropout",
    "FrequencyAdaptedEvidencePooling",
    "FrequencyElementwiseRotation",
    "FrequencyGatedProjection",
    "FrequencyMobiusProjection",
    "FrequencyPhaseActivation",
    "GatedProjection",
    "LogSpaceProjection",
    "LogSpaceProjectionWithDropout",
    "LossAndScore",
    "LowRankFrequencyAdaptedEvidencePooling",
    "LowRankMobiusSummation",
    "MobiusParameterization",
    "MobiusSummation",
    "ObservationParticleState",
    "ParticleInference",
    "PhasorInputConfiguration",
    "PhasorTargetConfiguration",
    "PhasorTargetNode",
    "RecurrentPhaseFocusing",
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
    "phase_warp",
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
