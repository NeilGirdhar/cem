"""Self-instrumental purification components."""

from cem.sip.benchmark import (
    CausalBenchmarkResult,
    CausalBenchmarkTrajectory,
    SIPBenchmarkResult,
    run_direct_injection_benchmark,
    run_inherited_instrument_benchmark,
    run_intention_sensation_benchmark,
    run_synthetic_sip_benchmark,
)
from cem.sip.emitter import EmitterOutput, SIPEmitter
from cem.sip.explanatory_coupling import ExplanatoryCoupling, ExplanatoryCouplingOutput
from cem.sip.objectives import purification_loss, witness_loss
from cem.sip.score import ScoreOutput, SIPScore
from cem.sip.training import (
    SIPTrainingHistory,
    train_explanatory_coupling_adversarial,
    train_instrument_map,
    train_score_adversarial,
)

__all__ = [
    "CausalBenchmarkResult",
    "CausalBenchmarkTrajectory",
    "EmitterOutput",
    "ExplanatoryCoupling",
    "ExplanatoryCouplingOutput",
    "SIPBenchmarkResult",
    "SIPEmitter",
    "SIPScore",
    "SIPTrainingHistory",
    "ScoreOutput",
    "purification_loss",
    "run_direct_injection_benchmark",
    "run_inherited_instrument_benchmark",
    "run_intention_sensation_benchmark",
    "run_synthetic_sip_benchmark",
    "train_explanatory_coupling_adversarial",
    "train_instrument_map",
    "train_score_adversarial",
    "witness_loss",
]
