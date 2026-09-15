"""Self-instrumental purification components."""

from cem.sip.benchmark import (
    CausalBenchmarkResult,
    CausalBenchmarkTrajectory,
    CreditBenchmarkResult,
    SIPBenchmarkResult,
    run_direct_injection_benchmark,
    run_inherited_instrument_benchmark,
    run_intention_sensation_benchmark,
    run_synthetic_sip_benchmark,
    run_td_error_benchmark,
    simulate_remaining_food,
)
from cem.sip.emitter import EmitterOutput, SIPEmitter
from cem.sip.explanatory_coupling import ExplanatoryCoupling, ExplanatoryCouplingOutput
from cem.sip.objectives import purification_loss, witness_loss
from cem.sip.predictor_witness import PredictorWitnessPair
from cem.sip.score import ScoreOutput, SIPScore
from cem.sip.td_error import SIPTDError, TDErrorOutput
from cem.sip.training import (
    SIPTrainingHistory,
    rollout_td_error,
    train_explanatory_coupling_adversarial,
    train_instrument_map,
    train_score_adversarial,
    train_td_error_adversarial,
)

__all__ = [
    "CausalBenchmarkResult",
    "CausalBenchmarkTrajectory",
    "CreditBenchmarkResult",
    "EmitterOutput",
    "ExplanatoryCoupling",
    "ExplanatoryCouplingOutput",
    "PredictorWitnessPair",
    "SIPBenchmarkResult",
    "SIPEmitter",
    "SIPScore",
    "SIPTDError",
    "SIPTrainingHistory",
    "ScoreOutput",
    "TDErrorOutput",
    "purification_loss",
    "rollout_td_error",
    "run_direct_injection_benchmark",
    "run_inherited_instrument_benchmark",
    "run_intention_sensation_benchmark",
    "run_synthetic_sip_benchmark",
    "run_td_error_benchmark",
    "simulate_remaining_food",
    "train_explanatory_coupling_adversarial",
    "train_instrument_map",
    "train_score_adversarial",
    "train_td_error_adversarial",
    "witness_loss",
]
