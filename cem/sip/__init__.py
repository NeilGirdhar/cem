"""Self-instrumental purification components."""

from cem.sip.benchmark import (
    CausalBenchmarkResult,
    SIPBenchmarkResult,
    run_action_sensation_benchmark,
    run_confounded_action_benchmark,
    run_synthetic_sip_benchmark,
)
from cem.sip.emitter import EmitterOutput, SIPEmitter
from cem.sip.model import SIPChain, SIPChainOutput
from cem.sip.objectives import purification_loss, witness_loss
from cem.sip.score import ScoreOutput, SIPScore
from cem.sip.training import (
    SIPTrainingHistory,
    train_chain_adversarial,
    train_score_adversarial,
)

__all__ = [
    "CausalBenchmarkResult",
    "EmitterOutput",
    "SIPBenchmarkResult",
    "SIPChain",
    "SIPChainOutput",
    "SIPEmitter",
    "SIPScore",
    "SIPTrainingHistory",
    "ScoreOutput",
    "purification_loss",
    "run_action_sensation_benchmark",
    "run_confounded_action_benchmark",
    "run_synthetic_sip_benchmark",
    "train_chain_adversarial",
    "train_score_adversarial",
    "witness_loss",
]
