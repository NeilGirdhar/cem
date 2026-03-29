"""Solution loops: training and inference execution, telemetry, and results collection."""

from .execution_context import ExecutionContext
from .execution_packet import ExecutionPacket
from .inference_context import infer_episodes
from .loss_telemetry import LossInferenceTelemetry, LossTrainingTelemetry
from .results import InferenceResults, TrainingResults
from .segment import segment_keys
from .telemetry import InferenceTelemetry, Telemetries, Telemetry, TrainingTelemetry
from .training_context import TrainingSegment, train_episodes
from .training_solution import TrainingSolution

__all__ = [
    "ExecutionContext",
    "ExecutionPacket",
    "InferenceResults",
    "InferenceTelemetry",
    "LossInferenceTelemetry",
    "LossTrainingTelemetry",
    "Telemetries",
    "Telemetry",
    "TrainingResults",
    "TrainingSegment",
    "TrainingSolution",
    "TrainingTelemetry",
    "infer_episodes",
    "segment_keys",
    "train_episodes",
]
