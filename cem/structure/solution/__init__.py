"""Solution loops: training and inference execution, telemetry, and results collection."""

from .execution_context import ExecutionContext
from .execution_packet import ExecutionPacket
from .inference_context import infer_episodes
from .loss_telemetry import LossTelemetry
from .results import InferenceResults, TrainingResults
from .segment import segment_keys
from .telemetry import Telemetries, Telemetry
from .training_context import TrainingSegment, train_episodes
from .training_solution import TrainingSolution

__all__ = [
    "ExecutionContext",
    "ExecutionPacket",
    "InferenceResults",
    "LossTelemetry",
    "Telemetries",
    "Telemetry",
    "TrainingResults",
    "TrainingSegment",
    "TrainingSolution",
    "infer_episodes",
    "segment_keys",
    "train_episodes",
]
