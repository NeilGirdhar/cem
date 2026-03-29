from typing import Any

import equinox as eqx

from .inference import SolutionState
from .telemetry import Telemetry


class TrainingResults(eqx.Module):
    count: int
    telemetries: dict[Telemetry, Any]  # This is over the entire data set.
    final_state: SolutionState


class InferenceResults(eqx.Module):
    count: int
    telemetries: dict[Telemetry, Any]  # This is over the entire data set.
