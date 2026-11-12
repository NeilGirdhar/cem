"""Aggregated telemetry and final training state returned by execution."""

from typing import Any

import equinox as eqx

from .inference import SolutionState
from .telemetry import Telemetry


class TrainingResults(eqx.Module):
    """Completed episode count, aggregated telemetry, and final parameter and optimizer state."""

    count: int
    telemetries: dict[Telemetry, Any]  # This is over the entire data set.
    final_state: SolutionState


class InferenceResults(eqx.Module):
    """Evaluated episode count and aggregated inference telemetry."""

    count: int
    telemetries: dict[Telemetry, Any]  # This is over the entire data set.
