from __future__ import annotations

import equinox as eqx
import rich.progress as rp

from .wandb_tools import WAndBInitSettings

from .telemetry import Telemetries


class ExecutionPacket(eqx.Module):
    progress_manager: rp.Progress | None = None
    telemetries: Telemetries = eqx.field(default_factory=Telemetries)
    wandb_settings: WAndBInitSettings | None = None
    enable_profiling: bool = False
    wandb_log_period: int = 1  # Training example period between log events to WandB.
