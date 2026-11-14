"""Construction settings for a SIP agent and its circuits."""

import math
from dataclasses import dataclass, field


@dataclass(frozen=True)
class CreditSettings:
    """The decay rate for all eligibility traces in a credit circuit."""

    trace_decay_rate: float

    def __post_init__(self) -> None:
        if not math.isfinite(self.trace_decay_rate) or self.trace_decay_rate <= 0:
            msg = "trace_decay_rate must be finite and positive"
            raise ValueError(msg)


@dataclass(frozen=True)
class MismatchSettings:
    """Hidden-layer widths for a mismatch circuit's maps and the weight of its proportional loss.

    A damping weight of zero removes the proportional loss from the prediction's update, which
    leaves integral-only purification. The proportional witness still trains on its own loss.
    """

    prediction_hidden_features: int | tuple[int, ...] = ()
    integral_witness_hidden_features: int | tuple[int, ...] = ()
    proportional_witness_hidden_features: int | tuple[int, ...] = ()
    damping_weight: float = 1.0

    def __post_init__(self) -> None:
        if not math.isfinite(self.damping_weight) or self.damping_weight < 0:
            msg = "damping_weight must be finite and nonnegative"
            raise ValueError(msg)


@dataclass(frozen=True)
class TDSettings:
    """The TD discount, hidden-layer widths for the circuit's maps, and its damping weight."""

    discount: float = 1.0
    prediction_hidden_features: int | tuple[int, ...] = ()
    integral_witness_hidden_features: int | tuple[int, ...] = ()
    proportional_witness_hidden_features: int | tuple[int, ...] = ()
    damping_weight: float = 1.0

    def __post_init__(self) -> None:
        if not math.isfinite(self.damping_weight) or self.damping_weight <= 0:
            msg = "damping_weight must be finite and positive"
            raise ValueError(msg)


@dataclass(frozen=True, kw_only=True)
class EnvironmentFeatures:
    """Widths of the observation, action, and reward at the environment boundary."""

    observation_features: int
    action_features: int
    reward_features: int = 1


@dataclass(frozen=True, kw_only=True)
class SIPAgentSettings:
    """Environment widths, internal feature widths, and circuit settings for a SIP agent."""

    environment: EnvironmentFeatures
    level_sizes: tuple[int, ...]
    prospect_sizes: tuple[int, ...] | None = None
    credit_settings: CreditSettings
    mismatch_settings: MismatchSettings = field(default_factory=MismatchSettings)
    td_settings: TDSettings = field(default_factory=TDSettings)
