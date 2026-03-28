"""Demo command group and registrations."""

from __future__ import annotations

from enum import Enum
from typing import Any

from cem.demo import AFPIVDemo
from cem.structure import Demo


class DemoEnum(Enum):
    afp = "afp"


demo_registry: dict[DemoEnum, Demo[Any]] = {
    DemoEnum.afp: AFPIVDemo(),
}
