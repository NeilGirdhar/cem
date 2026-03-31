"""Demo command group and registrations."""

from __future__ import annotations

from enum import Enum
from typing import Any

from cem.structure import Demo


class DemoEnum(Enum):
    pass


demo_registry: dict[DemoEnum, Demo[Any]] = {}
