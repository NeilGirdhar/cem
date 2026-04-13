"""Demo command group and registrations."""

from __future__ import annotations

from enum import Enum

from cem.demos.afp.demo import AFPDemo
from cem.demos.supervised.demo import SupervisedDemo
from cem.structure import Demo


class DemoEnum(Enum):
    supervised = "supervised"
    afp = "afp"


demo_registry: dict[DemoEnum, Demo] = {
    DemoEnum.supervised: SupervisedDemo(),
    DemoEnum.afp: AFPDemo(),
}
