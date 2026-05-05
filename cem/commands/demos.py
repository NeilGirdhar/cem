"""Demo registry."""

from __future__ import annotations

from enum import Enum

from cem.demos.afp.demo import afp_synthetic_iv_demo
from cem.demos.supervised.demo import (
    supervised_bike_sharing_demand_demo,
    supervised_cpu_activity_demo,
    supervised_elevators_demo,
    supervised_iris_demo,
)
from cem.structure.plotter.demo import Demo


class DemoEnum(Enum):
    supervised_iris = "supervised-iris"
    supervised_bike_sharing_demand = "supervised-bike-sharing-demand"
    supervised_elevators = "supervised-elevators"
    supervised_cpu_activity = "supervised-cpu-activity"
    afp_synthetic_iv = "afp-synthetic-iv"


demo_registry: dict[DemoEnum, Demo] = {
    DemoEnum.supervised_iris: supervised_iris_demo,
    DemoEnum.supervised_bike_sharing_demand: supervised_bike_sharing_demand_demo,
    DemoEnum.supervised_elevators: supervised_elevators_demo,
    DemoEnum.supervised_cpu_activity: supervised_cpu_activity_demo,
    DemoEnum.afp_synthetic_iv: afp_synthetic_iv_demo,
}
