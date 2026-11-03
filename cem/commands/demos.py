"""Demo registry."""

from enum import Enum

from cem.npn.experiments.supervised.bike_sharing import supervised_bike_sharing_demand_demo
from cem.npn.experiments.supervised.cpu_activity import supervised_cpu_activity_demo
from cem.npn.experiments.supervised.elevators import supervised_elevators_demo
from cem.npn.experiments.supervised.iris import supervised_iris_demo
from cem.structure.plotter.demo import Demo


class DemoEnum(Enum):
    supervised_iris = "supervised-iris"
    supervised_bike_sharing_demand = "supervised-bike-sharing-demand"
    supervised_elevators = "supervised-elevators"
    supervised_cpu_activity = "supervised-cpu-activity"


demo_registry: dict[DemoEnum, Demo] = {
    DemoEnum.supervised_iris: supervised_iris_demo,
    DemoEnum.supervised_bike_sharing_demand: supervised_bike_sharing_demand_demo,
    DemoEnum.supervised_elevators: supervised_elevators_demo,
    DemoEnum.supervised_cpu_activity: supervised_cpu_activity_demo,
}
