"""Experiment registry: single source of truth for the ``cem`` CLI.

Add a new experiment by inserting one entry into :data:`experiments`.
"""

from cem.npn.experiments.supervised.bike_sharing import supervised_bike_sharing_demand_experiment
from cem.npn.experiments.supervised.cpu_activity import supervised_cpu_activity_experiment
from cem.npn.experiments.supervised.elevators import supervised_elevators_experiment
from cem.npn.experiments.supervised.iris import supervised_iris_experiment
from cem.npn.experiments.supervised.missingness import supervised_missingness_experiment
from cem.structure.plotter import Experiment

experiments: dict[str, Experiment] = {
    experiment.name: experiment
    for experiment in [
        supervised_iris_experiment,
        supervised_bike_sharing_demand_experiment,
        supervised_elevators_experiment,
        supervised_cpu_activity_experiment,
        supervised_missingness_experiment,
    ]
}
