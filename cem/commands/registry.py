"""Experiments available from the command line, indexed by name."""

from cem.npn.experiments.supervised.calibration import supervised_calibration_experiment
from cem.npn.experiments.supervised.missingness import supervised_missingness_experiment
from cem.structure.plotter import Experiment

experiments: dict[str, Experiment] = {
    experiment.name: experiment
    for experiment in [
        supervised_missingness_experiment,
        supervised_calibration_experiment,
    ]
}
