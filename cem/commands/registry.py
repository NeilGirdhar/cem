"""Experiments available from the command line, indexed by name."""

from cem.framework.plotter import Experiment
from cem.npn.experiments.supervised.calibration import supervised_calibration_experiment
from cem.npn.experiments.supervised.missingness import supervised_missingness_experiment
from cem.sip.experiments.instruments import (
    sip_exclusion_repair_experiment,
    sip_inherited_instrument_experiment,
    sip_instrument_depth_experiment,
)
from cem.sip.experiments.prediction import sip_causal_mean_experiment

experiments: dict[str, Experiment] = {
    experiment.name: experiment
    for experiment in [
        supervised_missingness_experiment,
        supervised_calibration_experiment,
        sip_inherited_instrument_experiment,
        sip_instrument_depth_experiment,
        sip_exclusion_repair_experiment,
        sip_causal_mean_experiment,
    ]
}
