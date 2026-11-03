"""Experiment registry: single source of truth for the ``cem`` CLI.

Add a new experiment by inserting one entry into :data:`experiments`.
"""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import cem.npn.experiments.supervised.missingness as _missingness
import cem.sip.experiments.sip_identification as _sip_identification
from cem.npn.experiments.supervised.bike_sharing import supervised_bike_sharing_demand_demo
from cem.npn.experiments.supervised.cpu_activity import supervised_cpu_activity_demo
from cem.npn.experiments.supervised.elevators import supervised_elevators_demo
from cem.npn.experiments.supervised.iris import supervised_iris_demo
from cem.structure.plotter import Demo

from ._run import generate_from_demo


@dataclass(frozen=True)
class Experiment:
    """One entry in the experiment registry.

    Attributes:
        name: Identifier used on the command line and as the output file stem.
        generate: Returns the chart JSON the thesis consumes.
        demo: Optional :class:`Demo`. When set, enables ``cem optimize <name>``.
    """

    name: str
    generate: Callable[[], dict[str, Any]]
    demo: Demo | None = None


def _demo_experiment(demo: Demo) -> Experiment:
    return Experiment(name=demo.name, generate=lambda: generate_from_demo(demo), demo=demo)


def _run_missingness() -> dict[str, Any]:
    return _missingness.run_missingness_comparison()


def _run_sip_identification() -> dict[str, Any]:
    return _sip_identification.generate_sip_identification()


experiments: dict[str, Experiment] = {
    experiment.name: experiment
    for experiment in [
        _demo_experiment(supervised_iris_demo),
        _demo_experiment(supervised_bike_sharing_demand_demo),
        _demo_experiment(supervised_elevators_demo),
        _demo_experiment(supervised_cpu_activity_demo),
        Experiment(name="supervised-missingness", generate=_run_missingness),
        Experiment(name="sip-identification", generate=_run_sip_identification),
    ]
}
