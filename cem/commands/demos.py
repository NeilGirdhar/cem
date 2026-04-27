"""Demo registry."""

from __future__ import annotations

from enum import Enum

from cem.demos.afp.demo import afp_demo
from cem.demos.supervised.demo import (
    supervised_iris_demo,
    supervised_iris_spectral_demo,
    supervised_synthetic_regression_demo,
)
from cem.structure.plotter.demo import Demo


class DemoEnum(Enum):
    supervised_iris = "supervised-iris"
    supervised_iris_spectral = "supervised-iris-spectral"
    supervised_synthetic_regression = "supervised-synthetic-regression"
    afp = "afp"


demo_registry: dict[DemoEnum, Demo] = {
    DemoEnum.supervised_iris: supervised_iris_demo,
    DemoEnum.supervised_iris_spectral: supervised_iris_spectral_demo,
    DemoEnum.supervised_synthetic_regression: supervised_synthetic_regression_demo,
    DemoEnum.afp: afp_demo,
}
