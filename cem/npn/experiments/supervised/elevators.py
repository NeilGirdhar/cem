"""Elevators supervised experiment."""

from cem.structure.plotter import Variant

from .experiment import SupervisedExperiment
from .models import DatasetKind, LinkKind, SupervisedSolver

supervised_elevators_experiment = SupervisedExperiment(
    name="supervised-elevators",
    variants=[
        Variant(
            label="perceptron",
            solver=SupervisedSolver(
                dataset_kind=DatasetKind.elevators,
                link_kind=LinkKind.perceptron,
            ),
        ),
        Variant(
            label="natural_parameter",
            solver=SupervisedSolver(
                dataset_kind=DatasetKind.elevators,
                link_kind=LinkKind.natural_parameter,
            ),
        ),
    ],
)
