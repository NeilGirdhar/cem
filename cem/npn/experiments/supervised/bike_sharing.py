"""Bike-sharing demand supervised experiment."""

from cem.structure.plotter import Variant

from .experiment import SupervisedExperiment
from .models import DatasetKind, LinkKind, SupervisedSolver

supervised_bike_sharing_demand_experiment = SupervisedExperiment(
    name="supervised-bike-sharing-demand",
    variants=[
        Variant(
            label="perceptron",
            solver=SupervisedSolver(
                dataset_kind=DatasetKind.bike_sharing_demand,
                link_kind=LinkKind.perceptron,
            ),
        ),
        Variant(
            label="natural_parameter",
            solver=SupervisedSolver(
                dataset_kind=DatasetKind.bike_sharing_demand,
                link_kind=LinkKind.natural_parameter,
            ),
        ),
    ],
)
