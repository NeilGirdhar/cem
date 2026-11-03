"""Bike-sharing demand supervised demo."""

from cem.structure.plotter import Variant

from .demo import SupervisedDemo
from .models import DatasetKind, LinkKind, SupervisedSolver

supervised_bike_sharing_demand_demo = SupervisedDemo(
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
