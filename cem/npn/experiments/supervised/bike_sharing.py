"""Bike-sharing demand supervised demo."""

from .demo import SupervisedDemo, SupervisedVariant
from .models import DatasetKind, LinkKind

supervised_bike_sharing_demand_demo = SupervisedDemo(
    name="supervised-bike-sharing-demand",
    variants=[
        SupervisedVariant(
            dataset_kind=DatasetKind.bike_sharing_demand,
            link_kind=LinkKind.perceptron,
        ),
        SupervisedVariant(
            dataset_kind=DatasetKind.bike_sharing_demand,
            link_kind=LinkKind.natural_parameter,
        ),
    ],
)
