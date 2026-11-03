"""Elevators supervised demo."""

from .demo import SupervisedDemo, SupervisedVariant
from .models import DatasetKind, LinkKind

supervised_elevators_demo = SupervisedDemo(
    name="supervised-elevators",
    variants=[
        SupervisedVariant(dataset_kind=DatasetKind.elevators, link_kind=LinkKind.perceptron),
        SupervisedVariant(
            dataset_kind=DatasetKind.elevators,
            link_kind=LinkKind.natural_parameter,
        ),
    ],
)
