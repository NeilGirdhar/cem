"""Iris supervised demo."""

from .demo import SupervisedDemo, SupervisedVariant
from .models import DatasetKind, LinkKind

supervised_iris_demo = SupervisedDemo(
    name="supervised-iris",
    variants=[
        SupervisedVariant(dataset_kind=DatasetKind.iris, link_kind=LinkKind.perceptron),
        SupervisedVariant(dataset_kind=DatasetKind.iris, link_kind=LinkKind.natural_parameter),
    ],
)
