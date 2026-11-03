"""Iris supervised demo."""

from cem.structure.plotter import Variant

from .demo import SupervisedDemo
from .models import DatasetKind, LinkKind, SupervisedSolver

supervised_iris_demo = SupervisedDemo(
    name="supervised-iris",
    variants=[
        Variant(
            label="perceptron",
            solver=SupervisedSolver(dataset_kind=DatasetKind.iris, link_kind=LinkKind.perceptron),
        ),
        Variant(
            label="natural_parameter",
            solver=SupervisedSolver(
                dataset_kind=DatasetKind.iris,
                link_kind=LinkKind.natural_parameter,
            ),
        ),
    ],
)
