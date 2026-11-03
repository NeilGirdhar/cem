"""CPU-activity supervised experiment."""

from cem.structure.plotter import Variant

from .experiment import SupervisedExperiment
from .models import DatasetKind, LinkKind, SupervisedSolver

supervised_cpu_activity_experiment = SupervisedExperiment(
    name="supervised-cpu-activity",
    variants=[
        Variant(
            label="perceptron",
            solver=SupervisedSolver(
                dataset_kind=DatasetKind.cpu_activity,
                link_kind=LinkKind.perceptron,
            ),
        ),
        Variant(
            label="natural_parameter",
            solver=SupervisedSolver(
                dataset_kind=DatasetKind.cpu_activity,
                link_kind=LinkKind.natural_parameter,
            ),
        ),
    ],
)
