"""CPU-activity supervised demo."""

from .demo import SupervisedDemo, SupervisedVariant
from .models import DatasetKind, LinkKind

supervised_cpu_activity_demo = SupervisedDemo(
    name="supervised-cpu-activity",
    variants=[
        SupervisedVariant(dataset_kind=DatasetKind.cpu_activity, link_kind=LinkKind.perceptron),
        SupervisedVariant(
            dataset_kind=DatasetKind.cpu_activity,
            link_kind=LinkKind.natural_parameter,
        ),
    ],
)
