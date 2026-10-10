from tjax import JaxRealArray

from cem.structure.graph.node import NodeConfiguration


class PerceptronInputConfiguration(NodeConfiguration):
    """Holds flat distribution encodings for one perceptron inference step."""

    values: frozendict[str, JaxRealArray]
