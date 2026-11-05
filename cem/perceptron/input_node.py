"""Perceptron node output for carrying flat distribution encodings."""

from tjax import JaxRealArray

from cem.structure.graph.node import NodeConfiguration


class PerceptronInputConfiguration(NodeConfiguration):
    """Flat encodings of the distributions supplied to a perceptron node, keyed by field."""

    values: frozendict[str, JaxRealArray]
