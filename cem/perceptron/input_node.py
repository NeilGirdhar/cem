from __future__ import annotations

from tjax import JaxRealArray, frozendict

from cem.structure.graph.node import NodeConfiguration


class PerceptronInputConfiguration(NodeConfiguration):
    """Holds flat distribution encodings for one perceptron inference step."""

    values: frozendict[str, JaxRealArray]
