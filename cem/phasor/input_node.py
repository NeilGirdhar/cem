from __future__ import annotations

from tjax import frozendict

from cem.phasor.message import JaxComplexArray
from cem.structure.graph.node import NodeConfiguration


class PhasorInputConfiguration(NodeConfiguration):
    """Holds phasor encodings for one phasor inference step."""

    values: frozendict[str, JaxComplexArray]
