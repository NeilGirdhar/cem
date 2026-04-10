from __future__ import annotations

from tjax import frozendict

from cem.phasor.message import PhasorMessage
from cem.structure.graph.node import NodeConfiguration


class PhasorInputConfiguration(NodeConfiguration):
    """Holds PhasorMessage encodings for one phasor inference step."""

    values: frozendict[str, PhasorMessage]
