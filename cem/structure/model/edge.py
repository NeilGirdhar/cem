from __future__ import annotations

from collections.abc import Mapping
from dataclasses import InitVar
from typing import Protocol, override

from tjax import RngStream

from .module import Module
from .node import Node


class Edge(Module):
    """The base class for edges, which encode a relationship between nodes."""

    source: InitVar[Node]
    target: InitVar[Node]

    @override
    def __post_init__(
        self,
        streams: Mapping[str, RngStream],
        source: Node,
        target: Node,
    ) -> None:
        super().__post_init__(streams)

    def defines_order(self) -> bool:
        """When true, the edge's direction represents the order in which inference proceeds."""
        return True


class EdgeFactory(Protocol):
    def __call__(self, source: Node, target: Node, *, streams: Mapping[str, RngStream]) -> Edge: ...
