from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING

import equinox as eqx
import networkx as nx
from tjax import RngStream
from tjax.dataclasses import field

from .edge import EdgeFactory
from .node import Node

DiGraph = nx.DiGraph[str] if TYPE_CHECKING else nx.DiGraph


class EditableModel(eqx.Module):
    network: nx.DiGraph[str] = field(default_factory=DiGraph)

    # Graph access methods -------------------------------------------------------------------------
    def get_node(self, node_name: str) -> Node:
        if node_name not in self.network:
            msg = f"{node_name} not in network"
            raise ValueError(msg)
        node_dict = self.network.nodes[node_name]
        if "node" not in node_dict:
            msg = f"{node_name} has no node"
            raise ValueError(msg)
        node = node_dict["node"]
        assert isinstance(node, Node)
        return node

    # Construction methods -------------------------------------------------------------------------
    def add_node(self, node: Node) -> None:
        self.network.add_node(node.name, node=node)

    def add_edge(
        self, source: str, target: str, streams: Mapping[str, RngStream], create_edge: EdgeFactory
    ) -> None:
        source_node = self.get_node(source)
        target_node = self.get_node(target)
        edge = create_edge(source_node, target_node, streams=streams)
        self.network.add_edge(source, target, edge=edge)
