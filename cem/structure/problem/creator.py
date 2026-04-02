from __future__ import annotations

from collections.abc import Mapping
from enum import Enum, auto
from typing import Any

import equinox as eqx
from efax import ExpectationParametrization
from tjax import RngStream

from cem.structure.graph.node import Node

from .data_source import DataSource, ProblemState
from .problem import Problem


class NodeRole(Enum):
    observation = 0
    latent = auto()
    action = auto()
    reward = auto()


class ModelCreator[ProblemStateT: ProblemState](eqx.Module):
    data_source: DataSource
    problem: Problem

    def input_routing(self) -> Mapping[str, tuple[str, str]]:
        """Map every observation field name to a node and field name explicitly."""
        return {}

    def output_routing(self) -> Mapping[str, tuple[str, str]]:
        """Map every externally requested output field name to a node and field name explicitly."""
        return {}

    def create_node(
        self,
        name: str,
        role: NodeRole,
        ep: ExpectationParametrization[Any],
        *,
        streams: Mapping[str, RngStream],
    ) -> Node:
        raise NotImplementedError

    def create_model(self, *, streams: Mapping[str, RngStream]) -> dict[str, Node]:
        nodes = {}
        for name, ep in self.problem.observation_distributions().items():
            observation_node = self.create_node(name, NodeRole.observation, ep, streams=streams)
            nodes[observation_node.name] = observation_node
        return nodes
