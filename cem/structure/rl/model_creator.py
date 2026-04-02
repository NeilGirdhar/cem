from __future__ import annotations

from collections.abc import Mapping
from typing import override

from tjax import RngStream

from cem.structure.graph.node import Node
from cem.structure.problem.creator import ModelCreator, NodeRole
from cem.structure.problem.data_source import ProblemState

from .problem import RLProblem


class RLModelCreator[ProblemStateT: ProblemState](ModelCreator[ProblemStateT]):
    @override
    def create_model(self, *, streams: Mapping[str, RngStream]) -> dict[str, Node]:
        nodes = {}
        assert isinstance(self.problem, RLProblem)
        for name, ep in self.problem.observation_distributions().items():
            observation_node = self.create_node(name, NodeRole.observation, ep, streams=streams)
            nodes[observation_node.name] = observation_node
        for name, ep in self.problem.action_distributions().items():  # ty: ignore
            action_node = self.create_node(name, NodeRole.action, ep, streams=streams)
            nodes[action_node.name] = action_node
        for name, ep in self.problem.reward_distributions().items():  # ty: ignore
            reward_node = self.create_node(name, NodeRole.reward, ep, streams=streams)
            nodes[reward_node.name] = reward_node
        return nodes
