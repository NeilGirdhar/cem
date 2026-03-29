from __future__ import annotations

from collections.abc import Mapping
from typing import override

from tjax import RngStream

from cem.structure.model import EditableModel, ModelCreator, ProblemState, NodeRole

from .problem import RLProblem


class RLModelCreator[ProblemStateT: ProblemState](ModelCreator[ProblemStateT]):
    @override
    def create_model(self, model: EditableModel, streams: Mapping[str, RngStream]) -> None:
        assert isinstance(self.problem, RLProblem)
        for name, ep in self.problem.observation_distributions().items():
            observation_node = self.create_node(name, NodeRole.observation, ep, streams)
            model.add_node(observation_node)
        for name, ep in self.problem.action_distributions().items():  # ty: ignore
            action_node = self.create_node(name, NodeRole.action, ep, streams)
            model.add_node(action_node)
        for name, ep in self.problem.reward_distributions().items():  # ty: ignore
            reward_node = self.create_node(name, NodeRole.reward, ep, streams)
            model.add_node(reward_node)
        self.create_model_edges(model, streams)
