from __future__ import annotations

from collections.abc import Mapping
from typing import Any, override

from efax import ExpectationParametrization
from tjax import RngStream

from cem.structure.model import EditableModel, ModelCreator, Node, ProblemState
from .problem import RLProblem


class RLModelCreator[ProblemStateT: ProblemState](ModelCreator[ProblemStateT]):
    def create_action_node(
        self,
        name: str,
        streams: Mapping[str, RngStream],
        ep: ExpectationParametrization[Any],
    ) -> Node:
        raise NotImplementedError

    def create_reward_node(
        self,
        name: str,
        streams: Mapping[str, RngStream],
        ep: ExpectationParametrization[Any],
    ) -> Node:
        raise NotImplementedError

    @override
    def create_model(self, model: EditableModel, streams: Mapping[str, RngStream]) -> None:
        assert isinstance(self.problem, RLProblem)
        for name, ep in self.problem.observation_distributions().items():
            observation_node = self.create_observation_node(name, streams, ep)
            model.add_node(observation_node)
        for name, ep in self.problem.action_distributions().items():  # ty: ignore
            action_node = self.create_action_node(name, streams, ep)
            model.add_node(action_node)
        for name, ep in self.problem.reward_distributions().items():  # ty: ignore
            reward_node = self.create_reward_node(name, streams, ep)
            model.add_node(reward_node)
        self.create_model_edges(model, streams)
