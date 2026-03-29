from __future__ import annotations

from collections.abc import Mapping
from enum import Enum, auto
from typing import Any

import equinox as eqx
from efax import ExpectationParametrization
from tjax import RngStream

from cem.structure.graph.editable import EditableModel
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

    def create_node(
        self,
        name: str,
        role: NodeRole,
        ep: ExpectationParametrization[Any],
        streams: Mapping[str, RngStream],
    ) -> Node:
        raise NotImplementedError

    def create_model_edges(self, model: EditableModel, streams: Mapping[str, RngStream]) -> None:
        pass

    def create_model(self, model: EditableModel, streams: Mapping[str, RngStream]) -> None:
        for name, ep in self.problem.observation_distributions().items():
            observation_node = self.create_node(name, NodeRole.observation, ep, streams)
            model.add_node(observation_node)
        self.create_model_edges(model, streams)
