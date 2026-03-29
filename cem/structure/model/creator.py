from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import equinox as eqx
from efax import ExpectationParametrization
from tjax import RngStream

from .data_source import DataSource, ProblemState
from .editable import EditableModel
from .node import Node
from .problem import Problem


class ModelCreator[ProblemStateT: ProblemState](eqx.Module):
    data_source: DataSource
    problem: Problem

    def create_observation_node(
        self,
        name: str,
        streams: Mapping[str, RngStream],
        ep: ExpectationParametrization[Any],
    ) -> Node:
        raise NotImplementedError

    def create_model_edges(self, model: EditableModel, streams: Mapping[str, RngStream]) -> None:
        pass

    def create_model(self, model: EditableModel, streams: Mapping[str, RngStream]) -> None:
        for name, ep in self.problem.observation_distributions().items():
            observation_node = self.create_observation_node(name, streams, ep)
            model.add_node(observation_node)
        self.create_model_edges(model, streams)
