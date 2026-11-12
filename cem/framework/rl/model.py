"""The model interface for extracting action and reward fields during rollouts."""

from typing import Any

from cem.framework.graph.model import Model
from cem.framework.graph.node import NodeConfiguration
from cem.framework.problem.data_source import ProblemState


class RLModel[ProblemStateT: ProblemState](Model):
    """A model whose node outputs can be interpreted as action and reward fields."""

    def get_action(self, configurations: frozendict[str, NodeConfiguration]) -> dict[str, Any]:
        """Extract action fields from the given inference configurations."""
        raise NotImplementedError

    def get_reward(self, configurations: frozendict[str, NodeConfiguration]) -> dict[str, Any]:
        """Extract reward fields from the given inference configurations."""
        raise NotImplementedError
