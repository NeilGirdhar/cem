from __future__ import annotations

from typing import Any

from tjax import frozendict

from cem.structure.graph.model import Model
from cem.structure.graph.node import NodeConfiguration
from cem.structure.problem.data_source import ProblemState


class RLModel[ProblemStateT: ProblemState](Model):
    """Model subclass for reinforcement learning.

    Concrete subclasses implement :meth:`get_action` and :meth:`get_reward` to
    extract RL-relevant outputs from the model configurations produced by :meth:`infer`.
    """

    def get_action(self, configurations: frozendict[str, NodeConfiguration]) -> dict[str, Any]:
        """Extract action fields from the given inference configurations."""
        raise NotImplementedError

    def get_reward(self, configurations: frozendict[str, NodeConfiguration]) -> dict[str, Any]:
        """Extract reward fields from the given inference configurations."""
        raise NotImplementedError
