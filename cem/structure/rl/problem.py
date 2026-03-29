from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import equinox as eqx
from efax import ExpectationParametrization
from tjax import JaxArray, JaxBooleanArray, KeyArray

from cem.structure.problem.data_source import ProblemState
from cem.structure.problem.problem import Problem


class ProblemAction(eqx.Module):
    pass


class ProblemReward(eqx.Module):
    def total_reward(self) -> JaxArray:
        raise NotImplementedError


class RLProblem[
    ProblemStateT: ProblemState,
    ProblemActionT: ProblemAction,
    ProblemRewardT: ProblemReward,
](Problem):
    """This class encodes a reinforcement learning environment."""

    def action_distributions(self) -> Mapping[str, ExpectationParametrization[Any]]:
        raise NotImplementedError

    def reward_distributions(self) -> Mapping[str, ExpectationParametrization[Any]]:
        raise NotImplementedError

    def max_episode_steps(self) -> int:
        """The episode terminates when this many steps have been done."""
        return 1

    # def reward_threshold(self) -> float:
    #     """The average return threshold at which the problem is considered solved."""
    #     raise NotImplementedError

    def produce_action(
        self, action_fields: Mapping[str, Any], example_key: KeyArray
    ) -> ProblemActionT:
        """Assemble the node values for the action fields into a ProblemActionT.

        This function may canonicalize the action, e.g., clamping an output to an interval.
        """
        raise NotImplementedError

    def iterate_state(
        self,
        state: ProblemStateT,
        action: ProblemActionT,
    ) -> tuple[ProblemStateT, ProblemRewardT, JaxBooleanArray]:
        """Iterate the state.

        Returns:
            new_state: The next state object.
            reward: The next reward object.
            done: A scalar Boolean indicating doneness.
        """
        raise NotImplementedError
