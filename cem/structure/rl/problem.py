"""The environment interface for taking actions, reporting rewards, and ending episodes."""

from collections.abc import Mapping
from typing import Any

import equinox as eqx
from tjax import JaxArray, JaxBooleanArray, KeyArray

from cem.structure.problem.data_source import ProblemState
from cem.structure.problem.problem import Problem


class ProblemAction(eqx.Module):
    """An action in the form accepted by an environment."""


class ProblemReward(eqx.Module):
    """The reward components from one environment transition."""

    def total_reward(self) -> JaxArray:
        """Reduce the reward components to one scalar for the transition."""
        raise NotImplementedError


class RLProblem[
    ProblemStateT: ProblemState,
    ProblemActionT: ProblemAction,
    ProblemRewardT: ProblemReward,
](Problem):
    """An environment that converts model outputs into actions and advances its state."""

    def max_episode_steps(self) -> int:
        """Return the episode step limit, independent of environment termination."""
        return 1

    def produce_action(
        self, action_fields: Mapping[str, Any], example_key: KeyArray
    ) -> ProblemActionT:
        """Convert named model outputs into an action the environment accepts.

        The conversion can use the example key and enforce constraints such as bounds.
        """
        raise NotImplementedError

    def iterate_state(
        self,
        state: ProblemStateT,
        action: ProblemActionT,
    ) -> tuple[ProblemStateT, ProblemRewardT, JaxBooleanArray]:
        """Apply an action and return the next state, reward, and scalar termination flag."""
        raise NotImplementedError
