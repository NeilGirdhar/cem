from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import equinox as eqx
from efax import ExpectationParametrization

from .data_source import DataSource, ProblemObservation, ProblemState


class Problem(eqx.Module):
    """This class encodes a reinforcement learning environment."""

    def observation_distributions(self) -> Mapping[str, ExpectationParametrization[Any]]:
        raise NotImplementedError

    def extract_observation(self, state: ProblemState) -> ProblemObservation:
        return state

    def create_data_source(self) -> DataSource:
        raise NotImplementedError
