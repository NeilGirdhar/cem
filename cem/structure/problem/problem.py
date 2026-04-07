from __future__ import annotations

import equinox as eqx

from .data_source import DataSource, ProblemObservation, ProblemState


class Problem(eqx.Module):
    """This class encodes a reinforcement learning environment."""

    def extract_observation(self, state: ProblemState) -> ProblemObservation:
        return state

    def create_data_source(self) -> DataSource:
        raise NotImplementedError
