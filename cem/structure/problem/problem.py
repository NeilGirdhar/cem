"""The shared interface for supervised problems and reinforcement learning environments."""

import equinox as eqx

from .data_source import DataSource, ProblemObservation, ProblemState


class Problem(eqx.Module):
    """A problem that defines where examples come from and what the model observes."""

    def extract_observation(self, state: ProblemState) -> ProblemObservation:
        """Return the model-visible part of a state. The default exposes the full state."""
        return state

    def create_data_source(self, *, inference: bool = False) -> DataSource:
        """Select the data source for the requested training or inference split."""
        raise NotImplementedError
