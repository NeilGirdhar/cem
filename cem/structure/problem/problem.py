import equinox as eqx

from .data_source import DataSource, ProblemObservation, ProblemState


class Problem(eqx.Module):
    """This class encodes a reinforcement learning environment."""

    def extract_observation(self, state: ProblemState) -> ProblemObservation:
        return state

    def create_data_source(self, *, inference: bool = False) -> DataSource:
        """Create the training or inference data source."""
        raise NotImplementedError
