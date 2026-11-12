"""The data source interface and the distinction between problem states and observations."""

import equinox as eqx
from tjax import KeyArray


class ProblemObservation(eqx.Module):
    """The part of a problem state visible to the model at inference time."""


class ProblemState(ProblemObservation):
    """The full state of a sampled example or environment.

    The state can also serve as the observation when the model sees it in full.
    Partially observed problems expose a separate view.
    """


class DataSource(eqx.Module):
    """A source of initial problem states for sampled examples or episodes."""

    def initial_problem_state(self, example_key: KeyArray) -> ProblemState:
        """Sample an initial state with the given example key."""
        raise NotImplementedError
