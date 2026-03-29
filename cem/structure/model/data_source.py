import equinox as eqx
from tjax import KeyArray


class ProblemObservation(eqx.Module):
    """An observation is what is visible to inference.

    For an ordinary model, this is the input.  For an RL model, this is the observation in the POMDP
    sense.
    """


class ProblemState(ProblemObservation):
    """A problem state is the input that is sampled using the "example" key.

    For an ordinary model, this can include additional information that may be useful in plotting.
    For an RL model, this is the state in the POMDP sense.

    It inherits from ProblemObservation to reflect this "additional information" structure, and to
    facilitate returning the problem state as the observation in the common case.
    """


class DataSource(eqx.Module):
    """A data-producer produces the initial problem states."""

    def initial_problem_state(self, example_key: KeyArray) -> ProblemState:
        raise NotImplementedError
