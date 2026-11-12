"""Problems and data sources that present structural causal models to SIP circuits."""

from typing import override

import jax.numpy as jnp
from efax import NormalNP, NormalPP
from tjax import KeyArray

from cem.framework.problem import DataSource, Problem
from cem.framework.problem.data_source import ProblemObservation, ProblemState

from .scenarios import InheritedInstrumentSample, InheritedInstrumentScenario

_PARENT_INSTRUMENT_PRECISION = 1.0
_CHILD_OBSERVATION_PRECISION = 100.0


class InheritedInstrumentObservation(ProblemObservation):
    """The parents' instruments and the child's raw observation."""

    parent_instrument: NormalNP  # (k,)
    child_observation: NormalNP  # (1,)


class InheritedInstrumentState(InheritedInstrumentObservation, ProblemState):
    """The observation together with the structural sample that generated it."""

    sample: InheritedInstrumentSample


class InheritedInstrumentDataSource(DataSource):
    """Draw fresh samples from the scenario.

    Each parent is a root controllable feature whose inherited instrument is standard normal, so it
    injects unit-variance noise and emits an instrument with that noise as its mean and unit
    precision. The child's raw observation is the child's value, observed precisely.
    """

    scenario: InheritedInstrumentScenario

    @override
    def initial_problem_state(self, example_key: KeyArray) -> InheritedInstrumentState:
        sample = self.scenario.sample(example_key)
        noise = sample.injected_noise
        child = sample.child[jnp.newaxis]
        return InheritedInstrumentState(
            parent_instrument=NormalPP(
                noise, jnp.full_like(noise, _PARENT_INSTRUMENT_PRECISION)
            ).to_nat(),
            child_observation=NormalPP(
                child, jnp.full_like(child, _CHILD_OBSERVATION_PRECISION)
            ).to_nat(),
            sample=sample,
        )


class InheritedInstrumentProblem(Problem):
    """Fit a child's instrument map to the child's observation from its parents' instruments.

    Training and inference draw from the same scenario. Their samples are independent because the
    solver gives each split its own seed, so evaluation is always on held-out samples.
    """

    scenario: InheritedInstrumentScenario

    @override
    def create_data_source(self, *, inference: bool = False) -> InheritedInstrumentDataSource:
        return InheritedInstrumentDataSource(self.scenario)

    @override
    def extract_observation(self, state: ProblemState) -> InheritedInstrumentObservation:
        assert isinstance(state, InheritedInstrumentState)
        return InheritedInstrumentObservation(
            parent_instrument=state.parent_instrument,
            child_observation=state.child_observation,
        )
