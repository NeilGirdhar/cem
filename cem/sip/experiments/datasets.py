"""Problems and data sources that present structural causal models to SIP circuits."""

from typing import override

import jax.numpy as jnp
from efax import NormalNP, NormalPP
from tjax import KeyArray

from cem.framework.problem import DataSource, Problem
from cem.framework.problem.data_source import ProblemObservation, ProblemState

from .scenarios import InstrumentChainSample, InstrumentChainScenario

_PARENT_INSTRUMENT_PRECISION = 1.0
_FEATURE_OBSERVATION_PRECISION = 100.0


class InstrumentChainObservation(ProblemObservation):
    """The parents' instruments and the raw observation of each feature in the chain."""

    parent_instrument: NormalNP  # (k,)
    feature_observations: NormalNP  # (depth,)


class InstrumentChainState(InstrumentChainObservation, ProblemState):
    """The observation together with the structural sample that generated it."""

    sample: InstrumentChainSample


class InstrumentChainDataSource(DataSource):
    """Draw fresh samples from the scenario.

    Each parent is a root controllable feature whose inherited instrument is standard normal, so it
    injects unit-variance noise and emits an instrument with that noise as its mean and unit
    precision. Each feature's raw observation is the feature's value, observed precisely.
    """

    scenario: InstrumentChainScenario

    @override
    def initial_problem_state(self, example_key: KeyArray) -> InstrumentChainState:
        sample = self.scenario.sample(example_key)
        noise = sample.injected_noise
        features = sample.features[: self.scenario.depth]
        return InstrumentChainState(
            parent_instrument=NormalPP(
                noise, jnp.full_like(noise, _PARENT_INSTRUMENT_PRECISION)
            ).to_nat(),
            feature_observations=NormalPP(
                features, jnp.full_like(features, _FEATURE_OBSERVATION_PRECISION)
            ).to_nat(),
            sample=sample,
        )


class InstrumentChainProblem(Problem):
    """Fit each feature's instrument map to its observation from its predecessor's instrument.

    Training and inference draw from the same scenario. Their samples are independent because the
    solver gives each split its own seed, so evaluation is always on held-out samples.
    """

    scenario: InstrumentChainScenario

    @override
    def create_data_source(self, *, inference: bool = False) -> InstrumentChainDataSource:
        return InstrumentChainDataSource(self.scenario)

    @override
    def extract_observation(self, state: ProblemState) -> InstrumentChainObservation:
        assert isinstance(state, InstrumentChainState)
        return InstrumentChainObservation(
            parent_instrument=state.parent_instrument,
            feature_observations=state.feature_observations,
        )
