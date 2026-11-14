"""Problems and data sources that present structural causal models to SIP circuits."""

from typing import override

import jax.numpy as jnp
from efax import NormalNP, NormalPP
from tjax import KeyArray

from cem.framework.problem import DataSource, Problem
from cem.framework.problem.data_source import ProblemObservation, ProblemState
from cem.sip.messages import InstrumentedMessage

from .scenarios import (
    ConfoundedPredictionSample,
    ConfoundedPredictionScenario,
    InstrumentChainSample,
    InstrumentChainScenario,
)

_PARENT_INSTRUMENT_PRECISION = 1.0
_FEATURE_OBSERVATION_PRECISION = 100.0
# The variance of each feature's raw observation about the feature's value.
FEATURE_OBSERVATION_VARIANCE = 1.0 / _FEATURE_OBSERVATION_PRECISION


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


class ConfoundedPredictionObservation(ProblemObservation):
    """The parents' instrumented messages and the target's raw observation."""

    predictor: InstrumentedMessage  # (k,)
    target_observation: NormalNP  # (1,)


class ConfoundedPredictionState(ConfoundedPredictionObservation, ProblemState):
    """The observation together with the structural sample that generated it."""

    sample: ConfoundedPredictionSample


class ConfoundedPredictionDataSource(DataSource):
    """Draw fresh samples from the scenario.

    Each parent is a root controllable feature: its observation is its value, observed precisely,
    and its instrument has its injected noise as its mean and unit precision. The target's raw
    observation is the target's value, observed precisely.
    """

    scenario: ConfoundedPredictionScenario

    @override
    def initial_problem_state(self, example_key: KeyArray) -> ConfoundedPredictionState:
        sample = self.scenario.sample(example_key)
        noise = sample.injected_noise
        target = sample.target[jnp.newaxis]
        return ConfoundedPredictionState(
            predictor=InstrumentedMessage(
                observation=NormalPP(
                    sample.parents, jnp.full_like(noise, _FEATURE_OBSERVATION_PRECISION)
                ).to_nat(),
                instrument=NormalPP(
                    noise, jnp.full_like(noise, _PARENT_INSTRUMENT_PRECISION)
                ).to_nat(),
            ),
            target_observation=NormalPP(
                target, jnp.full_like(target, _FEATURE_OBSERVATION_PRECISION)
            ).to_nat(),
            sample=sample,
        )


class ConfoundedPredictionProblem(Problem):
    """Predict a target from its parents' observations, with the parents' instruments available.

    Training and inference draw from the same scenario. Their samples are independent because the
    solver gives each split its own seed, so evaluation is always on held-out samples.
    """

    scenario: ConfoundedPredictionScenario

    @override
    def create_data_source(self, *, inference: bool = False) -> ConfoundedPredictionDataSource:
        return ConfoundedPredictionDataSource(self.scenario)

    @override
    def extract_observation(self, state: ProblemState) -> ConfoundedPredictionObservation:
        assert isinstance(state, ConfoundedPredictionState)
        return ConfoundedPredictionObservation(
            predictor=state.predictor, target_observation=state.target_observation
        )
