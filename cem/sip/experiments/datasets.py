"""Problems and data sources that present structural causal models to SIP circuits."""

from typing import override

import jax.numpy as jnp
from efax import NormalNP, NormalPP
from tjax import JaxRealArray, KeyArray

from cem.framework.problem import DataSource, Problem
from cem.framework.problem.data_source import ProblemObservation, ProblemState
from cem.sip.messages import InstrumentedMessage

from .scenarios import (
    ConfoundedPredictionSample,
    ConfoundedPredictionScenario,
    DirectCreditSample,
    DirectCreditScenario,
    EndogenousGainSample,
    EndogenousGainScenario,
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
        return ConfoundedPredictionState(
            predictor=_root_message(sample.parents, sample.injected_noise),
            target_observation=_precise(sample.target[jnp.newaxis]),
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


def _precise(values: JaxRealArray) -> NormalNP:
    """Return a precise observation of the values."""
    return NormalPP(values, jnp.full_like(values, _FEATURE_OBSERVATION_PRECISION)).to_nat()


def _root_message(values: JaxRealArray, noise: JaxRealArray) -> InstrumentedMessage:
    """Return a root controllable feature's message: its precise value and its injected noise."""
    return InstrumentedMessage(
        observation=_precise(values),
        instrument=NormalPP(noise, jnp.full_like(noise, _PARENT_INSTRUMENT_PRECISION)).to_nat(),
    )


class EndogenousGainObservation(ProblemObservation):
    """The parent's and gain's instrumented messages and the target's raw observation."""

    predictor: InstrumentedMessage  # (1,)
    gain: InstrumentedMessage  # (1,)
    target_observation: NormalNP  # (1,)


class EndogenousGainState(EndogenousGainObservation, ProblemState):
    """The observation together with the structural sample that generated it."""

    sample: EndogenousGainSample


class EndogenousGainDataSource(DataSource):
    """Draw fresh samples from the scenario.

    The parent is a root controllable feature, as in `ConfoundedPredictionDataSource`. The gain's
    observation is the gain, observed precisely, and its instrument has the gain noise as its mean
    and unit precision.
    """

    scenario: EndogenousGainScenario

    @override
    def initial_problem_state(self, example_key: KeyArray) -> EndogenousGainState:
        sample = self.scenario.sample(example_key)
        prediction = sample.prediction
        return EndogenousGainState(
            predictor=_root_message(prediction.parents, prediction.injected_noise),
            gain=_root_message(sample.gain[jnp.newaxis], sample.gain_noise[jnp.newaxis]),
            target_observation=_precise(prediction.target[jnp.newaxis]),
            sample=sample,
        )


class EndogenousGainProblem(Problem):
    """Predict a target from its parent's observation, with gain weighting each example.

    Training and inference draw from the same scenario, with independent samples.
    """

    scenario: EndogenousGainScenario

    @override
    def create_data_source(self, *, inference: bool = False) -> EndogenousGainDataSource:
        return EndogenousGainDataSource(self.scenario)

    @override
    def extract_observation(self, state: ProblemState) -> EndogenousGainObservation:
        assert isinstance(state, EndogenousGainState)
        return EndogenousGainObservation(
            predictor=state.predictor,
            gain=state.gain,
            target_observation=state.target_observation,
        )


class DirectCreditState(ProblemState):
    """The feature's observation content and gain, and the sample that generated them.

    The content and the gain are precise observations. The model sees the whole state because the
    oracle goal depends on the value the feature emits, so the model evaluates the goal from the
    sample after emitting.
    """

    content: NormalNP  # (1,)
    gain: NormalNP  # (1,)
    sample: DirectCreditSample


class DirectCreditDataSource(DataSource):
    """Draw fresh samples from the scenario."""

    scenario: DirectCreditScenario

    @override
    def initial_problem_state(self, example_key: KeyArray) -> DirectCreditState:
        sample = self.scenario.sample(example_key)
        return DirectCreditState(
            content=_precise(sample.confounder[jnp.newaxis]),
            gain=_precise(sample.gain[jnp.newaxis]),
            sample=sample,
        )


class DirectCreditProblem(Problem):
    """Credit a controllable feature's experiments with an oracle goal.

    Training and inference draw from the same scenario, with independent samples.
    """

    scenario: DirectCreditScenario

    @override
    def create_data_source(self, *, inference: bool = False) -> DirectCreditDataSource:
        return DirectCreditDataSource(self.scenario)

    @override
    def extract_observation(self, state: ProblemState) -> DirectCreditState:
        assert isinstance(state, DirectCreditState)
        return state
