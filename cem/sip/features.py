"""SIP features: compose circuit leaves into per-node units.

See `features` and `sip-td-error-circuit` in `~/src/typst/thesis.typ`.

`SIPFeature` is the identification node (Emitter + Mismatch), playing the
sensation/intention/prospect role. `TDFeature` is the reward node (TDError),
with unit gain and no local noise.
"""

from collections.abc import Mapping
from typing import Self

import equinox as eqx
from efax import NormalNP
from tjax import JaxRealArray, KeyArray, RngStream

from cem.npn.gaussian import GaussianNPN
from cem.sip.credit import Credit
from cem.sip.emitter import Emitter
from cem.sip.messages import InstrumentedMessage
from cem.sip.mismatch import Mismatch
from cem.sip.td_error import TDError


class SIPFeatureSizes(eqx.Module):
    """Per-parent feature sizes for building one identification feature."""

    observation_content: int
    predictor: int
    gain: int


class SIPFeatureOutputs(eqx.Module):
    """One step of an identification feature."""

    observation: NormalNP  # post-gain, noised
    instrument: NormalNP  # post-gain, noised
    observation_mismatch: NormalNP  # 2n unit-presence demand signal
    reconstruction_cost: JaxRealArray  # scalar
    confounding_cost: JaxRealArray  # scalar
    loss: JaxRealArray  # scalar; combined surrogate


class SIPFeature(eqx.Module):
    """One identification node: Emitter + Mismatch.

    Exposes the thesis's five-input / five-output feature interface. The
    emitter builds the feature's observation and instrument from innovation
    content; the mismatch circuit measures reconstruction and purification
    error against the parents and emits the outgoing demand signal.
    """

    emitter: Emitter
    mismatch: Mismatch

    @classmethod
    def create(
        cls,
        n: int,
        sizes: SIPFeatureSizes,
        *,
        controllable: bool,
        trace_decay_rate: float = 0.0,
        streams: Mapping[str, RngStream],
    ) -> Self:
        observation_map = GaussianNPN.create(
            in_features=sizes.observation_content, out_features=n, streams=streams
        )
        instrument_map = GaussianNPN.create(
            in_features=sizes.predictor, out_features=n, streams=streams
        )
        baseline_map = GaussianNPN.create(
            in_features=sizes.observation_content, out_features=1, streams=streams
        )
        emitter = Emitter.create(
            observation_map=observation_map,
            instrument_map=instrument_map,
            credit=Credit.create(n, baseline_map, trace_decay_rate=trace_decay_rate),
            controllable=controllable,
        )
        mismatch = Mismatch.create(
            n,
            predictor_features=sizes.predictor,
            gain_features=sizes.gain,
            streams=streams,
        )
        return cls(emitter=emitter, mismatch=mismatch)

    def step(
        self,
        state: eqx.nn.State,
        key: KeyArray,
        *,
        observation_content: NormalNP,
        predictors: InstrumentedMessage,
        gain: InstrumentedMessage | None,
        goal: JaxRealArray,
        streams: Mapping[str, RngStream],
        inference: bool,
    ) -> tuple[eqx.nn.State, SIPFeatureOutputs]:
        """Advance the feature one time step.

        `gain=None` marks a recruiter feature at the top of the gain chain:
        the underlying `Mismatch` must have been created with `gain=0` so its
        witness map omits the missing gain parent.
        """
        gain_observation = None if gain is None else gain.observation
        gain_instrument = None if gain is None else gain.instrument
        new_state, emitter_outputs = self.emitter.step(
            state,
            key,
            observation_content=observation_content,
            predictor_instruments=predictors.instrument,
            gain_observation=gain_observation,
            goal=goal,
        )
        mismatch_outputs = self.mismatch.step(
            raw_observation=emitter_outputs.raw_observation,
            local_variance=emitter_outputs.local_variance,
            predictor_observations=predictors.observation,
            predictor_instruments=predictors.instrument,
            gain_observation=gain_observation,
            gain_instrument=gain_instrument,
            streams=streams,
            inference=inference,
        )
        return new_state, SIPFeatureOutputs(
            observation=emitter_outputs.observation,
            instrument=emitter_outputs.instrument,
            observation_mismatch=mismatch_outputs.mismatch,
            reconstruction_cost=mismatch_outputs.reconstruction_cost,
            confounding_cost=mismatch_outputs.confounding_cost,
            loss=emitter_outputs.loss + mismatch_outputs.loss,
        )


class RecruiterFeature(eqx.Module):
    """One recruiter node: a controllable `SIPFeature` with no gain parent.

    Recruiters sit at the top of the gain chain, so they have no incoming
    gain signals. The underlying `SIPFeature` is created with `gain=0` so
    its witness map omits gain features, and every step passes `gain=None`.
    A recruiter's own `observation` and `instrument` outputs serve as the
    `gain_observation` and `gain_instrument` of consumer features.
    """

    feature: SIPFeature

    @classmethod
    def create(
        cls,
        n: int,
        *,
        observation_content_features: int,
        predictor_features: int,
        trace_decay_rate: float = 0.0,
        streams: Mapping[str, RngStream],
    ) -> Self:
        sizes = SIPFeatureSizes(
            observation_content=observation_content_features,
            predictor=predictor_features,
            gain=0,
        )
        return cls(
            feature=SIPFeature.create(
                n, sizes, controllable=True, trace_decay_rate=trace_decay_rate, streams=streams
            )
        )

    def step(
        self,
        state: eqx.nn.State,
        key: KeyArray,
        *,
        observation_content: NormalNP,
        predictors: InstrumentedMessage,
        goal: JaxRealArray,
        streams: Mapping[str, RngStream],
        inference: bool,
    ) -> tuple[eqx.nn.State, SIPFeatureOutputs]:
        """Advance the recruiter one time step, with no gain parent."""
        return self.feature.step(
            state,
            key,
            observation_content=observation_content,
            predictors=predictors,
            gain=None,
            goal=goal,
            streams=streams,
            inference=inference,
        )


class TDFeatureOutputs(eqx.Module):
    """One step of a reward feature."""

    observation: NormalNP  # pass-through of the environment reward
    advantage: JaxRealArray  # TD error in reward, replaces observation mismatch
    reconstruction_cost: JaxRealArray  # scalar
    confounding_cost: JaxRealArray  # scalar
    uncertainty_loss: JaxRealArray  # scalar
    loss: JaxRealArray  # scalar; combined surrogate


class TDFeature(eqx.Module):
    """One reward node: TDError circuit alone.

    Reward features use fixed unit gain and are not controllable, so they
    skip the emitter's noise top-up, evidence scaling, and credit circuit.
    The feature's observation is the environment-supplied reward; the
    outgoing signal is the TD-error advantage.
    """

    td_error: TDError

    @classmethod
    def create(
        cls,
        n: int,
        *,
        predictor_features: int,
        discount: float = 1.0,
        streams: Mapping[str, RngStream],
    ) -> Self:
        return cls(
            td_error=TDError.create(
                n,
                predictor_features=predictor_features,
                discount=discount,
                streams=streams,
            )
        )

    def step(
        self,
        state: eqx.nn.State,
        *,
        reward_observation: NormalNP,
        predictors: InstrumentedMessage,
        streams: Mapping[str, RngStream],
        inference: bool,
    ) -> tuple[eqx.nn.State, TDFeatureOutputs]:
        """Advance the feature one time step."""
        new_state, td_outputs = self.td_error.step(
            state,
            reward_observation=reward_observation,
            predictor_observations=predictors.observation,
            predictor_instruments=predictors.instrument,
            streams=streams,
            inference=inference,
        )
        return new_state, TDFeatureOutputs(
            observation=reward_observation,
            advantage=td_outputs.advantage,
            reconstruction_cost=td_outputs.reconstruction_cost,
            confounding_cost=td_outputs.confounding_cost,
            uncertainty_loss=td_outputs.uncertainty_loss,
            loss=td_outputs.loss,
        )
