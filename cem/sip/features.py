"""SIP features: compose circuit leaves into per-node units.

See `features` and `sip-td-error-circuit` in the thesis.

`SIPFeature` is the identification node (Emitter and Mismatch), used in
sensation and intention couplings. `SensationInputFeature` is the
specialized sensation-stack input (no observation_map, no Credit).
`ProspectFeature` is the reward-critic predictor (Emitter and Credit, no
Mismatch). `TDFeature` is the reward node (TDError), with unit gain and no
local noise. `RecruiterFeature` wraps a controllable `SIPFeature` with no
gain parent.
"""

from collections.abc import Mapping
from typing import Self

import equinox as eqx
import jax.numpy as jnp
from efax import NormalNP
from tjax import JaxRealArray, KeyArray, RngStream

from cem.npn.gaussian import GaussianNPN
from cem.sip.credit import Credit
from cem.sip.emitter import Emitter
from cem.sip.messages import InstrumentedMessage, evidence_scale, mean_and_precision
from cem.sip.mismatch import Mismatch
from cem.sip.td_error import TDError


class SIPFeatureSizes(eqx.Module):
    """Per-parent feature sizes for building one identification feature."""

    observation_content: int
    predictor: int
    gain: int


class SIPFeatureOutputs(eqx.Module):
    """One step of an identification feature."""

    message: InstrumentedMessage  # post-gain, noised
    observation_mismatch: NormalNP  # 2n unit-presence demand signal
    reconstruction_cost: JaxRealArray  # scalar
    confounding_cost: JaxRealArray  # scalar
    dissonance: JaxRealArray  # scalar; reconstruction and confounding
    loss: JaxRealArray  # scalar; combined surrogate


class SIPFeature(eqx.Module):
    """A feature carrying the full identification pair: an `Emitter` and a `Mismatch`.

    See `features` in the thesis.
    """

    emitter: Emitter
    mismatch: Mismatch
    last_message_index: eqx.nn.StateIndex[InstrumentedMessage]

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
        return cls(
            emitter=emitter,
            mismatch=mismatch,
            last_message_index=eqx.nn.StateIndex(InstrumentedMessage.initial(n)),
        )

    def step(
        self,
        state: eqx.nn.State,
        key: KeyArray,
        *,
        observation_content: NormalNP,
        predictor: InstrumentedMessage,
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
        new_state, emitter_outputs = self.emitter.step(
            state,
            key,
            observation_content=observation_content,
            predictor_instrument=predictor.instrument,
            gain_observation=gain_observation,
            goal=goal,
        )
        mismatch_outputs = self.mismatch.step(
            raw_observation=emitter_outputs.raw_observation,
            local_variance=emitter_outputs.local_variance,
            predictor=predictor,
            gain=gain,
            streams=streams,
            inference=inference,
        )
        new_state = new_state.set(self.last_message_index, emitter_outputs.message)
        return new_state, SIPFeatureOutputs(
            message=emitter_outputs.message,
            observation_mismatch=mismatch_outputs.mismatch,
            reconstruction_cost=mismatch_outputs.reconstruction_cost,
            confounding_cost=mismatch_outputs.confounding_cost,
            dissonance=mismatch_outputs.dissonance,
            loss=emitter_outputs.loss + mismatch_outputs.loss,
        )


class SensationInputFeature(eqx.Module):
    """The bottom sensation feature: the environment observation is its own observation."""

    instrument_map: GaussianNPN
    mismatch: Mismatch
    last_message_index: eqx.nn.StateIndex[InstrumentedMessage]

    @classmethod
    def create(
        cls,
        n: int,
        *,
        predictor_features: int,
        gain_features: int = 0,
        streams: Mapping[str, RngStream],
    ) -> Self:
        return cls(
            instrument_map=GaussianNPN.create(
                in_features=predictor_features, out_features=n, streams=streams
            ),
            mismatch=Mismatch.create(
                n,
                predictor_features=predictor_features,
                gain_features=gain_features,
                streams=streams,
            ),
            last_message_index=eqx.nn.StateIndex(InstrumentedMessage.initial(n)),
        )

    def step(
        self,
        state: eqx.nn.State,
        *,
        observation: NormalNP,
        predictor: InstrumentedMessage,
        gain: InstrumentedMessage | None,
        streams: Mapping[str, RngStream],
        inference: bool,
    ) -> tuple[eqx.nn.State, SIPFeatureOutputs]:
        """Advance the sensation input one time step.

        Threads `state` through unchanged (no Credit state to update) so the
        caller can iterate the stack uniformly.
        """
        inherited_instrument = self.instrument_map.infer(*mean_and_precision(predictor.instrument))
        inherited_mean, _ = mean_and_precision(inherited_instrument)
        local_variance = jnp.zeros_like(inherited_mean)

        if gain is None:
            out_observation = observation
            out_instrument = inherited_instrument
        else:
            gain_mean, _ = mean_and_precision(gain.observation)
            out_observation = evidence_scale(observation, gain_mean)
            out_instrument = evidence_scale(inherited_instrument, gain_mean)

        mismatch_outputs = self.mismatch.step(
            raw_observation=observation,
            local_variance=local_variance,
            predictor=predictor,
            gain=gain,
            streams=streams,
            inference=inference,
        )
        message = InstrumentedMessage(observation=out_observation, instrument=out_instrument)
        state = state.set(self.last_message_index, message)
        return state, SIPFeatureOutputs(
            message=message,
            observation_mismatch=mismatch_outputs.mismatch,
            reconstruction_cost=mismatch_outputs.reconstruction_cost,
            confounding_cost=mismatch_outputs.confounding_cost,
            dissonance=mismatch_outputs.dissonance,
            loss=mismatch_outputs.loss,
        )


class RecruiterFeature(eqx.Module):
    """A feature that sets how strongly others participate; its observation is their gain."""

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
        predictor: InstrumentedMessage,
        goal: JaxRealArray,
        streams: Mapping[str, RngStream],
        inference: bool,
    ) -> tuple[eqx.nn.State, SIPFeatureOutputs]:
        """Advance the recruiter one time step, with no gain parent."""
        return self.feature.step(
            state,
            key,
            observation_content=observation_content,
            predictor=predictor,
            gain=None,
            goal=goal,
            streams=streams,
            inference=inference,
        )


class ProspectFeatureOutputs(eqx.Module):
    """One step of a prospect feature."""

    message: InstrumentedMessage  # post-noise
    loss: JaxRealArray  # scalar; emitter loss (credit and promoting link)


class ProspectFeature(eqx.Module):
    """A feature that causally predicts reward: an `Emitter` and a `Credit`, no `Mismatch`."""

    emitter: Emitter
    last_message_index: eqx.nn.StateIndex[InstrumentedMessage]

    @classmethod
    def create(
        cls,
        n: int,
        *,
        observation_content_features: int,
        predictor_features: int,
        controllable: bool = True,
        trace_decay_rate: float = 0.0,
        streams: Mapping[str, RngStream],
    ) -> Self:
        observation_map = GaussianNPN.create(
            in_features=observation_content_features, out_features=n, streams=streams
        )
        instrument_map = GaussianNPN.create(
            in_features=predictor_features, out_features=n, streams=streams
        )
        baseline_map = GaussianNPN.create(
            in_features=observation_content_features, out_features=1, streams=streams
        )
        emitter = Emitter.create(
            observation_map=observation_map,
            instrument_map=instrument_map,
            credit=Credit.create(n, baseline_map, trace_decay_rate=trace_decay_rate),
            controllable=controllable,
        )
        return cls(
            emitter=emitter,
            last_message_index=eqx.nn.StateIndex(InstrumentedMessage.initial(n)),
        )

    def step(
        self,
        state: eqx.nn.State,
        key: KeyArray,
        *,
        observation_content: NormalNP,
        predictor_instrument: NormalNP,
        gain: InstrumentedMessage | None,
        goal: JaxRealArray,
    ) -> tuple[eqx.nn.State, ProspectFeatureOutputs]:
        """Advance the prospect feature one time step.

        Passes `gain.observation` to the emitter for evidence scaling when a
        gain parent is wired in; the instrument half is unused because there
        is no mismatch circuit. `gain=None` skips evidence scaling.
        """
        gain_observation = None if gain is None else gain.observation
        new_state, emitter_outputs = self.emitter.step(
            state,
            key,
            observation_content=observation_content,
            predictor_instrument=predictor_instrument,
            gain_observation=gain_observation,
            goal=goal,
        )
        new_state = new_state.set(self.last_message_index, emitter_outputs.message)
        return new_state, ProspectFeatureOutputs(
            message=emitter_outputs.message,
            loss=emitter_outputs.loss,
        )


class TDFeatureOutputs(eqx.Module):
    """One step of a reward feature."""

    observation: NormalNP  # pass-through of the environment reward
    advantage: JaxRealArray  # TD error in reward, replaces observation mismatch
    reconstruction_cost: JaxRealArray  # scalar
    confounding_cost: JaxRealArray  # scalar
    dissonance: JaxRealArray  # scalar; reconstruction and confounding
    uncertainty_loss: JaxRealArray  # scalar
    loss: JaxRealArray  # scalar; combined surrogate


class TDFeature(eqx.Module):
    """A reward feature: the signal the agent seeks to increase, learned by a `TDError` circuit."""

    td_error: TDError
    last_dissonance_index: eqx.nn.StateIndex[JaxRealArray]
    last_advantage_index: eqx.nn.StateIndex[JaxRealArray]

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
            ),
            last_dissonance_index=eqx.nn.StateIndex(jnp.asarray(0.0, dtype=jnp.float64)),
            last_advantage_index=eqx.nn.StateIndex(jnp.zeros(n, dtype=jnp.float64)),
        )

    def step(
        self,
        state: eqx.nn.State,
        *,
        reward_observation: NormalNP,
        predictor: InstrumentedMessage,
        streams: Mapping[str, RngStream],
        inference: bool,
    ) -> tuple[eqx.nn.State, TDFeatureOutputs]:
        """Advance the feature one time step."""
        new_state, td_outputs = self.td_error.step(
            state,
            reward_observation=reward_observation,
            predictor=predictor,
            streams=streams,
            inference=inference,
        )
        dissonance = td_outputs.reconstruction_cost + td_outputs.confounding_cost
        new_state = new_state.set(self.last_dissonance_index, dissonance)
        new_state = new_state.set(self.last_advantage_index, td_outputs.advantage)
        return new_state, TDFeatureOutputs(
            observation=reward_observation,
            advantage=td_outputs.advantage,
            reconstruction_cost=td_outputs.reconstruction_cost,
            confounding_cost=td_outputs.confounding_cost,
            dissonance=dissonance,
            uncertainty_loss=td_outputs.uncertainty_loss,
            loss=td_outputs.loss,
        )
