"""Compose SIP circuits into sensation, intention, recruiter, prospect, and reward features.

See `features` and `sip-td-error-circuit` in the thesis.
"""

from collections.abc import Mapping
from typing import Self

import equinox as eqx
import jax.numpy as jnp
from efax import NormalNP
from tjax import JaxRealArray, KeyArray, RngStream

from cem.npn.gaussian import GaussianNeuralProjection
from cem.sip.credit import Credit
from cem.sip.emitter import Emitter
from cem.sip.instrument import inherit_instrument, instrument_loss
from cem.sip.losses import SIPLossOutputs
from cem.sip.messages import InstrumentedMessage, evidence_scale
from cem.sip.mismatch import Mismatch
from cem.sip.settings import CreditSettings, MismatchSettings, TDSettings
from cem.sip.td_error import TDError


class SIPFeatureSizes(eqx.Module):
    """Input widths for an identification feature's observation content, predictor, and gain."""

    observation_content: int
    predictor: int
    gain: int


class SIPFeatureOutputs(SIPLossOutputs):
    """An instrumented observation, its demand for explanation, and the training costs."""

    message: InstrumentedMessage  # post-gain, noised
    observation_mismatch: NormalNP  # 2n unit-presence demand signal
    reconstruction_cost: JaxRealArray  # scalar
    confounding_cost: JaxRealArray  # scalar


class SIPFeature(eqx.Module):
    """Emit an instrumented observation and learn to predict it from parent features.

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
        credit_settings: CreditSettings,
        mismatch_settings: MismatchSettings,
        streams: Mapping[str, RngStream],
    ) -> Self:
        """Build an identification feature with separate widths for its three parent inputs.

        Observation content supplies the emitted observation and the credit baseline. The
        predictor supplies the causal prediction and its instrument. The gain width controls
        whether the witness also receives a gain instrument.
        """
        observation_map = GaussianNeuralProjection.create(
            in_features=sizes.observation_content, out_features=n, streams=streams
        )
        instrument_map = GaussianNeuralProjection.create(
            in_features=sizes.predictor, out_features=n, streams=streams
        )
        baseline_map = GaussianNeuralProjection.create(
            in_features=sizes.observation_content, out_features=1, streams=streams
        )
        emitter = Emitter.create(
            observation_map=observation_map,
            instrument_map=instrument_map,
            credit=Credit.create(n, baseline_map, credit_settings=credit_settings),
            controllable=controllable,
        )
        mismatch_circuit = Mismatch.create(
            n,
            predictor_features=sizes.predictor,
            gain_features=sizes.gain,
            mismatch_settings=mismatch_settings,
            streams=streams,
        )
        return cls(
            emitter=emitter,
            mismatch=mismatch_circuit,
            last_message_index=eqx.nn.StateIndex(InstrumentedMessage.initial(n)),
        )

    def _step(
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
        """Emit an instrumented observation and score the predictor against it.

        Score the raw observation with its added local variance and save the emitted message
        in state. With no gain parent, the feature must have been built with zero gain
        features so the witness accepts predictor instruments alone.
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
            loss=emitter_outputs.loss + mismatch_outputs.loss,
        )


class SensationFeature(SIPFeature):
    """Learn a sensation from the reconstruction and confounding costs below it."""

    def step(  # ruff: ignore[too-many-arguments]
        self,
        state: eqx.nn.State,
        key: KeyArray,
        *,
        observation_content: NormalNP,
        predictor: InstrumentedMessage,
        gain: InstrumentedMessage | None,
        reconstruction: JaxRealArray,
        confounding: JaxRealArray,
        streams: Mapping[str, RngStream],
        inference: bool,
    ) -> tuple[eqx.nn.State, SIPFeatureOutputs]:
        return self._step(
            state,
            key,
            observation_content=observation_content,
            predictor=predictor,
            gain=gain,
            goal=reconstruction + confounding,
            streams=streams,
            inference=inference,
        )


class IntentionFeature(SIPFeature):
    """Learn an intention from reward, curiosity, and tracking advantages."""

    def step(  # ruff: ignore[too-many-arguments]
        self,
        state: eqx.nn.State,
        key: KeyArray,
        *,
        observation_content: NormalNP,
        predictor: InstrumentedMessage,
        gain: InstrumentedMessage | None,
        reward: JaxRealArray,
        curiosity: JaxRealArray,
        tracking: JaxRealArray,
        streams: Mapping[str, RngStream],
        inference: bool,
    ) -> tuple[eqx.nn.State, SIPFeatureOutputs]:
        return self._step(
            state,
            key,
            observation_content=observation_content,
            predictor=predictor,
            gain=gain,
            goal=reward + curiosity + tracking,
            streams=streams,
            inference=inference,
        )


class SensationInputFeature(eqx.Module):
    """Use the supplied observation as the prediction target at the base of the sensation stack."""

    instrument_map: GaussianNeuralProjection
    mismatch: Mismatch
    last_message_index: eqx.nn.StateIndex[InstrumentedMessage]

    @classmethod
    def create(
        cls,
        n: int,
        *,
        predictor_features: int,
        gain_features: int = 0,
        mismatch_settings: MismatchSettings,
        streams: Mapping[str, RngStream],
    ) -> Self:
        return cls(
            instrument_map=GaussianNeuralProjection.create(
                in_features=predictor_features, out_features=n, streams=streams
            ),
            mismatch=Mismatch.create(
                n,
                predictor_features=predictor_features,
                gain_features=gain_features,
                mismatch_settings=mismatch_settings,
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
        """Pair the supplied observation with an inherited instrument and score its prediction.

        The supplied observation is already the target, so there is no learned observation map
        or local noise. The instrument loss fits the inherited instrument to the supplied
        observation. Save the outgoing message after any gain scaling.
        """
        inherited_instrument = inherit_instrument(self.instrument_map, predictor.instrument)
        local_variance = jnp.zeros_like(inherited_instrument.mean())

        gain_mean = None if gain is None else gain.observation.mean()
        if gain_mean is None:
            out_observation = observation
            out_instrument = inherited_instrument
        else:
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
            loss=mismatch_outputs.loss
            + instrument_loss(inherited_instrument, observation, gain_mean),
        )


class RecruiterFeature(eqx.Module):
    """Emit the gain that determines how strongly other features participate."""

    feature: SensationFeature

    @classmethod
    def create(
        cls,
        n: int,
        *,
        observation_content_features: int,
        predictor_features: int,
        credit_settings: CreditSettings,
        mismatch_settings: MismatchSettings,
        streams: Mapping[str, RngStream],
    ) -> Self:
        sizes = SIPFeatureSizes(
            observation_content=observation_content_features,
            predictor=predictor_features,
            gain=0,
        )
        return cls(
            feature=SensationFeature.create(
                n,
                sizes,
                controllable=True,
                credit_settings=credit_settings,
                mismatch_settings=mismatch_settings,
                streams=streams,
            )
        )

    def step(
        self,
        state: eqx.nn.State,
        key: KeyArray,
        *,
        observation_content: NormalNP,
        predictor: InstrumentedMessage,
        reconstruction: JaxRealArray,
        confounding: JaxRealArray,
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
            reconstruction=reconstruction,
            confounding=-confounding,
            streams=streams,
            inference=inference,
        )


class ProspectFeatureOutputs(SIPLossOutputs):
    """An instrumented reward predictor and the losses that train it."""

    message: InstrumentedMessage


class ProspectFeature(eqx.Module):
    """Supply an instrumented predictor that helps the critic learn what causes reward."""

    emitter: Emitter
    last_message_index: eqx.nn.StateIndex[InstrumentedMessage]

    @classmethod
    def create(
        cls,
        n: int,
        *,
        observation_content_features: int,
        predictor_features: int,
        credit_settings: CreditSettings,
        streams: Mapping[str, RngStream],
    ) -> Self:
        observation_map = GaussianNeuralProjection.create(
            in_features=observation_content_features, out_features=n, streams=streams
        )
        instrument_map = GaussianNeuralProjection.create(
            in_features=predictor_features, out_features=n, streams=streams
        )
        baseline_map = GaussianNeuralProjection.create(
            in_features=observation_content_features, out_features=1, streams=streams
        )
        emitter = Emitter.create(
            observation_map=observation_map,
            instrument_map=instrument_map,
            credit=Credit.create(n, baseline_map, credit_settings=credit_settings),
            controllable=False,
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
        reconstruction: JaxRealArray,
        confounding: JaxRealArray,
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
            goal=reconstruction + confounding,
        )
        new_state = new_state.set(self.last_message_index, emitter_outputs.message)
        return new_state, ProspectFeatureOutputs(
            message=emitter_outputs.message,
            loss=emitter_outputs.loss,
        )


class TDFeatureOutputs(SIPLossOutputs):
    """The observed reward, the advantage signal, and the costs of predicting reward."""

    observation: NormalNP  # the supplied reward observation, passed through unchanged
    advantage: JaxRealArray  # TD error in reward, replaces observation mismatch
    reconstruction_cost: JaxRealArray  # scalar
    confounding_cost: JaxRealArray  # scalar
    uncertainty_loss: JaxRealArray  # scalar


class TDCosts(eqx.Module):
    """Prediction costs retained for the next prospect credit update."""

    reconstruction: JaxRealArray
    confounding: JaxRealArray


class TDFeature(eqx.Module):
    """Learn the value of observed reward and supply credit signals that guide the agent."""

    td_error: TDError
    last_costs_index: eqx.nn.StateIndex[TDCosts]
    last_advantage_index: eqx.nn.StateIndex[JaxRealArray]

    @classmethod
    def create(
        cls,
        n: int,
        *,
        predictor_features: int,
        td_settings: TDSettings,
        streams: Mapping[str, RngStream],
    ) -> Self:
        zero_cost = jnp.asarray(0.0, dtype=jnp.float64)
        return cls(
            td_error=TDError.create(
                n,
                predictor_features=predictor_features,
                td_settings=td_settings,
                streams=streams,
            ),
            last_costs_index=eqx.nn.StateIndex(
                TDCosts(reconstruction=zero_cost, confounding=zero_cost)
            ),
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
        """Score the reward prediction and retain the signals for the next credit update.

        Save each prediction cost and the difference between the bootstrap target mean and
        the delayed prediction mean.
        """
        new_state, td_outputs = self.td_error.step(
            state,
            reward_observation=reward_observation,
            predictor=predictor,
            streams=streams,
            inference=inference,
        )
        new_state = new_state.set(
            self.last_costs_index,
            TDCosts(
                reconstruction=td_outputs.reconstruction_cost,
                confounding=td_outputs.confounding_cost,
            ),
        )
        new_state = new_state.set(self.last_advantage_index, td_outputs.advantage)
        return new_state, TDFeatureOutputs(
            observation=reward_observation,
            advantage=td_outputs.advantage,
            reconstruction_cost=td_outputs.reconstruction_cost,
            confounding_cost=td_outputs.confounding_cost,
            uncertainty_loss=td_outputs.uncertainty_loss,
            loss=td_outputs.loss,
        )
