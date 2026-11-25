"""SIP reward critic: n prospects plus one TD feature."""

from collections.abc import Mapping, Sequence
from typing import Self

import equinox as eqx
import jax.random as jr
from efax import NormalNP
from tjax import KeyArray, RngStream

from cem.sip.emitter import EmitterEmission
from cem.sip.features import (
    ProspectFeature,
    ProspectFeatureOutputs,
    TDFeature,
    TDFeatureOutputs,
)
from cem.sip.losses import SIPLossOutputs
from cem.sip.messages import InstrumentedMessage, concat_instrumented_messages
from cem.sip.settings import TDSettings


class RewardCriticOutputs(SIPLossOutputs):
    """Prospect messages, reward credit signals, and their training losses."""

    prospects: tuple[ProspectFeatureOutputs, ...]
    td: TDFeatureOutputs


class RewardCritic(eqx.Module):
    """Learn to predict return from prospects and supply credit for reward-seeking behaviour.

    See `reward-prediction` in the thesis.
    """

    prospect_features: tuple[ProspectFeature, ...]
    td_feature: TDFeature

    @classmethod
    def create(
        cls,
        prospect_sizes: Sequence[int],
        *,
        observation_content_features: Sequence[int],
        predictor_features: Sequence[int],
        reward_features: int = 1,
        td_settings: TDSettings,
        streams: Mapping[str, RngStream],
    ) -> Self:
        """Build `len(prospect_sizes)` prospects plus one TD feature.

        `observation_content_features[i]` and `predictor_features[i]` are the
        observation-content and predictor-instrument dimensions for prospect
        `i`. The TD feature's predictor dimension is `sum(prospect_sizes)`
        (the concatenated prospect outputs).
        """
        if len(prospect_sizes) == 0:
            msg = "prospect_sizes must be non-empty"
            raise ValueError(msg)
        if len(observation_content_features) != len(prospect_sizes):
            msg = "observation_content_features must have one entry per prospect"
            raise ValueError(msg)
        if len(predictor_features) != len(prospect_sizes):
            msg = "predictor_features must have one entry per prospect"
            raise ValueError(msg)
        prospects = tuple(
            ProspectFeature.create(
                size,
                observation_content_features=obs_cf,
                predictor_features=pred_f,
                streams=streams,
            )
            for size, obs_cf, pred_f in zip(
                prospect_sizes,
                observation_content_features,
                predictor_features,
                strict=True,
            )
        )
        td_feature = TDFeature.create(
            reward_features,
            predictor_features=sum(prospect_sizes),
            td_settings=td_settings,
            streams=streams,
        )
        return cls(prospect_features=prospects, td_feature=td_feature)

    def step(
        self,
        state: eqx.nn.State,
        key: KeyArray,
        *,
        observation_content: Sequence[NormalNP],
        predictors: Sequence[InstrumentedMessage],
        gains: Sequence[InstrumentedMessage | None] | None = None,
        reward_observation: NormalNP,
        streams: Mapping[str, RngStream],
        inference: bool,
    ) -> tuple[eqx.nn.State, RewardCriticOutputs]:
        """Emit the prospects, run the TD feature, then credit the prospects.

        Each prospect emits with its own `observation_content[i]`, `predictors[i]`, and optional
        `gains[i]`. The emitted messages are concatenated on each channel to form the
        TD feature's predictor. The TD feature's reconstruction loss and confounding objective
        from this step score the prospects' previous emission, so each prospect takes its credit
        after the TD feature runs.
        """
        if len(observation_content) != len(self.prospect_features):
            msg = "observation_content must have one entry per prospect"
            raise ValueError(msg)
        if len(predictors) != len(self.prospect_features):
            msg = "predictors must have one entry per prospect"
            raise ValueError(msg)
        if gains is None:
            gains = [None] * len(self.prospect_features)
        if len(gains) != len(self.prospect_features):
            msg = "gains must have one entry per prospect"
            raise ValueError(msg)
        keys = jr.split(key, len(self.prospect_features))
        emissions: list[EmitterEmission] = []
        for feature, feature_key, obs_content, predictor, gain in zip(
            self.prospect_features,
            keys,
            observation_content,
            predictors,
            gains,
            strict=True,
        ):
            state, emission = feature.emit(
                state,
                feature_key,
                observation_content=obs_content,
                predictor=predictor,
                gain=gain,
            )
            emissions.append(emission)
        td_predictor = concat_instrumented_messages(*[e.message for e in emissions])
        state, td_out = self.td_feature.step(
            state,
            reward_observation=reward_observation,
            predictor=td_predictor,
            streams=streams,
            inference=inference,
        )
        prospect_outputs: list[ProspectFeatureOutputs] = []
        for feature, emission in zip(self.prospect_features, emissions, strict=True):
            state, out = feature.credit(
                state,
                emission,
                reconstruction=td_out.reconstruction_loss,
                confounding=td_out.confounding_objective,
            )
            prospect_outputs.append(out)
        return state, RewardCriticOutputs(
            prospects=tuple(prospect_outputs),
            td=td_out,
            loss=sum((out.loss for out in prospect_outputs), td_out.loss),
        )
