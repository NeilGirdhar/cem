"""SIP reward critic: n prospects plus one TD feature."""

from collections.abc import Mapping, Sequence
from typing import Self

import equinox as eqx
import jax.random as jr
from efax import NormalNP
from tjax import KeyArray, RngStream

from cem.sip.features import (
    ProspectFeature,
    ProspectFeatureOutputs,
    TDFeature,
    TDFeatureOutputs,
)
from cem.sip.losses import SIPLossOutputs, combine_losses
from cem.sip.messages import InstrumentedMessage, concat_instrumented_messages
from cem.sip.settings import CreditSettings, TDSettings


class RewardCriticOutputs(SIPLossOutputs):
    """Prospect messages, reward credit signals, and their training losses."""

    prospects: tuple[ProspectFeatureOutputs, ...]
    td: TDFeatureOutputs


class RewardCritic(eqx.Module):
    """Learn to predict return from prospects and supply credit for reward-seeking behaviour.

    See `reward-critic` in the thesis.
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
        credit_settings: CreditSettings,
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
                credit_settings=credit_settings,
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
        predictor_instruments: Sequence[NormalNP],
        gains: Sequence[InstrumentedMessage | None] | None = None,
        reward_observation: NormalNP,
        streams: Mapping[str, RngStream],
        inference: bool,
    ) -> tuple[eqx.nn.State, RewardCriticOutputs]:
        """Run the prospects, then the TD feature.

        Each prospect is advanced with its own `observation_content[i]`,
        `predictor_instruments[i]`, optional `gains[i]`, and the TD feature's
        previous reconstruction and confounding costs. The prospect outputs are concatenated on
        each channel to form the TD feature's predictor.
        """
        if len(observation_content) != len(self.prospect_features):
            msg = "observation_content must have one entry per prospect"
            raise ValueError(msg)
        if len(predictor_instruments) != len(self.prospect_features):
            msg = "predictor_instruments must have one entry per prospect"
            raise ValueError(msg)
        if gains is None:
            gains = [None] * len(self.prospect_features)
        if len(gains) != len(self.prospect_features):
            msg = "gains must have one entry per prospect"
            raise ValueError(msg)
        costs = state.get(self.td_feature.last_costs_index)
        keys = jr.split(key, len(self.prospect_features))
        prospect_outputs: list[ProspectFeatureOutputs] = []
        for feature, feature_key, obs_content, pred_inst, gain in zip(
            self.prospect_features,
            keys,
            observation_content,
            predictor_instruments,
            gains,
            strict=True,
        ):
            state, out = feature.step(
                state,
                feature_key,
                observation_content=obs_content,
                predictor_instrument=pred_inst,
                gain=gain,
                reconstruction=costs.reconstruction,
                confounding=costs.confounding,
            )
            prospect_outputs.append(out)
        td_predictor = concat_instrumented_messages(*[p.message for p in prospect_outputs])
        state, td_out = self.td_feature.step(
            state,
            reward_observation=reward_observation,
            predictor=td_predictor,
            streams=streams,
            inference=inference,
        )
        return state, RewardCriticOutputs(
            prospects=tuple(prospect_outputs),
            td=td_out,
            balanced_losses=combine_losses(
                td_out.balanced_losses, *(out.balanced_losses for out in prospect_outputs)
            ),
            fixed_weight_loss=sum(
                (out.fixed_weight_loss for out in prospect_outputs), td_out.fixed_weight_loss
            ),
        )
