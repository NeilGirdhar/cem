"""SIP reward critic: n prospects plus one TD feature.

See `reward-critic` in `~/src/typst/thesis.typ`.
"""

from collections.abc import Mapping, Sequence
from typing import Self

import equinox as eqx
import jax.random as jr
from efax import NormalNP
from tjax import JaxRealArray, KeyArray, RngStream

from cem.sip.features import (
    ProspectFeature,
    ProspectFeatureOutputs,
    TDFeature,
    TDFeatureOutputs,
)
from cem.sip.messages import InstrumentedMessage, concat_messages


class RewardCriticOutputs(eqx.Module):
    """One step of the reward critic."""

    prospects: tuple[ProspectFeatureOutputs, ...]
    td: TDFeatureOutputs
    loss: JaxRealArray  # scalar; sum of prospect losses and the TD loss


class RewardCritic(eqx.Module):
    """A bank of `ProspectFeature`s feeding one `TDFeature`.

    The prospects run in parallel (they do not exchange messages with each
    other); their outgoing `observation` and `instrument` are concatenated
    into one `InstrumentedMessage` that becomes the TD feature's predictor.
    The TD feature's `dissonance` (reconstruction + confounding) is the
    prospect features' goal, fed back with a one-step delay through the
    TD feature's `last_dissonance_index`.
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
        discount: float = 1.0,
        controllable: bool = True,
        trace_decay_rate: float = 0.0,
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
                controllable=controllable,
                trace_decay_rate=trace_decay_rate,
                streams=streams,
            )
            for size, obs_cf, pred_f in zip(
                prospect_sizes,
                observation_content_features,
                predictor_features,
                strict=True,
            )
        )
        td = TDFeature.create(
            reward_features,
            predictor_features=sum(prospect_sizes),
            discount=discount,
            streams=streams,
        )
        return cls(prospect_features=prospects, td_feature=td)

    def step(
        self,
        state: eqx.nn.State,
        key: KeyArray,
        *,
        observation_content: Sequence[NormalNP],
        predictor_instruments: Sequence[NormalNP],
        reward_observation: NormalNP,
        streams: Mapping[str, RngStream],
        inference: bool,
    ) -> tuple[eqx.nn.State, RewardCriticOutputs]:
        """Run the prospects, then the TD feature.

        Each prospect is advanced with its own `observation_content[i]`,
        `predictor_instruments[i]`, and the TD feature's previous-step
        `dissonance` as goal (read from `last_dissonance_index`). The
        prospect outputs are concatenated on each channel to form the TD
        feature's predictor.
        """
        if len(observation_content) != len(self.prospect_features):
            msg = "observation_content must have one entry per prospect"
            raise ValueError(msg)
        if len(predictor_instruments) != len(self.prospect_features):
            msg = "predictor_instruments must have one entry per prospect"
            raise ValueError(msg)
        prospect_goal = state.get(self.td_feature.last_dissonance_index)
        keys = jr.split(key, len(self.prospect_features))
        prospect_outputs: list[ProspectFeatureOutputs] = []
        for feature, feature_key, obs_content, pred_inst in zip(
            self.prospect_features,
            keys,
            observation_content,
            predictor_instruments,
            strict=True,
        ):
            state, out = feature.step(
                state,
                feature_key,
                observation_content=obs_content,
                predictor_instruments=pred_inst,
                goal=prospect_goal,
            )
            prospect_outputs.append(out)
        td_predictors = InstrumentedMessage(
            observation=concat_messages(*[p.observation for p in prospect_outputs]),
            instrument=concat_messages(*[p.instrument for p in prospect_outputs]),
        )
        state, td_out = self.td_feature.step(
            state,
            reward_observation=reward_observation,
            predictors=td_predictors,
            streams=streams,
            inference=inference,
        )
        total_loss = td_out.loss
        for out in prospect_outputs:
            total_loss += out.loss
        return state, RewardCriticOutputs(
            prospects=tuple(prospect_outputs),
            td=td_out,
            loss=total_loss,
        )
