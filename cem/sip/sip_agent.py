"""SIP agent: Sensations, Intentions, and RewardCritic wired together.

See `causal-agent-arch-figure` and the agent architecture section of
`~/src/typst/thesis.typ`.
"""

from collections.abc import Mapping, Sequence
from typing import Self

import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
from efax import NormalNP
from tjax import KeyArray, RngStream

from cem.sip.features import SIPFeatureOutputs
from cem.sip.intentions import Intentions
from cem.sip.messages import concat_messages
from cem.sip.reward_critic import RewardCritic, RewardCriticOutputs
from cem.sip.sensations import Sensations


class SIPAgentOutputs(eqx.Module):
    """One step of a SIP agent."""

    sensations: tuple[SIPFeatureOutputs, ...]
    intentions: tuple[SIPFeatureOutputs, ...]
    critic: RewardCriticOutputs
    action: NormalNP  # bottom intention's observation


class SIPAgent(eqx.Module):
    """One SIP agent: `Sensations`, `Intentions`, and `RewardCritic`.

    The three sub-circuits share one `eqx.nn.State` and shuttle their
    messages through it:

    - Sensations run first with the environment observation as layer-0
      content. Each sensation level's external predictor is the matching
      intention level's previous-step `last_message_index` (its efference).
    - Intentions run top-down. Readouts are the current sensations'
      observations (the affordance readout — @affordance in the thesis).
      Each intention level's external predictor is the matching sensation's
      just-emitted instrumented message. The Credit goal (advantage) at
      every level is the TD feature's previous-step advantage scalar.
    - The reward critic runs last. Each prospect sees a diffuse readout —
      the concatenation of every sensation and intention observation — and
      its predictor instruments come from the matching sensation and
      intention level. Prospect goals and TD advantage feed back with a
      one-step delay through the TD feature's `last_dissonance_index` and
      `last_advantage_index`.
    - The agent's action is the bottom intention's outgoing observation,
      which the environment consumes.

    The simplest configuration assumes one sensation level per intention
    level (`level_sizes` common to both), and one prospect per level with
    its predictor instruments drawn from the matching sensation / intention.
    """

    sensations: Sensations
    intentions: Intentions
    reward_critic: RewardCritic

    @classmethod
    def create(
        cls,
        level_sizes: Sequence[int],
        *,
        prospect_sizes: Sequence[int] | None = None,
        reward_features: int = 1,
        discount: float = 1.0,
        streams: Mapping[str, RngStream],
    ) -> Self:
        """Build a SIP agent with matching sensation and intention stacks.

        `level_sizes` is the per-level width shared by sensations and
        intentions (`sizes[0]` is the environment-observation dimension and
        the bottom-intention action dimension). `prospect_sizes` defaults
        to `level_sizes`: one prospect per level with the same width.
        """
        level_sizes = tuple(level_sizes)
        if len(level_sizes) == 0:
            msg = "level_sizes must be non-empty"
            raise ValueError(msg)
        prospect_sizes = tuple(prospect_sizes) if prospect_sizes is not None else level_sizes
        if len(prospect_sizes) != len(level_sizes):
            msg = "prospect_sizes must have one entry per level"
            raise ValueError(msg)
        diffuse_content = 2 * sum(level_sizes)
        sensations = Sensations.create(
            level_sizes,
            # external predictor i carries the matching intention's efference.
            predictor_features=list(level_sizes),
            streams=streams,
        )
        intentions = Intentions.create(
            level_sizes,
            # external predictor i carries the matching sensation's message.
            predictor_features=list(level_sizes),
            # readout i is the matching sensation's observation.
            readout_features=list(level_sizes),
            streams=streams,
        )
        reward_critic = RewardCritic.create(
            prospect_sizes,
            # diffuse readout: every sensation + every intention observation.
            observation_content_features=[diffuse_content] * len(prospect_sizes),
            # specific instrument links: matching sensation + intention level.
            predictor_features=[
                level_sizes[k] + level_sizes[k] for k in range(len(prospect_sizes))
            ],
            reward_features=reward_features,
            discount=discount,
            streams=streams,
        )
        return cls(
            sensations=sensations,
            intentions=intentions,
            reward_critic=reward_critic,
        )

    def step(
        self,
        state: eqx.nn.State,
        key: KeyArray,
        *,
        observation: NormalNP,
        reward_observation: NormalNP,
        streams: Mapping[str, RngStream],
        inference: bool,
    ) -> tuple[eqx.nn.State, SIPAgentOutputs]:
        """Run one agent step: sensations, intentions, reward critic."""
        key_s, key_i, key_c = jr.split(key, 3)
        n_levels = 1 + len(self.sensations.interior_features)
        intention_features = self.intentions.features
        if len(intention_features) != n_levels:
            msg = "sensation and intention stacks must have the same number of levels"
            raise ValueError(msg)

        # Sensations: external predictor at level i = intention i's previous
        # instrumented message (its efference).
        sensation_predictors = [
            state.get(intention_features[i].last_message_index) for i in range(n_levels)
        ]
        state, sensation_outputs = self.sensations.infer(
            state,
            key_s,
            observation=observation,
            predictors=sensation_predictors,
            streams=streams,
            inference=inference,
        )

        # Intentions: readout = sensation obs; predictor = sensation inst message;
        # advantage = TD's previous-step advantage scalar (broadcast per layer).
        intention_readouts = [out.message.observation for out in sensation_outputs]
        intention_predictors = [out.message for out in sensation_outputs]
        last_advantage = state.get(self.reward_critic.td_feature.last_advantage_index)
        advantage_scalar = jnp.sum(last_advantage)
        advantages = [advantage_scalar for _ in range(n_levels)]
        state, intention_outputs = self.intentions.infer(
            state,
            key_i,
            readouts=intention_readouts,
            predictors=intention_predictors,
            advantages=advantages,
            streams=streams,
            inference=inference,
        )

        # Prospects: diffuse readout = concat of all sensation + intention obs;
        # predictor instruments = matching sensation + intention instrument.
        diffuse = concat_messages(
            *[out.message.observation for out in sensation_outputs],
            *[out.message.observation for out in intention_outputs],
        )
        prospect_observation_content = [diffuse] * len(self.reward_critic.prospect_features)
        prospect_predictor_instruments = [
            concat_messages(
                sensation_outputs[k].message.instrument, intention_outputs[k].message.instrument
            )
            for k in range(len(self.reward_critic.prospect_features))
        ]
        state, critic_outputs = self.reward_critic.step(
            state,
            key_c,
            observation_content=prospect_observation_content,
            predictor_instruments=prospect_predictor_instruments,
            reward_observation=reward_observation,
            streams=streams,
            inference=inference,
        )

        return state, SIPAgentOutputs(
            sensations=sensation_outputs,
            intentions=intention_outputs,
            critic=critic_outputs,
            action=intention_outputs[0].message.observation,
        )
