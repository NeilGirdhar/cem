"""SIP agent: Sensations, Intentions, and RewardCritic wired together.

See the agent architecture section of the thesis.
"""

from collections.abc import Mapping
from typing import Self

import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
from efax import NormalNP, NormalPP
from tjax import KeyArray, RngStream

from cem.sip.features import SIPFeatureOutputs
from cem.sip.intentions import Intentions
from cem.sip.losses import SIPLossOutputs
from cem.sip.messages import concat_messages
from cem.sip.reward_critic import RewardCritic, RewardCriticOutputs
from cem.sip.sensations import Sensations
from cem.sip.settings import SIPAgentSettings


class SIPAgentOutputs(SIPLossOutputs):
    """The next action and the sensation, intention, and critic results for that step."""

    sensations: tuple[SIPFeatureOutputs, ...]
    intentions: tuple[SIPFeatureOutputs, ...]
    critic: RewardCriticOutputs
    action: NormalNP  # bottom intention's observation


class SIPAgent(eqx.Module):
    """Interpret observations and choose actions that seek higher expected reward.

    See `causal-agent-arch-figure` in the thesis.
    """

    sensations: Sensations
    intentions: Intentions
    reward_critic: RewardCritic

    @classmethod
    def create(
        cls,
        agent_settings: SIPAgentSettings,
        *,
        streams: Mapping[str, RngStream],
    ) -> Self:
        """Build a SIP agent for the given environment widths and interior level widths.

        `level_sizes` lists shared sensation and intention widths above the environment level.
        `prospect_sizes` includes the environment level and gives the width on each side.
        When omitted, the prospect widths on both sides match the sensation widths.
        """
        environment = agent_settings.environment
        assert environment.observation_features > 0
        assert environment.action_features > 0
        sensation_sizes = (
            environment.observation_features + environment.action_features,
            *agent_settings.level_sizes,
        )
        intention_sizes = (environment.action_features, *agent_settings.level_sizes)
        # Give each sensation and intention feature its own prospect so the critic can
        # use its instrument to estimate that feature's effect on reward.
        per_side_sizes = (
            sensation_sizes
            if agent_settings.prospect_sizes is None
            else agent_settings.prospect_sizes
        )
        if len(per_side_sizes) != len(sensation_sizes):
            msg = "prospect_sizes must have one entry per level (1 + len(level_sizes))"
            raise ValueError(msg)
        prospect_sizes_full = per_side_sizes + per_side_sizes
        diffuse_content = sum(sensation_sizes) + sum(intention_sizes)
        n_levels = len(sensation_sizes)
        # Augment reward with one curiosity (confounding cost) and one tracking
        # (negated reconstruction cost) channel per sensation feature.
        reward_features = environment.reward_features + 2 * n_levels
        sensations = Sensations.create(
            sensation_sizes,
            # Each sensation receives the previous message from the matching intention.
            predictor_features=list(intention_sizes),
            credit_settings=agent_settings.credit_settings,
            mismatch_settings=agent_settings.mismatch_settings,
            streams=streams,
        )
        intentions = Intentions.create(
            intention_sizes,
            readout_features=sensation_sizes,
            credit_settings=agent_settings.credit_settings,
            mismatch_settings=agent_settings.mismatch_settings,
            streams=streams,
        )
        reward_critic = RewardCritic.create(
            prospect_sizes_full,
            # The diffuse readout gives every prospect the full observation context.
            observation_content_features=[diffuse_content] * len(prospect_sizes_full),
            # Match one instrument source to each prospect, with sensations before intentions.
            predictor_features=list(sensation_sizes) + list(intention_sizes),
            reward_features=reward_features,
            credit_settings=agent_settings.credit_settings,
            td_settings=agent_settings.td_settings,
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
        environment_reward_observation: NormalNP,
        streams: Mapping[str, RngStream],
        inference: bool,
    ) -> tuple[eqx.nn.State, SIPAgentOutputs]:
        """Choose the next action and update the critic from the latest observation and reward.

        Sensations interpret the observation in the context of the previous action.
        Intentions use those sensations and the previous critic advantage. The critic
        then scores the environment reward together with the current sensation costs.
        """
        key_s, key_i, key_c = jr.split(key, 3)
        n_levels = 1 + len(self.sensations.interior_features)
        intention_features = self.intentions.features
        if len(intention_features) != n_levels:
            msg = "sensation and intention stacks must have the same number of levels"
            raise ValueError(msg)

        # The bottom sensation receives an efference copy of the previous action alongside
        # the environment observation. The first step uses a standard normal for the action
        # so the raw observation has positive precision on those channels.
        sensation_predictors = [
            state.get(intention_features[i].last_message_index) for i in range(n_levels)
        ]
        last_action = state.get(intention_features[0].last_message_index).observation
        sensation_input = concat_messages(observation, last_action)
        state, sensation_outputs = self.sensations.infer(
            state,
            key_s,
            observation=sensation_input,
            predictors=sensation_predictors,
            streams=streams,
            inference=inference,
        )

        intention_readouts = [out.message.observation for out in sensation_outputs]
        # The critic stores environment reward channels, then curiosity and tracking.
        last_advantage = state.get(self.reward_critic.td_feature.last_advantage_index)
        reward_end = last_advantage.shape[0] - 2 * n_levels
        curiosity_end = reward_end + n_levels
        state, intention_outputs = self.intentions.infer(
            state,
            key_i,
            readouts=intention_readouts,
            reward=jnp.sum(last_advantage[:reward_end]),
            curiosity=jnp.sum(last_advantage[reward_end:curiosity_end]),
            tracking=jnp.sum(last_advantage[curiosity_end:]),
            streams=streams,
            inference=inference,
        )

        # All prospects share the observation context. Each gets an instrument from one
        # feature so its contribution to reward prediction can track that feature's variation.
        diffuse = concat_messages(
            *[out.message.observation for out in sensation_outputs],
            *[out.message.observation for out in intention_outputs],
        )
        prospect_observation_content = [diffuse] * len(self.reward_critic.prospect_features)
        prospect_predictor_instruments = [out.message.instrument for out in sensation_outputs] + [
            out.message.instrument for out in intention_outputs
        ]
        # Each sensation contributes its confounding cost as curiosity and its negated
        # reconstruction cost as tracking, so tracking rewards predictable sensation. Both enter
        # the reward observation with unit precision.
        curiosity = jnp.stack([out.confounding_cost for out in sensation_outputs])
        tracking = -jnp.stack([out.reconstruction_cost for out in sensation_outputs])
        intrinsic_precision = jnp.ones_like(curiosity)
        reward_observation = concat_messages(
            environment_reward_observation,
            NormalPP(curiosity, intrinsic_precision).to_nat(),
            NormalPP(tracking, intrinsic_precision).to_nat(),
        )
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
            loss=sum(
                (out.loss for out in (*sensation_outputs, *intention_outputs)), critic_outputs.loss
            ),
            sensations=sensation_outputs,
            intentions=intention_outputs,
            critic=critic_outputs,
            action=intention_outputs[0].message.observation,
        )
