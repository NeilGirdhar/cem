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

from cem.npn.gaussian import normal_from_mean_and_presence
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

    - Sensations run first. Layer 0's content is the environment
      observation concatenated with the previous step's action (the
      eligibility trace, read from the bottom intention's
      `last_message_index`). Each sensation level's external predictor is
      the matching intention level's previous-step `last_message_index`
      (its efference).
    - Intentions run top-down. Readouts are the current sensations'
      observations (the affordance readout — @affordance in the thesis).
      Each intention level's external predictor is the matching sensation's
      just-emitted instrumented message. The Credit goal (advantage) at
      every level is the TD feature's previous-step advantage scalar.
    - The reward critic runs last. There is one prospect per sensation
      feature and one per intention feature (`2 * (1 + len(level_sizes))`
      total, sensations first, then intentions). Each prospect sees the
      same diffuse readout (the concatenation of every sensation and
      intention observation), but its predictor instrument is the
      instrument half of the single feature it corresponds to — so each
      prospect answers "did this specific component cause reward?" and the
      TD witness isolates per-feature causal credit. The critic predicts
      a reward vector of width `environment_reward_features + 2 * L`
      (with `L = 1 + len(level_sizes)` the number of sensation levels):
      environment reward first, then one curiosity channel per sensation
      (its confounding cost) and one tracking channel per sensation (its
      reconstruction cost). Prospect goals and TD advantage feed back with
      a one-step delay through the TD feature's `last_dissonance_index`
      and `last_advantage_index`. The intention-side goal is the scalar
      sum of the augmented advantage broadcast to every intention level;
      the learned-controllable-view refinement is tracked in
      `~/src/typst/todo.txt` under "Controllable pseudo-rewards".
    - The agent's action is the bottom intention's outgoing observation,
      which the environment consumes.
    """

    sensations: Sensations
    intentions: Intentions
    reward_critic: RewardCritic

    @classmethod
    def create(
        cls,
        level_sizes: Sequence[int],
        *,
        env_observation_features: int,
        env_action_features: int,
        prospect_sizes: Sequence[int] | None = None,
        environment_reward_features: int = 1,
        discount: float = 1.0,
        streams: Mapping[str, RngStream],
    ) -> Self:
        """Build a SIP agent with aligned sensation and intention stacks.

        `env_observation_features` and `env_action_features` are the
        externally fixed input and output widths of the agent (the
        environment's observation and action dims). `level_sizes[i]` is the
        shared sensation/intention width for *interior* level `i + 1`;
        `level_sizes=[]` builds a single-level agent with no interior.

        Internals:

        - Sensation level 0 width = `env_observation_features +
          env_action_features` (it consumes `concat(env observation,
          previous action)` — the eligibility trace).
        - Intention level 0 width = `env_action_features` (its outgoing
          observation is the action).
        - Sensation and intention level `i + 1` share width `level_sizes[i]`.

        `prospect_sizes` defaults to the full per-level widths (one width
        per level, sensation and intention side sharing): one prospect per
        feature, with the same width on each side.

        `environment_reward_features` is the width of the environment
        reward vector. The critic internally predicts
        `environment_reward_features + 2 * L` channels (with
        `L = 1 + len(level_sizes)`): the environment reward, then one
        curiosity channel per sensation (confounding cost) and one tracking
        channel per sensation (reconstruction cost).
        """
        assert env_observation_features > 0
        assert env_action_features > 0
        level_sizes = tuple(level_sizes)
        sensation_sizes = (env_observation_features + env_action_features, *level_sizes)
        intention_sizes = (env_action_features, *level_sizes)
        # One prospect per sensation feature and one per intention feature: each
        # prospect answers "did this component's experimental noise cause reward?",
        # with the TD witness attributing credit through the specific instrument.
        per_side_sizes = tuple(prospect_sizes) if prospect_sizes is not None else sensation_sizes
        if len(per_side_sizes) != len(sensation_sizes):
            msg = "prospect_sizes must have one entry per level (1 + len(level_sizes))"
            raise ValueError(msg)
        prospect_sizes_full = per_side_sizes + per_side_sizes
        diffuse_content = sum(sensation_sizes) + sum(intention_sizes)
        n_levels = len(sensation_sizes)
        # Augment reward with one curiosity (confounding cost) and one tracking
        # (reconstruction cost) channel per sensation feature.
        reward_features = environment_reward_features + 2 * n_levels
        sensations = Sensations.create(
            sensation_sizes,
            # external predictor i carries the matching intention's efference.
            predictor_features=list(intention_sizes),
            streams=streams,
        )
        intentions = Intentions.create(
            intention_sizes,
            # external predictor i carries the matching sensation's message.
            predictor_features=list(sensation_sizes),
            # readout i is the matching sensation's observation.
            readout_features=list(sensation_sizes),
            streams=streams,
        )
        reward_critic = RewardCritic.create(
            prospect_sizes_full,
            # diffuse readout: every sensation and every intention observation.
            observation_content_features=[diffuse_content] * len(prospect_sizes_full),
            # specific instrument links: sensation levels first, then intention levels.
            predictor_features=list(sensation_sizes) + list(intention_sizes),
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
        environment_reward_observation: NormalNP,
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
        # instrumented message (its efference). Sensation level 0's input is
        # concat(env observation, previous action) — the eligibility trace that
        # tells the stack which action was just taken. On the first step the
        # previous action is the standard-normal observation half of
        # `InstrumentedMessage.initial`, which keeps the raw-observation path
        # well-posed.
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

        # Prospects: one per sensation feature and one per intention feature.
        # Each sees the same diffuse readout (every sensation and intention
        # observation) as its state, but its specific instrument is the feature
        # it's attributing reward credit to. Order: all sensations, then all
        # intentions, matching SIPAgent.create's prospect layout.
        diffuse = concat_messages(
            *[out.message.observation for out in sensation_outputs],
            *[out.message.observation for out in intention_outputs],
        )
        prospect_observation_content = [diffuse] * len(self.reward_critic.prospect_features)
        prospect_predictor_instruments = [out.message.instrument for out in sensation_outputs] + [
            out.message.instrument for out in intention_outputs
        ]
        # Augment reward with intrinsic channels: curiosity is the confounding
        # cost at every sensation feature, tracking is the reconstruction cost
        # at every sensation feature. Placeholder precision 1 on the intrinsic
        # channels; loss scaling across environment reward, curiosity, and
        # tracking is tracked in `docs/sip-plan.md`.
        curiosity = jnp.stack([out.confounding_cost for out in sensation_outputs])
        tracking = jnp.stack([out.reconstruction_cost for out in sensation_outputs])
        intrinsic_precision = jnp.ones_like(curiosity)
        reward_observation = concat_messages(
            environment_reward_observation,
            normal_from_mean_and_presence(curiosity, intrinsic_precision),
            normal_from_mean_and_presence(tracking, intrinsic_precision),
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
            sensations=sensation_outputs,
            intentions=intention_outputs,
            critic=critic_outputs,
            action=intention_outputs[0].message.observation,
        )
