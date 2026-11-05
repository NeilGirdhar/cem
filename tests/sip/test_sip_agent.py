from collections.abc import Mapping

import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
from tjax import RngStream

from cem.sip.sip_agent import SIPAgent

from .tools import _message

ENV_OBS = 2
ENV_ACT = 2
INTERIOR = [3]


def _agent(streams: Mapping[str, RngStream]) -> SIPAgent:
    return SIPAgent.create(
        INTERIOR,
        env_observation_features=ENV_OBS,
        env_action_features=ENV_ACT,
        environment_reward_features=1,
        streams=streams,
    )


def test_sip_agent_create_matches_coupling_shapes(
    streams: Mapping[str, RngStream],
) -> None:
    agent = _agent(streams)
    n_levels = 1 + len(INTERIOR)
    # Sensations have one SensationInputFeature and len(INTERIOR) interior SIPFeatures.
    assert len(agent.sensations.interior_features) == len(INTERIOR)
    # Intentions have n_levels SIPFeatures.
    assert len(agent.intentions.features) == n_levels
    predictor_widths = [
        feature.mismatch.prediction_map.input_pool.dummy_mean.value.size
        for feature in agent.intentions.features
    ]
    assert predictor_widths == [0, ENV_ACT, *INTERIOR[:-1]]
    # One prospect per sensation feature and one per intention feature.
    assert len(agent.reward_critic.prospect_features) == 2 * n_levels


def test_sip_agent_step_returns_action_at_bottom_intention(
    streams: Mapping[str, RngStream],
) -> None:
    agent = _agent(streams)
    state = eqx.nn.State(agent)
    _, outputs = agent.step(
        state,
        jr.key(0),
        observation=_message(jnp.ones(ENV_OBS), jnp.ones(ENV_OBS)),
        environment_reward_observation=_message(jnp.asarray([0.3]), jnp.ones(1)),
        streams=streams,
        inference=True,
    )
    bottom_intention = outputs.intentions[0].message.observation
    assert jnp.allclose(outputs.action.mean_times_precision, bottom_intention.mean_times_precision)
    assert outputs.action.mean_times_precision.shape == (ENV_ACT,)


def test_sip_agent_step_produces_one_output_per_level(
    streams: Mapping[str, RngStream],
) -> None:
    agent = _agent(streams)
    state = eqx.nn.State(agent)
    _, outputs = agent.step(
        state,
        jr.key(0),
        observation=_message(jnp.ones(ENV_OBS), jnp.ones(ENV_OBS)),
        environment_reward_observation=_message(jnp.asarray([0.3]), jnp.ones(1)),
        streams=streams,
        inference=True,
    )
    n_levels = 1 + len(INTERIOR)
    assert len(outputs.sensations) == n_levels
    assert len(outputs.intentions) == n_levels
    sensation_sizes = [ENV_OBS + ENV_ACT, *INTERIOR]
    intention_sizes = [ENV_ACT, *INTERIOR]
    for i, size in enumerate(sensation_sizes):
        assert outputs.sensations[i].message.observation.mean_times_precision.shape == (size,)
    for i, size in enumerate(intention_sizes):
        assert outputs.intentions[i].message.observation.mean_times_precision.shape == (size,)
