from collections.abc import Mapping

import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
from tjax import RngStream

from cem.sip.sip_agent import SIPAgent

from .tools import _message


def _agent(streams: Mapping[str, RngStream]) -> SIPAgent:
    return SIPAgent.create([2, 3], reward_features=1, streams=streams)


def test_sip_agent_create_matches_coupling_shapes(
    streams: Mapping[str, RngStream],
) -> None:
    agent = _agent(streams)
    # Sensations have one SensationInputFeature + N-1 interior SIPFeatures.
    assert len(agent.sensations.interior_features) == 1
    # Intentions have N SIPFeatures.
    assert len(agent.intentions.features) == len([2, 3])
    # One prospect per level.
    assert len(agent.reward_critic.prospect_features) == len([2, 3])


def test_sip_agent_step_returns_action_at_bottom_intention(
    streams: Mapping[str, RngStream],
) -> None:
    agent = _agent(streams)
    state = eqx.nn.State(agent)
    _, outputs = agent.step(
        state,
        jr.key(0),
        observation=_message(jnp.ones(2), jnp.ones(2)),
        reward_observation=_message(jnp.asarray([0.3]), jnp.ones(1)),
        streams=streams,
        inference=True,
    )
    bottom_intention = outputs.intentions[0].message.observation
    assert jnp.allclose(outputs.action.mean_times_precision, bottom_intention.mean_times_precision)
    assert outputs.action.mean_times_precision.shape == (2,)


def test_sip_agent_step_produces_one_output_per_level(
    streams: Mapping[str, RngStream],
) -> None:
    level_sizes = [2, 3]
    agent = SIPAgent.create(level_sizes, reward_features=1, streams=streams)
    state = eqx.nn.State(agent)
    _, outputs = agent.step(
        state,
        jr.key(0),
        observation=_message(jnp.ones(level_sizes[0]), jnp.ones(level_sizes[0])),
        reward_observation=_message(jnp.asarray([0.3]), jnp.ones(1)),
        streams=streams,
        inference=True,
    )
    assert len(outputs.sensations) == len(level_sizes)
    assert len(outputs.intentions) == len(level_sizes)
    for i, size in enumerate(level_sizes):
        assert outputs.sensations[i].message.observation.mean_times_precision.shape == (size,)
        assert outputs.intentions[i].message.observation.mean_times_precision.shape == (size,)
