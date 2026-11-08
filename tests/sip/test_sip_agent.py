from collections.abc import Mapping

import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
from tjax import RngStream

from cem.sip import (
    CreditSettings,
    EnvironmentFeatures,
    MismatchSettings,
    SIPAgent,
    SIPAgentSettings,
    TDSettings,
)
from cem.structure.graph import LossBalancedParameter, SummedGradientParameter

from .tools import CREDIT_SETTINGS, _message

ENV_OBS = 2
ENV_ACT = 2
INTERIOR = (3,)


def _agent(streams: Mapping[str, RngStream]) -> SIPAgent:
    return SIPAgent.create(
        SIPAgentSettings(
            environment=EnvironmentFeatures(observation_features=ENV_OBS, action_features=ENV_ACT),
            level_sizes=INTERIOR,
            credit_settings=CREDIT_SETTINGS,
        ),
        streams=streams,
    )


def test_sip_agent_settings_reach_their_circuits(streams: Mapping[str, RngStream]) -> None:
    settings = SIPAgentSettings(
        environment=EnvironmentFeatures(
            observation_features=ENV_OBS,
            action_features=ENV_ACT,
            reward_features=2,
        ),
        level_sizes=INTERIOR,
        prospect_sizes=(2, 4),
        credit_settings=CreditSettings(trace_decay_rate=0.25),
        mismatch_settings=MismatchSettings(
            prediction_hidden_features=5, witness_hidden_features=(7,)
        ),
        td_settings=TDSettings(
            discount=0.8, prediction_hidden_features=(9,), witness_hidden_features=8
        ),
    )
    agent = SIPAgent.create(settings, streams=streams)
    sensations = agent.sensations.interior_features
    intentions = agent.intentions.features
    prospects = agent.reward_critic.prospect_features
    for feature in (*sensations, *intentions, *prospects):
        assert jnp.isclose(
            feature.emitter.credit.trace_decay_rate.value, settings.credit_settings.trace_decay_rate
        )
    for feature in (*sensations, *prospects):
        observation_weight = feature.emitter.observation_map.layers[0].weight
        assert isinstance(observation_weight, LossBalancedParameter)
        assert observation_weight.loss_names == ("confounding", "reconstruction")

    mismatches = (
        agent.sensations.input_feature.mismatch,
        *(f.mismatch for f in sensations),
        *(f.mismatch for f in intentions),
    )
    for mismatch in mismatches:
        assert mismatch.prediction_map.layers[0].bias.value.shape == (5,)
        assert mismatch.witness_map.layers[0].bias.value.shape == (7,)
        prediction_weight = mismatch.prediction_map.layers[0].weight
        assert isinstance(prediction_weight, LossBalancedParameter)
        assert prediction_weight.loss_names == ("confounding", "reconstruction")
        assert isinstance(mismatch.witness_map.layers[0].weight, SummedGradientParameter)

    td_error = agent.reward_critic.td_feature.td_error
    assert jnp.isclose(td_error.discount.value, settings.td_settings.discount)
    assert td_error.prediction_map.layers[0].bias.value.shape == (9,)
    assert td_error.witness_map.layers[0].bias.value.shape == (8,)
    assert td_error.prediction_map.layers[-1].bias.value.shape == (6,)
    assert td_error.prediction_map.input_pool.dummy_mean.value.shape == (12,)
    td_prediction_weight = td_error.prediction_map.layers[0].weight
    assert isinstance(td_prediction_weight, LossBalancedParameter)
    assert td_prediction_weight.loss_names == ("confounding", "reconstruction", "uncertainty")
    assert isinstance(td_error.witness_map.layers[0].weight, SummedGradientParameter)


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
