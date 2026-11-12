from collections.abc import Mapping, Sequence
from dataclasses import replace

import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
import pytest
from efax import NormalNP, NormalPP
from tjax import KeyArray, RngStream

from cem.sip import (
    CreditSettings,
    EnvironmentFeatures,
    InstrumentedMessage,
    MismatchSettings,
    RewardCritic,
    RewardCriticOutputs,
    SIPActionConfiguration,
    SIPAgent,
    SIPAgentSettings,
    SIPModel,
    SIPObservation,
    TDSettings,
)
from cem.structure.graph import FixedParameter, LearnableParameter
from cem.structure.problem import DataSource, Problem, ProblemState
from cem.structure.solution.training_solution import TrainingSolution

from .tools import CREDIT_SETTINGS, _i_message, _message

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
        assert isinstance(observation_weight, LearnableParameter)

    mismatches = (
        agent.sensations.input_feature.mismatch,
        *(f.mismatch for f in sensations),
        *(f.mismatch for f in intentions),
    )
    for mismatch in mismatches:
        assert mismatch.prediction_map.layers[0].bias.value.shape == (5,)
        assert mismatch.witness_map.perceptron.layers[0].bias.value.shape == (7,)
        prediction_weight = mismatch.prediction_map.layers[0].weight
        assert isinstance(prediction_weight, LearnableParameter)
        assert isinstance(mismatch.witness_map.perceptron.layers[0].weight, LearnableParameter)

    td_error = agent.reward_critic.td_feature.td_error
    assert jnp.isclose(td_error.discount.value, settings.td_settings.discount)
    assert td_error.prediction_map.layers[0].bias.value.shape == (9,)
    assert td_error.witness_map.perceptron.layers[0].bias.value.shape == (8,)
    assert td_error.prediction_map.layers[-1].bias.value.shape == (6,)
    assert td_error.prediction_map.input_pool.dummy_mean.value.shape == (12,)
    td_prediction_weight = td_error.prediction_map.layers[0].weight
    assert isinstance(td_prediction_weight, LearnableParameter)
    assert isinstance(td_error.witness_map.perceptron.layers[0].weight, LearnableParameter)


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


def test_tracking_rewards_low_reconstruction_cost(
    streams: Mapping[str, RngStream], monkeypatch: pytest.MonkeyPatch
) -> None:
    """The reward observation ends with curiosity and the negated reconstruction costs."""
    agent = _agent(streams)
    captured: list[NormalNP] = []
    original_step = RewardCritic.step

    def capture(  # ruff: ignore[too-many-arguments]
        self: RewardCritic,
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
        captured.append(reward_observation)
        return original_step(
            self,
            state,
            key,
            observation_content=observation_content,
            predictor_instruments=predictor_instruments,
            gains=gains,
            reward_observation=reward_observation,
            streams=streams,
            inference=inference,
        )

    monkeypatch.setattr(RewardCritic, "step", capture)
    _, outputs = agent.step(
        eqx.nn.State(agent),
        jr.key(0),
        observation=_message(jnp.ones(ENV_OBS), jnp.ones(ENV_OBS)),
        environment_reward_observation=_message(jnp.asarray([0.3]), jnp.ones(1)),
        streams=streams,
        inference=True,
    )
    n_levels = len(outputs.sensations)
    rewards = captured[0].mean()
    curiosity = jnp.stack([out.confounding_cost for out in outputs.sensations])
    reconstruction = jnp.stack([out.reconstruction_cost for out in outputs.sensations])
    assert jnp.allclose(rewards[-2 * n_levels : -n_levels], curiosity)
    assert jnp.allclose(rewards[-n_levels:], -reconstruction)


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


class _ProblemState(ProblemState):
    observation: SIPObservation


class _DataSource(DataSource):
    example: _ProblemState

    def initial_problem_state(self, example_key: KeyArray) -> _ProblemState:
        return self.example


class _Problem(Problem):
    data_source: _DataSource

    def create_data_source(self, *, inference: bool = False) -> DataSource:
        return self.data_source

    def extract_observation(self, state: ProblemState) -> SIPObservation:
        assert isinstance(state, _ProblemState)
        return state.observation


def test_shared_training_updates_sip_predictions(
    streams: Mapping[str, RngStream],
) -> None:
    model = SIPModel.create(_agent(streams))
    previous_intention = _i_message(
        NormalPP(jnp.ones(ENV_ACT), jnp.ones(ENV_ACT)),
        NormalPP(jnp.ones(ENV_ACT), jnp.ones(ENV_ACT)),
    )
    model = replace(
        model,
        initial_agent_state=FixedParameter(
            model.initial_state().set(
                model.agent.intentions.features[0].last_message_index, previous_intention
            )
        ),
    )
    observation = SIPObservation(
        observation=_message(jnp.ones(ENV_OBS), jnp.ones(ENV_OBS)),
        reward=_message(jnp.asarray([0.3]), jnp.ones(1)),
    )
    data_source = _DataSource(_ProblemState(observation))
    problem = _Problem(data_source)
    transforms = model.gradient_transformations(0.001)
    solution = TrainingSolution.create(problem, model, transforms)
    result = solution.inference.train_one_episode(
        1, jr.key(0), jr.key(1), transforms, data_source, problem, solution.solution_state
    )

    action = result.inference_result.model_configuration["action"]
    assert isinstance(action, SIPActionConfiguration)
    assert jnp.all(jnp.isfinite(action.action.mean_times_precision))

    updated = solution.inference.assemble_model(
        result.solution_state.dis_learnable_parameters.assembled()
    )
    assert isinstance(updated, SIPModel)
    old = model.agent.sensations.input_feature.mismatch
    new = updated.agent.sensations.input_feature.mismatch
    assert not jnp.allclose(
        new.prediction_map.layers[0].weight.value, old.prediction_map.layers[0].weight.value
    )
    assert not jnp.allclose(
        new.witness_map.perceptron.layers[0].weight.value,
        old.witness_map.perceptron.layers[0].weight.value,
    )
    old_td = model.agent.reward_critic.td_feature.td_error.prediction_map.layers[0].weight.value
    new_td = updated.agent.reward_critic.td_feature.td_error.prediction_map.layers[0].weight.value
    assert not jnp.allclose(new_td, old_td)
