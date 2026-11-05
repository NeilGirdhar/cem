from collections.abc import Mapping

import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
from tjax import RngStream

from cem.sip.features import ProspectFeature
from cem.sip.messages import InstrumentedMessage, concat_messages
from cem.sip.reward_critic import RewardCritic

from .tools import _message


def _prospect(n: int, streams: Mapping[str, RngStream]) -> ProspectFeature:
    return ProspectFeature.create(
        n,
        observation_content_features=2,
        predictor_features=2,
        streams=streams,
    )


def test_prospect_feature_has_no_mismatch(streams: Mapping[str, RngStream]) -> None:
    feature = _prospect(2, streams)
    assert hasattr(feature, "emitter")
    assert not hasattr(feature, "mismatch")


def test_prospect_feature_outputs_have_expected_shapes(
    streams: Mapping[str, RngStream],
) -> None:
    n = 3
    feature = _prospect(n, streams)
    state = eqx.nn.State(feature)
    _, outputs = feature.step(
        state,
        jr.key(0),
        observation_content=_message(jnp.asarray([0.3, -0.1]), jnp.ones(2)),
        predictor_instrument=_message(jnp.asarray([0.5, 0.1]), jnp.ones(2)),
        gain=None,
        goal=jnp.asarray(0.5),
    )
    assert outputs.message.observation.mean_times_precision.shape == (n,)
    assert outputs.message.instrument.mean_times_precision.shape == (n,)
    assert outputs.loss.shape == ()


def test_prospect_message_is_independent_of_the_local_noise_key(
    streams: Mapping[str, RngStream],
) -> None:
    feature = _prospect(2, streams)
    messages = []
    for seed in (0, 1):
        _, outputs = feature.step(
            eqx.nn.State(feature),
            jr.key(seed),
            observation_content=_message(jnp.asarray([0.3, -0.1]), jnp.ones(2)),
            predictor_instrument=_message(jnp.asarray([0.5, 0.1]), jnp.ones(2)),
            gain=None,
            goal=jnp.asarray(0.5),
        )
        messages.append(outputs.message)
    assert eqx.tree_equal(*messages)


def test_prospect_feature_stores_message_into_state(
    streams: Mapping[str, RngStream],
) -> None:
    n = 2
    feature = _prospect(n, streams)
    state = eqx.nn.State(feature)
    new_state, outputs = feature.step(
        state,
        jr.key(0),
        observation_content=_message(jnp.asarray([0.3, -0.1]), jnp.ones(2)),
        predictor_instrument=_message(jnp.asarray([0.5, 0.1]), jnp.ones(2)),
        gain=None,
        goal=jnp.asarray(0.5),
    )
    stored = new_state.get(feature.last_message_index)
    assert jnp.allclose(
        stored.observation.mean_times_precision, outputs.message.observation.mean_times_precision
    )
    assert jnp.allclose(
        stored.instrument.mean_times_precision, outputs.message.instrument.mean_times_precision
    )


def _critic(streams: Mapping[str, RngStream]) -> RewardCritic:
    return RewardCritic.create(
        [2, 3],
        observation_content_features=[2, 2],
        predictor_features=[2, 2],
        reward_features=1,
        streams=streams,
    )


def test_reward_critic_td_predictor_dim_equals_sum_of_prospect_sizes(
    streams: Mapping[str, RngStream],
) -> None:
    critic = _critic(streams)
    # TDError's witness_map input is 2 * (predictor_features + gain_features)
    # = 2 * sum(prospect_sizes); check the first layer's input dim.
    witness = critic.td_feature.td_error.witness_map.layers[0].weight.value
    assert witness.shape[-1] == 2 * (2 + 3)


def test_reward_critic_step_runs_end_to_end(
    streams: Mapping[str, RngStream],
) -> None:
    critic = _critic(streams)
    state = eqx.nn.State(critic)
    _, outputs = critic.step(
        state,
        jr.key(0),
        observation_content=[
            _message(jnp.asarray([0.3, -0.1]), jnp.ones(2)),
            _message(jnp.asarray([0.2, 0.4]), jnp.ones(2)),
        ],
        predictor_instruments=[
            _message(jnp.asarray([0.5, 0.1]), jnp.ones(2)),
            _message(jnp.asarray([-0.3, 0.2]), jnp.ones(2)),
        ],
        reward_observation=_message(jnp.asarray([0.6]), jnp.ones(1)),
        streams=streams,
        inference=True,
    )
    prospect_sizes = [2, 3]
    assert len(outputs.prospects) == len(prospect_sizes)
    for out, size in zip(outputs.prospects, prospect_sizes, strict=True):
        assert out.message.observation.mean_times_precision.shape == (size,)
    assert outputs.td.observation.mean_times_precision.shape == (1,)
    assert jnp.isfinite(outputs.loss)


def test_reward_critic_feeds_prospect_outputs_as_td_predictor(
    streams: Mapping[str, RngStream],
) -> None:
    """The TD feature sees the concatenated prospect observations/instruments as its predictors."""
    critic = _critic(streams)
    key = jr.key(0)
    observation_content = [
        _message(jnp.asarray([0.3, -0.1]), jnp.ones(2)),
        _message(jnp.asarray([0.2, 0.4]), jnp.ones(2)),
    ]
    predictor_instruments = [
        _message(jnp.asarray([0.5, 0.1]), jnp.ones(2)),
        _message(jnp.asarray([-0.3, 0.2]), jnp.ones(2)),
    ]
    reward_observation = _message(jnp.asarray([0.6]), jnp.ones(1))

    state = eqx.nn.State(critic)
    _, outputs = critic.step(
        state,
        key,
        observation_content=observation_content,
        predictor_instruments=predictor_instruments,
        reward_observation=reward_observation,
        streams=streams,
        inference=True,
    )

    # Re-run prospects manually with the same keys, then TD directly.
    keys = jr.split(key, len(critic.prospect_features))
    fresh = eqx.nn.State(critic)
    goal = fresh.get(critic.td_feature.last_dissonance_index)
    prospects: list = []
    for feature, k, oc, pi in zip(
        critic.prospect_features, keys, observation_content, predictor_instruments, strict=True
    ):
        fresh, out = feature.step(
            fresh,
            k,
            observation_content=oc,
            predictor_instrument=pi,
            gain=None,
            goal=goal,
        )
        prospects.append(out)
    expected_predictor = InstrumentedMessage(
        observation=concat_messages(*[p.message.observation for p in prospects]),
        instrument=concat_messages(*[p.message.instrument for p in prospects]),
    )
    _, td_out = critic.td_feature.step(
        fresh,
        reward_observation=reward_observation,
        predictor=expected_predictor,
        streams=streams,
        inference=True,
    )
    assert jnp.allclose(outputs.td.advantage, td_out.advantage)
    assert jnp.allclose(outputs.td.dissonance, td_out.dissonance)


def test_reward_critic_feeds_td_dissonance_back_as_prospect_goal(
    streams: Mapping[str, RngStream],
) -> None:
    """Prospect features consume the TD feature's previous-step dissonance as goal."""
    critic = _critic(streams)
    state = eqx.nn.State(critic)

    # Seed the TD feature's last_dissonance_index with a known non-zero value.
    stored_goal = jnp.asarray(0.75, dtype=jnp.float64)
    state = state.set(critic.td_feature.last_dissonance_index, stored_goal)

    # Run with the same key; the prospects should see `stored_goal` as goal.
    key = jr.key(0)
    observation_content = [
        _message(jnp.asarray([0.3, -0.1]), jnp.ones(2)),
        _message(jnp.asarray([0.2, 0.4]), jnp.ones(2)),
    ]
    predictor_instruments = [
        _message(jnp.asarray([0.5, 0.1]), jnp.ones(2)),
        _message(jnp.asarray([-0.3, 0.2]), jnp.ones(2)),
    ]
    reward_observation = _message(jnp.asarray([0.6]), jnp.ones(1))
    _, critic_outputs = critic.step(
        state,
        key,
        observation_content=observation_content,
        predictor_instruments=predictor_instruments,
        reward_observation=reward_observation,
        streams=streams,
        inference=True,
    )

    # Rerun the first prospect directly with goal=stored_goal; outputs should match.
    keys = jr.split(key, len(critic.prospect_features))
    fresh = eqx.nn.State(critic)
    _, direct = critic.prospect_features[0].step(
        fresh,
        keys[0],
        observation_content=observation_content[0],
        predictor_instrument=predictor_instruments[0],
        gain=None,
        goal=stored_goal,
    )
    assert jnp.allclose(
        critic_outputs.prospects[0].message.observation.mean_times_precision,
        direct.message.observation.mean_times_precision,
    )
