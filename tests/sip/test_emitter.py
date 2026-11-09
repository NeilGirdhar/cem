from collections.abc import Mapping

import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
from efax import NormalNP
from jax import tree
from tjax import RngStream

from cem.npn.gaussian import GaussianNPN
from cem.sip.credit import Credit
from cem.sip.emitter import Emitter
from cem.sip.messages import mean_and_precision

from .tools import CREDIT_SETTINGS, _message


def _emitter(
    n: int,
    *,
    content_dim: int,
    instrument_dim: int,
    controllable: bool,
    streams: Mapping[str, RngStream],
) -> Emitter:
    observation_map = GaussianNPN.create(
        in_features=content_dim, out_features=n, hidden_features=(), streams=streams
    )
    instrument_map = GaussianNPN.create(
        in_features=instrument_dim, out_features=n, hidden_features=(), streams=streams
    )
    baseline_map = GaussianNPN.create(
        in_features=content_dim, out_features=1, hidden_features=(), streams=streams
    )
    credit = Credit.create(n, baseline_map, credit_settings=CREDIT_SETTINGS)
    return Emitter.create(
        observation_map=observation_map,
        instrument_map=instrument_map,
        credit=credit,
        controllable=controllable,
    )


def _inherited(emitter: Emitter, predictor_instrument: NormalNP) -> tuple[jnp.ndarray, jnp.ndarray]:
    inst_mean, inst_precision = mean_and_precision(predictor_instrument)
    inherited = emitter.instrument_map.infer(inst_mean, inst_precision)
    return mean_and_precision(inherited)


def test_local_variance_matches_inherited_moments(
    streams: Mapping[str, RngStream],
) -> None:
    """V_local equals max(0, var_inh - mean_inh^2) at the actual inherited moments."""
    n = 3
    emitter = _emitter(n, content_dim=1, instrument_dim=4, controllable=True, streams=streams)
    predictor_instrument = _message(
        jnp.asarray([0.4, -0.3, 0.1, 0.2]), jnp.asarray([1.0, 2.0, 0.5, 1.0])
    )
    inherited_mean, inherited_precision = _inherited(emitter, predictor_instrument)
    expected = jnp.maximum(0.0, jnp.reciprocal(inherited_precision) - jnp.square(inherited_mean))
    state = eqx.nn.State(emitter)
    _, outputs = emitter.step(
        state,
        jr.key(0),
        observation_content=_message(jnp.asarray([0.3]), jnp.ones(1)),
        predictor_instrument=predictor_instrument,
        gain_observation=_message(jnp.ones(n), jnp.ones(n)),
        goals={"reconstruction": jnp.asarray(0.0), "confounding": jnp.asarray(0.0)},
    )
    assert jnp.allclose(outputs.local_variance, expected)


def test_same_displacement_shifts_both_means(streams: Mapping[str, RngStream]) -> None:
    """One fresh displacement shifts the raw observation and inherited instrument alike.

    `(noised_inst.mean - inherited.mean)` and `(raw.mean - obs_map(content).mean)`
    must be equal. The `raw_observation` the emitter returns is pre-noise.
    """
    n = 2
    emitter = _emitter(n, content_dim=1, instrument_dim=n, controllable=True, streams=streams)
    state = eqx.nn.State(emitter)
    predictor_instrument = _message(jnp.asarray([1.0, -1.0]), jnp.asarray([0.5, 0.25]))
    inherited_mean, _ = _inherited(emitter, predictor_instrument)
    _, outputs = emitter.step(
        state,
        jr.key(7),
        observation_content=_message(jnp.asarray([0.3]), jnp.ones(1)),
        predictor_instrument=predictor_instrument,
        gain_observation=_message(jnp.ones(n), jnp.ones(n)),
        goals={"reconstruction": jnp.asarray(0.0), "confounding": jnp.asarray(0.0)},
    )
    noised_inst_mean, _ = mean_and_precision(outputs.noised_instrument)
    inst_displacement = noised_inst_mean - inherited_mean
    # `observation` is gain-scaled with gain=1 here, so its mean equals the noised
    # raw mean; `raw_observation.mean` is pre-noise.
    raw_mean, _ = mean_and_precision(outputs.raw_observation)
    obs_mean, _ = mean_and_precision(outputs.message.observation)
    obs_displacement = obs_mean - raw_mean
    assert jnp.allclose(inst_displacement, obs_displacement)


def test_non_controllable_passes_through(streams: Mapping[str, RngStream]) -> None:
    """Without noise, `observation` equals `evidence_scale(raw_observation, gain)`."""
    n = 2
    emitter = _emitter(n, content_dim=1, instrument_dim=n, controllable=False, streams=streams)
    state = eqx.nn.State(emitter)
    gain = jnp.asarray([2.0, 0.5])
    _, outputs = emitter.step(
        state,
        jr.key(0),
        observation_content=_message(jnp.asarray([0.4]), jnp.ones(1)),
        predictor_instrument=_message(jnp.asarray([0.1, 0.2]), jnp.asarray([1.0, 1.0])),
        gain_observation=_message(gain, jnp.ones(n)),
        goals={"reconstruction": jnp.asarray(0.0), "confounding": jnp.asarray(0.0)},
    )
    assert jnp.allclose(outputs.local_variance, 0.0)
    assert jnp.allclose(
        outputs.message.observation.mean_times_precision,
        gain * outputs.raw_observation.mean_times_precision,
    )
    assert jnp.allclose(
        outputs.message.observation.negative_half_precision,
        gain * outputs.raw_observation.negative_half_precision,
    )


def test_credit_cotangent_reaches_observation_map(
    streams: Mapping[str, RngStream],
) -> None:
    """Named credit applies the mean trace to the observation map's parameters."""
    n = 2
    emitter = _emitter(n, content_dim=1, instrument_dim=n, controllable=False, streams=streams)
    state = eqx.nn.State(emitter)

    def loss_fn(em: Emitter) -> tuple[jnp.ndarray, tuple[dict[str, jnp.ndarray], jnp.ndarray]]:
        new_state, outputs = em.step(
            state,
            jr.key(0),
            observation_content=_message(jnp.asarray([0.5]), jnp.ones(1)),
            predictor_instrument=_message(jnp.asarray([0.1, -0.2]), jnp.asarray([1.0, 1.0])),
            gain_observation=_message(jnp.ones(n), jnp.ones(n)),
            goals={"reconstruction": jnp.asarray(1.0), "confounding": jnp.asarray(0.0)},
        )
        mean_trace = new_state.get(em.trace_index).mean
        return outputs.balanced_losses["reconstruction"], (
            outputs.balanced_losses,
            mean_trace.layers[0].weight.value,
        )

    (_, (losses, mean_weight_trace)), grads = eqx.filter_value_and_grad(loss_fn, has_aux=True)(
        emitter
    )
    assert set(losses) == {"reconstruction", "confounding"}
    assert jnp.allclose(losses["confounding"], 0.0)
    first_layer_weight_grad = grads.observation_map.layers[0].weight.value
    assert jnp.any(jnp.abs(first_layer_weight_grad) > 0)
    assert jnp.allclose(first_layer_weight_grad, -mean_weight_trace)


def test_observation_projection_stops_gradients_at_its_input(
    streams: Mapping[str, RngStream],
) -> None:
    emitter = _emitter(2, content_dim=1, instrument_dim=2, controllable=False, streams=streams)
    content = _message(jnp.asarray([0.5]), jnp.ones(1))

    def raw_mean(inputs: tuple[Emitter, NormalNP]) -> jnp.ndarray:
        model, observation_content = inputs
        observation, _ = model.observation_projection.infer(observation_content)
        mean, _ = mean_and_precision(observation)
        return jnp.sum(mean)

    emitter_gradient, content_gradient = eqx.filter_grad(raw_mean)((emitter, content))
    assert jnp.allclose(content_gradient.mean_times_precision, 0.0)
    assert jnp.allclose(content_gradient.negative_half_precision, 0.0)
    assert jnp.any(emitter_gradient.observation_map.layers[-1].bias.value != 0)


def test_instrument_mismatch_trains_only_the_instrument_map(
    streams: Mapping[str, RngStream],
) -> None:
    """The emitter's loss trains its instrument map but not the parents' instruments or gain."""
    n = 2
    emitter = _emitter(n, content_dim=1, instrument_dim=n, controllable=True, streams=streams)
    state = eqx.nn.State(emitter)

    def loss_fn(
        em: Emitter, predictor_instrument: NormalNP, gain_observation: NormalNP
    ) -> jnp.ndarray:
        _, outputs = em.step(
            state,
            jr.key(0),
            observation_content=_message(jnp.asarray([0.5]), jnp.ones(1)),
            predictor_instrument=predictor_instrument,
            gain_observation=gain_observation,
            goals={"reward": jnp.asarray(0.0)},
        )
        return outputs.loss

    model_gradient, instrument_gradient, gain_gradient = eqx.filter_grad(
        lambda args: loss_fn(*args)
    )(
        (
            emitter,
            _message(jnp.asarray([0.1, -0.2]), jnp.asarray([1.0, 2.0])),
            _message(jnp.asarray([0.6, 0.9]), jnp.ones(n)),
        )
    )
    assert jnp.any(jnp.abs(model_gradient.instrument_map.layers[-1].weight.value) > 0)
    assert all(
        jnp.allclose(leaf, 0.0) for leaf in tree.leaves((instrument_gradient, gain_gradient))
    )
