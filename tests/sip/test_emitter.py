from collections.abc import Mapping

import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
from efax import NormalNP
from tjax import RngStream

from cem.npn.gaussian import GaussianNPN, normal_from_mean_and_presence
from cem.sip.credit import Credit
from cem.sip.emitter import Emitter
from cem.sip.messages import mean_and_precision


def _message(mean: jnp.ndarray, precision: jnp.ndarray) -> NormalNP:
    return normal_from_mean_and_presence(mean, precision)


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
    credit = Credit.create(n, baseline_map, trace_decay_rate=0.0)
    return Emitter.create(
        observation_map=observation_map,
        instrument_map=instrument_map,
        credit=credit,
        controllable=controllable,
    )


def _inherited(
    emitter: Emitter, predictor_instruments: NormalNP
) -> tuple[jnp.ndarray, jnp.ndarray]:
    inst_mean, inst_precision = mean_and_precision(predictor_instruments)
    inherited = emitter.instrument_map.infer(inst_mean, inst_precision)
    return mean_and_precision(inherited)


def test_local_variance_matches_inherited_moments(
    streams: Mapping[str, RngStream],
) -> None:
    """V_local equals max(0, var_inh - mean_inh^2) at the actual inherited moments."""
    n = 3
    emitter = _emitter(n, content_dim=1, instrument_dim=4, controllable=True, streams=streams)
    predictor_instruments = _message(
        jnp.array([0.4, -0.3, 0.1, 0.2]), jnp.array([1.0, 2.0, 0.5, 1.0])
    )
    inherited_mean, inherited_precision = _inherited(emitter, predictor_instruments)
    expected = jnp.maximum(0.0, jnp.reciprocal(inherited_precision) - jnp.square(inherited_mean))
    state = eqx.nn.State(emitter)
    _, outputs = emitter.step(
        state,
        jr.key(0),
        observation_content=_message(jnp.array([0.3]), jnp.ones(1)),
        predictor_instruments=predictor_instruments,
        gain_observation=_message(jnp.ones(n), jnp.ones(n)),
        goal=jnp.asarray(0.0),
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
    predictor_instruments = _message(jnp.array([1.0, -1.0]), jnp.array([0.5, 0.25]))
    inherited_mean, _ = _inherited(emitter, predictor_instruments)
    _, outputs = emitter.step(
        state,
        jr.key(7),
        observation_content=_message(jnp.array([0.3]), jnp.ones(1)),
        predictor_instruments=predictor_instruments,
        gain_observation=_message(jnp.ones(n), jnp.ones(n)),
        goal=jnp.asarray(0.0),
    )
    noised_inst_mean, _ = mean_and_precision(outputs.noised_instrument)
    inst_displacement = noised_inst_mean - inherited_mean
    # `observation` is gain-scaled with gain=1 here, so its mean equals the noised
    # raw mean; `raw_observation.mean` is pre-noise.
    raw_mean, _ = mean_and_precision(outputs.raw_observation)
    obs_mean, _ = mean_and_precision(outputs.observation)
    obs_displacement = obs_mean - raw_mean
    assert jnp.allclose(inst_displacement, obs_displacement)


def test_non_controllable_passes_through(streams: Mapping[str, RngStream]) -> None:
    """Without noise, `observation` equals `evidence_scale(raw_observation, gain)`."""
    n = 2
    emitter = _emitter(n, content_dim=1, instrument_dim=n, controllable=False, streams=streams)
    state = eqx.nn.State(emitter)
    gain = jnp.array([2.0, 0.5])
    _, outputs = emitter.step(
        state,
        jr.key(0),
        observation_content=_message(jnp.array([0.4]), jnp.ones(1)),
        predictor_instruments=_message(jnp.array([0.1, 0.2]), jnp.array([1.0, 1.0])),
        gain_observation=_message(gain, jnp.ones(n)),
        goal=jnp.asarray(0.0),
    )
    assert jnp.allclose(outputs.local_variance, 0.0)
    assert jnp.allclose(
        outputs.observation.mean_times_precision,
        gain * outputs.raw_observation.mean_times_precision,
    )
    assert jnp.allclose(
        outputs.observation.negative_half_precision,
        gain * outputs.raw_observation.negative_half_precision,
    )


def test_credit_cotangent_reaches_observation_map(
    streams: Mapping[str, RngStream],
) -> None:
    """The promoting link sends a nonzero cotangent to the observation map's params."""
    n = 2
    emitter = _emitter(n, content_dim=1, instrument_dim=n, controllable=False, streams=streams)
    state = eqx.nn.State(emitter)

    def loss_fn(em: Emitter) -> jnp.ndarray:
        _, outputs = em.step(
            state,
            jr.key(0),
            observation_content=_message(jnp.array([0.5]), jnp.ones(1)),
            predictor_instruments=_message(jnp.array([0.1, -0.2]), jnp.array([1.0, 1.0])),
            gain_observation=_message(jnp.ones(n), jnp.ones(n)),
            goal=jnp.asarray(1.0),
        )
        return outputs.loss

    grads = eqx.filter_grad(loss_fn)(emitter)
    first_layer_weight_grad = grads.observation_map.layers[0].weight.value
    assert jnp.any(jnp.abs(first_layer_weight_grad) > 0)
