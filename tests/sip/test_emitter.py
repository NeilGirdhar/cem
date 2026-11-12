from collections.abc import Mapping

import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
from efax import NormalNP
from jax import tree
from tjax import RngStream

from cem.npn.gaussian import GaussianNeuralProjection
from cem.sip.credit import Credit
from cem.sip.emitter import Emitter, ObservationModel

from .tools import CREDIT_SETTINGS, _message

GOAL_NAMES = ("reconstruction", "confounding")


def _emitter(
    n: int,
    *,
    content_dim: int,
    instrument_dim: int,
    controllable: bool,
    streams: Mapping[str, RngStream],
) -> Emitter:
    def projection(in_features: int, out_features: int) -> GaussianNeuralProjection:
        return GaussianNeuralProjection.create(
            in_features=in_features, out_features=out_features, hidden_features=(), streams=streams
        )

    m = len(GOAL_NAMES)
    observation_model = ObservationModel(
        observation_map=projection(content_dim, n), share_map=projection(content_dim, m * n)
    )
    credit = Credit.create(n, m, projection(content_dim, m), credit_settings=CREDIT_SETTINGS)
    return Emitter.create(
        observation_model,
        projection(instrument_dim, n),
        credit,
        goal_names=GOAL_NAMES,
        controllable=controllable,
    )


def _inherited(emitter: Emitter, predictor_instrument: NormalNP) -> tuple[jnp.ndarray, jnp.ndarray]:
    inherited = emitter.instrument_map.infer(predictor_instrument).to_precision_parametrization()
    return inherited.mean, inherited.precision


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
    noised_inst_mean = outputs.noised_instrument.mean()
    inst_displacement = noised_inst_mean - inherited_mean
    # `observation` is gain-scaled with gain=1 here, so its mean equals the noised
    # raw mean; `raw_observation.mean` is pre-noise.
    raw_mean = outputs.raw_observation.mean()
    obs_mean = outputs.message.observation.mean()
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


def test_component_loss_bundles_its_credit(streams: Mapping[str, RngStream]) -> None:
    """Each component's loss combines its mean, precision, and shrinkage credit."""
    n = 2
    emitter = _emitter(n, content_dim=1, instrument_dim=n, controllable=False, streams=streams)
    state = eqx.nn.State(emitter)
    content = _message(jnp.asarray([0.5]), jnp.ones(1))
    goals = {"reconstruction": jnp.asarray(1.0), "confounding": jnp.asarray(-0.5)}

    def loss_fn(em: Emitter) -> tuple[jnp.ndarray, tuple[eqx.nn.State, jnp.ndarray]]:
        new_state, outputs = em.step(
            state,
            jr.key(0),
            observation_content=content,
            predictor_instrument=_message(jnp.asarray([0.1, -0.2]), jnp.asarray([1.0, 1.0])),
            gain_observation=_message(jnp.ones(n), jnp.ones(n)),
            goals=goals,
        )
        assert set(outputs.balanced_losses) == set(GOAL_NAMES)
        return outputs.balanced_losses["confounding"], (
            new_state,
            outputs.credit_outputs.goal_prediction_error[1],
        )

    (_, (new_state, error)), grads = eqx.filter_value_and_grad(loss_fn, has_aux=True)(emitter)
    traces = new_state.get(emitter.trace_index)
    curvature_trace = new_state.get(emitter.credit.trace_index).curvature[1]

    def shrinkage_loss(model: ObservationModel) -> jnp.ndarray:
        projection = eqx.tree_at(lambda e: e.observation_projection.model, emitter, model)
        projected, _ = projection.observation_projection.infer(content)
        mean = projected.raw_observation.mean()
        return 0.5 * jnp.sum(curvature_trace * jnp.square(mean))

    shrinkage_gradient = eqx.filter_grad(shrinkage_loss)(emitter.observation_projection.model)
    expected = tree.map(
        lambda mean, precision, shrinkage: (
            -goals["confounding"] * mean - error * precision + shrinkage
        ),
        traces.mean,
        traces.precision[1],
        shrinkage_gradient,
    )
    gradient = grads.observation_projection.model
    assert any(jnp.any(jnp.abs(leaf) > 0) for leaf in tree.leaves(gradient))
    assert tree.all(tree.map(jnp.allclose, gradient, expected))


def test_observation_projection_stops_gradients_at_its_input(
    streams: Mapping[str, RngStream],
) -> None:
    emitter = _emitter(2, content_dim=1, instrument_dim=2, controllable=False, streams=streams)
    content = _message(jnp.asarray([0.5]), jnp.ones(1))

    def raw_mean(inputs: tuple[Emitter, NormalNP]) -> jnp.ndarray:
        model, observation_content = inputs
        projected, _ = model.observation_projection.infer(observation_content)
        mean = projected.raw_observation.mean()
        return jnp.sum(mean)

    emitter_gradient, content_gradient = eqx.filter_grad(raw_mean)((emitter, content))
    assert jnp.allclose(content_gradient.mean_times_precision, 0.0)
    assert jnp.allclose(content_gradient.negative_half_precision, 0.0)
    assert jnp.any(emitter_gradient.observation_map.layers[-1].bias.value != 0)


def test_instrument_loss_trains_only_the_instrument_map(
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
            goals={"reconstruction": jnp.asarray(0.0), "confounding": jnp.asarray(0.0)},
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
