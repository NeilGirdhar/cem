"""Alternating optimization for real-valued SIP score circuits."""

from collections.abc import Callable, Mapping

import equinox as eqx
import jax
import jax.numpy as jnp
from tjax import JaxRealArray, RngStream

from cem.sip.emitter import SIPEmitter
from cem.sip.explanatory_coupling import ExplanatoryCoupling
from cem.sip.objectives import purification_loss, witness_loss
from cem.sip.score import SIPScore
from cem.sip.td_error import SIPTDError, TDErrorOutput


class SIPTrainingHistory(eqx.Module):
    """Objective values recorded during alternating SIP optimization."""

    purification_losses: JaxRealArray
    witness_losses: JaxRealArray


def _zero_tree(value: object) -> object:
    return jax.tree.map(
        lambda leaf: jnp.zeros_like(leaf) if eqx.is_array(leaf) else leaf,
        value,
    )


def train_instrument_map(
    emitter: SIPEmitter,
    observation: JaxRealArray,
    predictor_instruments: JaxRealArray,
    *,
    steps: int,
    learning_rate: float,
    checkpoint: Callable[[int, SIPEmitter, JaxRealArray], None] | None = None,
    checkpoint_interval: int = 1,
) -> tuple[SIPEmitter, JaxRealArray]:
    """Train an emitter's instrument map against its observed value.

    The instrument score compares the inherited instrument with the observation. Its
    suppressing cotangent updates only the instrument map, implementing the emitter's
    first-stage regression from parent instruments to its observation.
    """
    if steps < 1:
        msg = "steps must be positive"
        raise ValueError(msg)
    if learning_rate < 0.0:
        msg = "learning_rate must be nonnegative"
        raise ValueError(msg)
    if observation.shape[-1] != emitter.observation_features:
        msg = "observation dimensions do not match the emitter"
        raise ValueError(msg)
    if predictor_instruments.shape[-1] != emitter.predictor_instrument_features:
        msg = "predictor instrument dimensions do not match the emitter"
        raise ValueError(msg)
    if checkpoint_interval < 1:
        msg = "checkpoint_interval must be positive"
        raise ValueError(msg)

    def objective(current: SIPEmitter) -> JaxRealArray:
        prediction = current.infer_inherited_instrument(predictor_instruments)
        score = prediction - observation
        return 0.5 * jnp.mean(jnp.sum(jnp.square(score), axis=-1))

    losses: list[JaxRealArray] = []
    if checkpoint is not None:
        checkpoint(0, emitter, objective(emitter))
    for step in range(steps):
        loss, gradients = eqx.filter_value_and_grad(objective)(emitter)
        emitter = eqx.apply_updates(
            emitter,
            jax.tree.map(lambda value: -learning_rate * value, gradients),
        )
        losses.append(loss)
        if checkpoint is not None and ((step + 1) % checkpoint_interval == 0 or step + 1 == steps):
            checkpoint(step + 1, emitter, loss)

    return emitter, jnp.stack(losses)


def train_score_adversarial(  # ruff: ignore[too-many-arguments]
    score: SIPScore,
    observation: JaxRealArray,
    predictor_observations: JaxRealArray,
    predictor_instruments: JaxRealArray,
    gain: JaxRealArray,
    *,
    steps: int,
    predictor_learning_rate: float,
    witness_learning_rate: float,
    confounding_weight: float = 1.0,
    streams: Mapping[str, RngStream],
) -> tuple[SIPScore, SIPTrainingHistory]:
    """Train predictor and witness paths with alternating bounded objectives."""
    if steps < 1:
        msg = "steps must be positive"
        raise ValueError(msg)
    if predictor_learning_rate < 0.0 or witness_learning_rate < 0.0:
        msg = "learning rates must be nonnegative"
        raise ValueError(msg)

    purification_losses: list[JaxRealArray] = []
    witness_losses: list[JaxRealArray] = []
    for _ in range(steps):

        def predictor_objective(current: SIPScore) -> JaxRealArray:
            output = current.infer(
                observation,
                predictor_observations,
                predictor_instruments,
                gain,
                streams=streams,
                inference=True,
            )
            return purification_loss(output, confounding_weight=confounding_weight)

        predictor_value, predictor_gradients = eqx.filter_value_and_grad(predictor_objective)(score)
        predictor_gradients = eqx.tree_at(
            lambda gradients: gradients.predictor.witness_map,
            predictor_gradients,
            _zero_tree(predictor_gradients.predictor.witness_map),
        )
        score = eqx.apply_updates(
            score,
            jax.tree.map(
                lambda value: -predictor_learning_rate * value,
                predictor_gradients,
            ),
        )

        def witness_objective(current: SIPScore) -> JaxRealArray:
            output = current.infer(
                observation,
                predictor_observations,
                predictor_instruments,
                gain,
                streams=streams,
                inference=True,
            )
            return witness_loss(output)

        witness_value, witness_gradients = eqx.filter_value_and_grad(witness_objective)(score)
        witness_gradients = eqx.tree_at(
            lambda gradients: gradients.predictor.prediction_map,
            witness_gradients,
            _zero_tree(witness_gradients.predictor.prediction_map),
        )
        score = eqx.apply_updates(
            score,
            jax.tree.map(
                lambda value: -witness_learning_rate * value,
                witness_gradients,
            ),
        )
        purification_losses.append(predictor_value)
        witness_losses.append(witness_value)

    return score, SIPTrainingHistory(
        purification_losses=jnp.stack(purification_losses),
        witness_losses=jnp.stack(witness_losses),
    )


def train_explanatory_coupling_adversarial(  # ruff: ignore[too-many-arguments]
    coupling: ExplanatoryCoupling,
    innovation: JaxRealArray,
    goal: JaxRealArray,
    source_gain: JaxRealArray,
    parent_instruments: JaxRealArray,
    target_observation: JaxRealArray,
    target_gain: JaxRealArray,
    *,
    steps: int,
    predictor_learning_rate: float,
    witness_learning_rate: float,
    confounding_weight: float = 1.0,
    streams: Mapping[str, RngStream],
) -> tuple[ExplanatoryCoupling, SIPTrainingHistory]:
    """Alternate purification and witness updates through an explanatory coupling."""
    if steps < 1:
        msg = "steps must be positive"
        raise ValueError(msg)
    if predictor_learning_rate < 0.0 or witness_learning_rate < 0.0:
        msg = "learning rates must be nonnegative"
        raise ValueError(msg)

    purification_losses: list[JaxRealArray] = []
    witness_losses: list[JaxRealArray] = []
    for _ in range(steps):

        def predictor_objective(current: ExplanatoryCoupling) -> JaxRealArray:
            output = current.infer(
                innovation,
                goal,
                source_gain,
                parent_instruments,
                target_observation,
                target_gain,
                streams=streams,
                inference=False,
            )
            return purification_loss(output.target, confounding_weight=confounding_weight)

        predictor_value, predictor_gradients = eqx.filter_value_and_grad(predictor_objective)(
            coupling
        )
        predictor_gradients = eqx.tree_at(
            lambda gradients: gradients.score.predictor.witness_map,
            predictor_gradients,
            _zero_tree(predictor_gradients.score.predictor.witness_map),
        )
        coupling = eqx.apply_updates(
            coupling,
            jax.tree.map(
                lambda value: -predictor_learning_rate * value,
                predictor_gradients,
            ),
        )

        def witness_objective(current: ExplanatoryCoupling) -> JaxRealArray:
            output = current.infer(
                innovation,
                goal,
                source_gain,
                parent_instruments,
                target_observation,
                target_gain,
                streams=streams,
                inference=False,
            )
            return witness_loss(output.target)

        witness_value, witness_gradients = eqx.filter_value_and_grad(witness_objective)(coupling)
        witness_gradients = eqx.tree_at(
            lambda gradients: gradients.emitter,
            witness_gradients,
            _zero_tree(witness_gradients.emitter),
        )
        witness_gradients = eqx.tree_at(
            lambda gradients: gradients.score.predictor.prediction_map,
            witness_gradients,
            _zero_tree(witness_gradients.score.predictor.prediction_map),
        )
        coupling = eqx.apply_updates(
            coupling,
            jax.tree.map(
                lambda value: -witness_learning_rate * value,
                witness_gradients,
            ),
        )
        purification_losses.append(predictor_value)
        witness_losses.append(witness_value)

    return coupling, SIPTrainingHistory(
        purification_losses=jnp.stack(purification_losses),
        witness_losses=jnp.stack(witness_losses),
    )


def rollout_td_error(  # ruff: ignore[too-many-arguments]
    td_error: SIPTDError,
    observations: JaxRealArray,
    predictor_observations: JaxRealArray,
    predictor_instruments: JaxRealArray,
    gains: JaxRealArray,
    initial_state: JaxRealArray,
    initial_instrument: JaxRealArray,
    *,
    streams: Mapping[str, RngStream],
    inference: bool,
) -> tuple[TDErrorOutput, ...]:
    """Unroll a TD-error circuit across a leading time axis, carrying its delay.

    ``observations``, ``predictor_observations``, ``predictor_instruments``, and
    ``gains`` each have a leading time axis of the episode's length. ``initial_state``
    and ``initial_instrument`` produce the delayed prediction and witness fed to the
    first step, playing the role of a stored baseline formed before the episode.
    """
    horizon = observations.shape[0]
    unit_gain = jnp.ones_like(gains[0])
    delayed_prediction = td_error.predictor.prediction(
        initial_state,
        unit_gain,
        streams=streams,
        inference=inference,
    )
    delayed_witness = td_error.predictor.witness(
        initial_instrument,
        unit_gain,
        streams=streams,
        inference=inference,
    )
    outputs: list[TDErrorOutput] = []
    for step in range(horizon):
        output = td_error.infer(
            observations[step],
            predictor_observations[step],
            predictor_instruments[step],
            gains[step],
            delayed_prediction,
            delayed_witness,
            streams=streams,
            inference=inference,
        )
        outputs.append(output)
        delayed_prediction = output.delayed_prediction
        delayed_witness = output.delayed_witness
    return tuple(outputs)


def train_td_error_adversarial(  # ruff: ignore[too-many-arguments]
    td_error: SIPTDError,
    observations: JaxRealArray,
    predictor_observations: JaxRealArray,
    predictor_instruments: JaxRealArray,
    gains: JaxRealArray,
    initial_state: JaxRealArray,
    initial_instrument: JaxRealArray,
    *,
    steps: int,
    predictor_learning_rate: float,
    witness_learning_rate: float,
    confounding_weight: float = 1.0,
    streams: Mapping[str, RngStream],
) -> tuple[SIPTDError, SIPTrainingHistory]:
    """Train a TD-error circuit's predictor and witness paths across an episode."""
    if steps < 1:
        msg = "steps must be positive"
        raise ValueError(msg)
    if predictor_learning_rate < 0.0 or witness_learning_rate < 0.0:
        msg = "learning rates must be nonnegative"
        raise ValueError(msg)

    def _rollout(current: SIPTDError) -> tuple[TDErrorOutput, ...]:
        return rollout_td_error(
            current,
            observations,
            predictor_observations,
            predictor_instruments,
            gains,
            initial_state,
            initial_instrument,
            streams=streams,
            inference=True,
        )

    purification_losses: list[JaxRealArray] = []
    witness_losses: list[JaxRealArray] = []
    for _ in range(steps):

        def predictor_objective(current: SIPTDError) -> JaxRealArray:
            outputs = _rollout(current)
            losses = jnp.stack(
                [
                    purification_loss(output, confounding_weight=confounding_weight)
                    for output in outputs
                ]
            )
            return jnp.mean(losses)

        predictor_value, predictor_gradients = eqx.filter_value_and_grad(predictor_objective)(
            td_error
        )
        predictor_gradients = eqx.tree_at(
            lambda gradients: gradients.predictor.witness_map,
            predictor_gradients,
            _zero_tree(predictor_gradients.predictor.witness_map),
        )
        td_error = eqx.apply_updates(
            td_error,
            jax.tree.map(
                lambda value: -predictor_learning_rate * value,
                predictor_gradients,
            ),
        )

        def witness_objective(current: SIPTDError) -> JaxRealArray:
            outputs = _rollout(current)
            losses = jnp.stack([witness_loss(output) for output in outputs])
            return jnp.mean(losses)

        witness_value, witness_gradients = eqx.filter_value_and_grad(witness_objective)(td_error)
        witness_gradients = eqx.tree_at(
            lambda gradients: gradients.predictor.prediction_map,
            witness_gradients,
            _zero_tree(witness_gradients.predictor.prediction_map),
        )
        td_error = eqx.apply_updates(
            td_error,
            jax.tree.map(
                lambda value: -witness_learning_rate * value,
                witness_gradients,
            ),
        )
        purification_losses.append(predictor_value)
        witness_losses.append(witness_value)

    return td_error, SIPTrainingHistory(
        purification_losses=jnp.stack(purification_losses),
        witness_losses=jnp.stack(witness_losses),
    )
