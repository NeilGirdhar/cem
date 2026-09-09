"""Alternating optimization for real-valued SIP score circuits."""

from collections.abc import Mapping

import equinox as eqx
import jax
import jax.numpy as jnp
from tjax import JaxRealArray, RngStream

from cem.sip.model import SIPChain
from cem.sip.objectives import purification_loss, witness_loss
from cem.sip.score import SIPScore


class SIPTrainingHistory(eqx.Module):
    """Objective values recorded during alternating SIP optimization."""

    purification_losses: JaxRealArray
    witness_losses: JaxRealArray


def _zero_tree(value: object) -> object:
    return jax.tree.map(
        lambda leaf: jnp.zeros_like(leaf) if eqx.is_array(leaf) else leaf,
        value,
    )


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
            lambda gradients: gradients.witness_map,
            predictor_gradients,
            _zero_tree(predictor_gradients.witness_map),
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
            lambda gradients: gradients.prediction_map,
            witness_gradients,
            _zero_tree(witness_gradients.prediction_map),
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


def train_chain_adversarial(  # ruff: ignore[too-many-arguments]
    chain: SIPChain,
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
) -> tuple[SIPChain, SIPTrainingHistory]:
    """Alternate purification and witness updates through an entire SIP chain."""
    if steps < 1:
        msg = "steps must be positive"
        raise ValueError(msg)
    if predictor_learning_rate < 0.0 or witness_learning_rate < 0.0:
        msg = "learning rates must be nonnegative"
        raise ValueError(msg)

    purification_losses: list[JaxRealArray] = []
    witness_losses: list[JaxRealArray] = []
    for _ in range(steps):

        def predictor_objective(current: SIPChain) -> JaxRealArray:
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

        predictor_value, predictor_gradients = eqx.filter_value_and_grad(predictor_objective)(chain)
        predictor_gradients = eqx.tree_at(
            lambda gradients: gradients.score.witness_map,
            predictor_gradients,
            _zero_tree(predictor_gradients.score.witness_map),
        )
        chain = eqx.apply_updates(
            chain,
            jax.tree.map(
                lambda value: -predictor_learning_rate * value,
                predictor_gradients,
            ),
        )

        def witness_objective(current: SIPChain) -> JaxRealArray:
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

        witness_value, witness_gradients = eqx.filter_value_and_grad(witness_objective)(chain)
        witness_gradients = eqx.tree_at(
            lambda gradients: gradients.emitter,
            witness_gradients,
            _zero_tree(witness_gradients.emitter),
        )
        witness_gradients = eqx.tree_at(
            lambda gradients: gradients.score.prediction_map,
            witness_gradients,
            _zero_tree(witness_gradients.score.prediction_map),
        )
        chain = eqx.apply_updates(
            chain,
            jax.tree.map(
                lambda value: -witness_learning_rate * value,
                witness_gradients,
            ),
        )
        purification_losses.append(predictor_value)
        witness_losses.append(witness_value)

    return chain, SIPTrainingHistory(
        purification_losses=jnp.stack(purification_losses),
        witness_losses=jnp.stack(witness_losses),
    )
