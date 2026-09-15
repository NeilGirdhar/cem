"""Real-valued TD-error circuit for self-instrumental purification."""

from collections.abc import Mapping
from typing import Self

import equinox as eqx
import jax.numpy as jnp
from jax.lax import stop_gradient
from tjax import JaxRealArray, RngStream

from cem.sip.predictor_witness import PredictorWitnessPair, feature_vector


class TDErrorOutput(eqx.Module):
    """A temporal-difference error and the delayed state carried to the next step.

    ``observation_score`` is the TD error emitted to advantage links. It shares its
    name and the ``reconstruction_loss``/``witness``/``confounding_error`` fields
    with ``ScoreOutput`` so both circuits train under the same ``purification_loss``
    and ``witness_loss`` objectives. ``delayed_prediction`` and ``delayed_witness``
    are this step's current prediction and witness, undelayed here so gradients
    reach the maps that produced them when the next step's score trains them; pass
    them to the next call as its ``delayed_prediction``/``delayed_witness``.
    """

    observation_score: JaxRealArray
    reconstruction_loss: JaxRealArray
    witness: JaxRealArray
    confounding_error: JaxRealArray
    delayed_prediction: JaxRealArray
    delayed_witness: JaxRealArray


class SIPTDError(eqx.Module):
    """Temporally baseline a real-valued observation with a bootstrapped TD error.

    Where SIPScore subtracts the current, gain-scaled prediction from the
    observation, this circuit subtracts the *delayed* prediction and witness
    carried in from the previous step, and bootstraps its target with the current
    step's stopped prediction (@temporal-baseline). The delayed baseline was fixed
    before this step's cause acted, so unlike an ordinary score, it cannot explain
    away that cause's effect on this step's observation; the discarded stopped copy
    lets the current prediction serve only as next step's bootstrap value, not as a
    target chasing itself. Reconstruction and confounding losses train the delayed
    prediction to match the bootstrap target, pairing the TD error with the delayed
    witness that accompanied it (@sip-td-error-circuit).

    The prediction and witness maps mirror SIPScore's; see PredictorWitnessPair.
    """

    predictor: PredictorWitnessPair
    discount: float = eqx.field(static=True)
    observation_features: int = eqx.field(static=True)
    predictor_observation_features: int = eqx.field(static=True)
    predictor_instrument_features: int = eqx.field(static=True)

    @classmethod
    def create(
        cls,
        predictor_observation_features: int,
        predictor_instrument_features: int,
        observation_features: int,
        *,
        discount: float = 1.0,
        hidden_features: int | tuple[int, ...] = (),
        streams: Mapping[str, RngStream],
    ) -> Self:
        if not 0.0 < discount <= 1.0:
            msg = "discount must lie in (0, 1]"
            raise ValueError(msg)
        predictor = PredictorWitnessPair.create(
            predictor_observation_features,
            predictor_instrument_features,
            observation_features,
            hidden_features=hidden_features,
            streams=streams,
        )
        return cls(
            predictor=predictor,
            discount=discount,
            observation_features=observation_features,
            predictor_observation_features=predictor_observation_features,
            predictor_instrument_features=predictor_instrument_features,
        )

    def infer(
        self,
        observation: JaxRealArray,
        predictor_observations: JaxRealArray,
        predictor_instruments: JaxRealArray,
        gain: JaxRealArray,
        delayed_prediction: JaxRealArray,
        delayed_witness: JaxRealArray,
        *,
        streams: Mapping[str, RngStream],
        inference: bool,
    ) -> TDErrorOutput:
        """Score one step and produce the delayed state to carry into the next one.

        A terminal step passes zero gain, forcing its current prediction to zero so
        the bootstrap target carries no value beyond the episode.
        """
        observation = feature_vector(observation, self.observation_features)
        current_prediction = self.predictor.prediction(
            predictor_observations,
            gain,
            streams=streams,
            inference=inference,
        )
        current_witness = self.predictor.witness(
            predictor_instruments,
            gain,
            streams=streams,
            inference=inference,
        )
        bootstrap_target = observation + self.discount * stop_gradient(current_prediction)
        observation_score = delayed_prediction - bootstrap_target
        reconstruction_loss = 0.5 * jnp.sum(jnp.square(observation_score), axis=-1)
        confounding_error = jnp.sum(observation_score * delayed_witness, axis=-1)
        return TDErrorOutput(
            observation_score=observation_score,
            reconstruction_loss=reconstruction_loss,
            witness=delayed_witness,
            confounding_error=confounding_error,
            delayed_prediction=current_prediction,
            delayed_witness=current_witness,
        )
