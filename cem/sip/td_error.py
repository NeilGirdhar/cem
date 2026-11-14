"""SIP TD-error circuit: bootstrap TD reward prediction with integral and proportional witnesses."""

from collections.abc import Mapping
from typing import Self

import equinox as eqx
import jax
import jax.numpy as jnp
from efax import NormalNP, NormalVP
from tjax import JaxRealArray, RngStream

from cem.framework.graph import FixedParameter
from cem.npn.gaussian import GaussianNeuralProjection
from cem.sip.losses import SIPLossOutputs
from cem.sip.messages import InstrumentedMessage, variance_from_precision
from cem.sip.parameters import WitnessMultiplierParameter
from cem.sip.settings import TDSettings
from cem.sip.witness import (
    IntegralWitnessMap,
    ProportionalWitnessMap,
    witness_losses,
    witness_residuals,
)


class TDTraces(eqx.Module):
    """One-step delay buffer for the TD-error circuit's predictor inputs."""

    predictor: InstrumentedMessage

    @classmethod
    def initial(cls, predictor_features: int) -> Self:
        return cls(predictor=InstrumentedMessage.standard(predictor_features))


class TDErrorOutputs(SIPLossOutputs):
    """The advantage signal and the losses for value prediction and its witnesses."""

    advantage: JaxRealArray  # shape (n,); outgoing TD-error mean signal
    reconstruction_cost: JaxRealArray  # scalar
    confounding_cost: JaxRealArray  # scalar
    uncertainty_loss: JaxRealArray  # scalar
    integral_witness_loss: JaxRealArray  # scalar
    proportional_witness_loss: JaxRealArray  # scalar
    witness_norm_loss: JaxRealArray  # scalar


class _Moments(eqx.Module):
    """The bootstrap target and the delayed prediction it scores."""

    bootstrap: NormalVP
    delayed: NormalVP


class TDError(eqx.Module):
    """Learn to predict return and assign credit through temporal-difference error.

    See `sip-td-error-circuit` in the thesis.
    """

    prediction_map: GaussianNeuralProjection
    integral_witness_map: IntegralWitnessMap
    proportional_witness_map: ProportionalWitnessMap
    witness_multiplier: WitnessMultiplierParameter  # scalar
    discount: FixedParameter[JaxRealArray]  # scalar
    damping_weight: FixedParameter[JaxRealArray]  # scalar
    delay_index: eqx.nn.StateIndex[TDTraces]

    @classmethod
    def create(
        cls,
        n: int,
        *,
        predictor_features: int,
        td_settings: TDSettings,
        streams: Mapping[str, RngStream],
    ) -> Self:
        return cls(
            prediction_map=GaussianNeuralProjection.create(
                in_features=predictor_features,
                out_features=n,
                hidden_features=td_settings.prediction_hidden_features,
                streams=streams,
            ),
            integral_witness_map=IntegralWitnessMap.create(
                n,
                instrument_features=predictor_features,
                hidden_features=td_settings.integral_witness_hidden_features,
                streams=streams,
            ),
            proportional_witness_map=ProportionalWitnessMap.create(
                n,
                instrument_features=predictor_features,
                hidden_features=td_settings.proportional_witness_hidden_features,
                streams=streams,
            ),
            witness_multiplier=WitnessMultiplierParameter(jnp.zeros(())),
            discount=FixedParameter(jnp.asarray(td_settings.discount, dtype=jnp.float64)),
            damping_weight=FixedParameter(
                jnp.asarray(td_settings.damping_weight, dtype=jnp.float64)
            ),
            delay_index=eqx.nn.StateIndex(TDTraces.initial(predictor_features)),
        )

    def step(
        self,
        state: eqx.nn.State,
        *,
        reward_observation: NormalNP,
        predictor: InstrumentedMessage,
        streams: Mapping[str, RngStream],
        inference: bool,
    ) -> tuple[eqx.nn.State, TDErrorOutputs]:
        """Score the previous step's value prediction against the current bootstrap target.

        Evaluate the prediction and both witnesses on the stored predictor so the witnesses test
        the instruments associated with the prediction being scored. Return the difference between
        the target mean and the delayed prediction mean as advantage, and save the current
        predictor for the next step. Reward features have unit gain.
        """
        delayed = state.get(self.delay_index)
        delayed_prediction = self._predict(delayed.predictor.observation)
        delayed_instruments = delayed.predictor.instrument
        current_prediction = self._predict(predictor.observation)

        moments = self._moments(reward_observation, current_prediction, delayed_prediction)
        residuals = witness_residuals(moments.bootstrap, moments.delayed)

        reconstruction_cost = _reconstruction_cost(moments)
        uncertainty_loss = jnp.sum(moments.delayed.variance)
        losses = witness_losses(
            residuals,
            integral_witness=self.integral_witness_map.infer(
                delayed_instruments, streams=streams, inference=inference
            ),
            proportional_witness=self.proportional_witness_map.infer(
                delayed_instruments, streams=streams, inference=inference
            ),
            gain=None,
            multiplier=self.witness_multiplier.value,
            damping_weight=self.damping_weight.value,
        )

        advantage = moments.bootstrap.mean - moments.delayed.mean

        new_state = state.set(self.delay_index, TDTraces(predictor=predictor))

        return new_state, TDErrorOutputs(
            advantage=advantage,
            reconstruction_cost=reconstruction_cost,
            confounding_cost=losses.confounding_cost,
            uncertainty_loss=uncertainty_loss,
            integral_witness_loss=losses.integral_witness_loss,
            proportional_witness_loss=losses.proportional_witness_loss,
            witness_norm_loss=losses.witness_norm_loss,
            loss=reconstruction_cost + uncertainty_loss + losses.loss,
        )

    def _predict(self, predictor_observations: NormalNP) -> NormalNP:
        """Predict return with the predictor observations held fixed.

        The features that supplied the observations learn from their own goals rather than from
        the TD costs. See `sip-td-error-circuit` in the thesis.
        """
        return self.prediction_map.infer(jax.lax.stop_gradient(predictor_observations))

    def _moments(
        self,
        reward_observation: NormalNP,
        current_prediction: NormalNP,
        delayed_prediction: NormalNP,
    ) -> _Moments:
        """Form a fixed Gaussian target from observed reward and discounted current value.

        Add the variances as for independent terms, scaling value variance by the squared
        discount. Stop gradients through both target terms so the fit updates the delayed
        prediction.
        """
        reward = reward_observation.to_precision_parametrization()
        current = current_prediction.to_variance_parametrization()
        gamma = self.discount.value
        bootstrap = NormalVP(
            reward.mean + gamma * current.mean,
            variance_from_precision(reward.precision) + jnp.square(gamma) * current.variance,
        )
        return _Moments(
            bootstrap=jax.lax.stop_gradient(bootstrap),
            delayed=delayed_prediction.to_variance_parametrization(),
        )


def _reconstruction_cost(moments: _Moments) -> JaxRealArray:
    return jnp.sum(moments.bootstrap.cross_entropy(moments.delayed))
