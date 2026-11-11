"""SIP emitter circuit: observation map, instrument map, noise top-up, gain."""

from collections.abc import Mapping
from typing import Self

import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
from efax import NormalNP, NormalPP
from jax.lax import stop_gradient
from tjax import JaxRealArray, KeyArray

from cem.npn.gaussian import GaussianNeuralProjection
from cem.sip.credit import Credit, CreditOutputs
from cem.sip.instrument import inherit_instrument, instrument_loss
from cem.sip.losses import SIPLossOutputs
from cem.sip.messages import InstrumentedMessage, evidence_scale
from cem.structure.graph import EligibilityTraining


class _ObservationCoordinates(eqx.Module):
    """Represent raw Gaussian means and precisions, with one entry per channel.

    The observation pullback accepts cotangents in the same coordinates. To assign
    credit to one coordinate, supply zeros for the other.
    """

    mean: JaxRealArray
    precision: JaxRealArray


type _ObservationProjection = EligibilityTraining[
    GaussianNeuralProjection, NormalNP, _ObservationCoordinates, NormalNP
]


class _ObservationTraces(eqx.Module):
    """Retain parameter cotangents from the mean and precision pullbacks.

    Both traces follow the observation map's PyTree structure. At each trainable leaf,
    they store the accumulated cotangent from credit applied to the corresponding
    output coordinate.
    """

    mean: GaussianNeuralProjection
    precision: GaussianNeuralProjection


class EmitterOutputs(SIPLossOutputs):
    """The outgoing message and the signals used to score and train its observation."""

    message: InstrumentedMessage
    raw_observation: NormalNP
    noised_instrument: NormalNP
    local_variance: JaxRealArray
    credit_outputs: CreditOutputs


class Emitter(eqx.Module):
    """Emit a feature's observation with an instrument that tracks its experimental variation.

    See `sip-emitter` in the thesis.
    """

    observation_projection: _ObservationProjection
    trace_index: eqx.nn.StateIndex[_ObservationTraces]
    instrument_map: GaussianNeuralProjection
    credit: Credit
    controllable: bool = eqx.field(static=True)

    @property
    def observation_map(self) -> GaussianNeuralProjection:
        return self.observation_projection.model

    def step(
        self,
        state: eqx.nn.State,
        key: KeyArray,
        *,
        observation_content: NormalNP,
        predictor_instrument: NormalNP,
        gain_observation: NormalNP | None,
        goals: Mapping[str, JaxRealArray],
    ) -> tuple[eqx.nn.State, EmitterOutputs]:
        """Emit an instrumented observation and compute credit for the raw observation.

        For a controllable emitter, add the same random displacement to the observation and
        its instrument. The fresh noise fills the positive gap between inherited instrument
        variance and squared inherited instrument mean.

        Apply gain to the precision of both outgoing messages. With no gain observation, emit
        both at their original precision and use unit gain for credit.

        Each named goal scales the mean trace. The fixed-weight loss uses the full goal prediction
        error for precision credit and baseline learning, applies shrinkage credit through the
        current pullback, and includes the instrument loss.
        """
        raw_observation, pullback = self.observation_projection.infer(observation_content)
        inherited_instrument = inherit_instrument(self.instrument_map, predictor_instrument)
        inherited = inherited_instrument.to_variance_parametrization()

        if self.controllable:
            local_variance = jnp.maximum(0.0, inherited.variance - jnp.square(inherited.mean))
            displacement = jnp.sqrt(local_variance) * jr.normal(key, inherited.mean.shape)
            noised_raw_observation = _shift_mean(raw_observation, displacement)
            noised_instrument = _shift_mean(inherited_instrument, displacement)
        else:
            local_variance = jnp.zeros_like(inherited.mean)
            noised_raw_observation = raw_observation
            noised_instrument = inherited_instrument

        gain = None if gain_observation is None else gain_observation.mean()
        if gain is None:
            observation = noised_raw_observation
            instrument = noised_instrument
        else:
            observation = evidence_scale(noised_raw_observation, gain)
            instrument = evidence_scale(noised_instrument, gain)

        new_state, credit_outputs = self.credit.step(
            state,
            gain_observation=gain_observation,
            noised_instrument=noised_instrument,
            observation_content=observation_content,
            raw_observation=raw_observation,
            goal=sum(goals.values(), jnp.asarray(0.0)),
        )

        traces = new_state.get(self.trace_index)
        mean_contribution = credit_outputs.mean_contribution
        precision_contribution = credit_outputs.precision_contribution
        decay = self.credit.trace_decay()
        mean_trace = self.observation_projection.accumulate(
            traces.mean,
            pullback,
            _ObservationCoordinates(
                mean=mean_contribution, precision=jnp.zeros_like(precision_contribution)
            ),
            decay=decay,
        )
        precision_trace = self.observation_projection.accumulate(
            traces.precision,
            pullback,
            _ObservationCoordinates(
                mean=jnp.zeros_like(mean_contribution), precision=precision_contribution
            ),
            decay=decay,
        )
        new_state = new_state.set(
            self.trace_index, _ObservationTraces(mean=mean_trace, precision=precision_trace)
        )
        (shrinkage_credit,) = pullback(
            _ObservationCoordinates(
                mean=credit_outputs.shrinkage_cotangent,
                precision=jnp.zeros_like(precision_contribution),
            )
        )
        balanced_losses = {
            name: self.observation_projection.get_loss(mean_trace, goal)
            for name, goal in goals.items()
        }
        fixed_weight_loss = self.observation_projection.get_loss(
            precision_trace, credit_outputs.goal_prediction_error
        ) + self.observation_projection.get_loss(shrinkage_credit, jnp.asarray(1.0))
        fixed_weight_loss += credit_outputs.goal_prediction_loss
        fixed_weight_loss += instrument_loss(inherited_instrument, raw_observation, gain)
        outputs = EmitterOutputs(
            message=InstrumentedMessage(observation, instrument),
            raw_observation=raw_observation,
            noised_instrument=noised_instrument,
            local_variance=local_variance,
            credit_outputs=credit_outputs,
            balanced_losses=balanced_losses,
            fixed_weight_loss=fixed_weight_loss,
        )
        return new_state, outputs

    @classmethod
    def create(
        cls,
        observation_map: GaussianNeuralProjection,
        instrument_map: GaussianNeuralProjection,
        credit: Credit,
        *,
        controllable: bool,
    ) -> Self:
        observation_projection: _ObservationProjection = EligibilityTraining(
            observation_map, _project_observation
        )
        zero_trace = observation_projection.zero_trace()
        return cls(
            observation_projection=observation_projection,
            trace_index=eqx.nn.StateIndex(
                _ObservationTraces(mean=zero_trace, precision=zero_trace)
            ),
            instrument_map=instrument_map,
            credit=credit,
            controllable=controllable,
        )


def _project_observation(
    model: GaussianNeuralProjection, content: NormalNP
) -> tuple[_ObservationCoordinates, NormalNP]:
    """Project the content while confining observation credit to the map parameters."""
    observation = model.infer(stop_gradient(content))
    moments = observation.to_precision_parametrization()
    return _ObservationCoordinates(mean=moments.mean, precision=moments.precision), observation


def _shift_mean(message: NormalNP, displacement: JaxRealArray) -> NormalNP:
    """Shift a Gaussian message's mean by `displacement`, preserving precision."""
    moments = message.to_precision_parametrization()
    return NormalPP(moments.mean + displacement, moments.precision).to_nat()
