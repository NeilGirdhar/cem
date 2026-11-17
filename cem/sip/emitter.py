"""SIP emitter circuit: observation map, instrument map, noise top-up, gain."""

from typing import Self

import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
from efax import NormalNP, NormalPP
from jax.lax import stop_gradient
from tjax import JaxRealArray, KeyArray, PyTree

from cem.framework.graph import EligibilityTraining
from cem.npn.gaussian import GaussianNeuralProjection
from cem.sip.credit import Credit, CreditEmission, CreditOutputs
from cem.sip.instrument import inherit_instrument, instrument_loss
from cem.sip.losses import SIPLossOutputs
from cem.sip.messages import InstrumentedMessage


class ObservationCoordinates(eqx.Module):
    """Represent raw Gaussian means and precisions, with one entry per channel.

    The observation pullback accepts cotangents in the same coordinates. To assign
    credit to one coordinate, supply zeros for the other.
    """

    mean: JaxRealArray
    precision: JaxRealArray


type _ObservationProjection = EligibilityTraining[
    GaussianNeuralProjection, NormalNP, ObservationCoordinates, NormalNP
]


class _ObservationTraces(eqx.Module):
    """The lag filter's state for the mean, precision, and shrinkage eligibility traces.

    Each follows the observation map's PyTree structure, with the filter's state at each trainable
    leaf. Their drifts are the cotangents that the mean and precision contributions and the
    shrinkage cotangent pull back to.
    """

    mean: PyTree
    precision: PyTree
    shrinkage: PyTree


class EmitterEmission(SIPLossOutputs):
    """The outgoing message, the signals used to score it, and the drifts awaiting a goal.

    The loss holds the instrument loss, the one term that does not wait for the goal.
    """

    message: InstrumentedMessage
    raw_observation: NormalNP
    noised_instrument: NormalNP
    local_variance: JaxRealArray
    mean_drift: GaussianNeuralProjection
    precision_drift: GaussianNeuralProjection
    shrinkage_drift: GaussianNeuralProjection
    credit_emission: CreditEmission


class EmitterOutputs(SIPLossOutputs):
    """The outgoing message and the signals used to score and train its observation."""

    message: InstrumentedMessage
    raw_observation: NormalNP
    noised_instrument: NormalNP
    local_variance: JaxRealArray
    credit_emission: CreditEmission
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

    def emit(
        self,
        state: eqx.nn.State,
        key: KeyArray,
        *,
        observation_content: NormalNP,
        predictor_instrument: NormalNP,
        gain_observation: NormalNP | None,
    ) -> tuple[eqx.nn.State, EmitterEmission]:
        """Emit an instrumented observation and record the drifts that a later goal will credit.

        For a controllable emitter, add the same random displacement to the observation and
        its instrument. The fresh noise fills the positive gap between inherited instrument
        variance and squared inherited instrument mean.

        Apply gain to the precision of both outgoing messages. With no gain observation, emit
        both at their original precision and use unit gain for credit.

        Every drift is pulled back through the map as it ran, so the goal that arrives later
        credits this emission's computation.
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
            # See `evidence-scaling-link` in the thesis.
            observation = noised_raw_observation.scale_evidence(gain)
            instrument = noised_instrument.scale_evidence(gain)

        credit_emission = self.credit.emit(
            gain_observation=gain_observation,
            noised_instrument=noised_instrument,
            observation_content=observation_content,
            raw_observation=raw_observation,
        )
        mean_contribution = credit_emission.mean_contribution
        precision_contribution = credit_emission.precision_contribution
        mean_drift = self.observation_projection.drift(
            pullback,
            ObservationCoordinates(
                mean=mean_contribution, precision=jnp.zeros_like(precision_contribution)
            ),
        )
        precision_drift = self.observation_projection.drift(
            pullback,
            ObservationCoordinates(
                mean=jnp.zeros_like(mean_contribution), precision=precision_contribution
            ),
        )
        shrinkage_drift = self.observation_projection.drift(
            pullback,
            ObservationCoordinates(
                mean=credit_emission.shrinkage_cotangent,
                precision=jnp.zeros_like(precision_contribution),
            ),
        )
        return state, EmitterEmission(
            message=InstrumentedMessage(observation, instrument),
            raw_observation=raw_observation,
            noised_instrument=noised_instrument,
            local_variance=local_variance,
            mean_drift=mean_drift,
            precision_drift=precision_drift,
            shrinkage_drift=shrinkage_drift,
            credit_emission=credit_emission,
            loss=instrument_loss(inherited_instrument, raw_observation, gain),
        )

    def credit_goal(
        self, state: eqx.nn.State, emission: EmitterEmission, goal: JaxRealArray
    ) -> tuple[eqx.nn.State, CreditOutputs, JaxRealArray]:
        """Credit the arriving goal to the retained emissions, then retain this emission.

        The goal scales the mean trace, the goal prediction error scales the precision trace, and
        the shrinkage trace applies at unit scale. The traces are read before this emission's
        drifts enter them, so every training signal for an emission arrives with the goal that
        follows it, and none arrives for an emission whose goal never does. Return the new state,
        the credit outputs, and the loss, which includes the goal prediction loss.
        """
        lag_filter = self.credit.lag_filter.value
        traces = state.get(self.trace_index)
        new_state, credit_outputs = self.credit.credit(state, emission.credit_emission, goal)
        loss = (
            self.observation_projection.get_loss(lag_filter.value(traces.mean), goal)
            + self.observation_projection.get_loss(
                lag_filter.value(traces.precision), credit_outputs.goal_prediction_error
            )
            + self.observation_projection.get_loss(
                lag_filter.value(traces.shrinkage), jnp.asarray(1.0)
            )
            + credit_outputs.goal_prediction_loss
        )
        new_state = new_state.set(
            self.trace_index,
            _ObservationTraces(
                mean=lag_filter.update(traces.mean, emission.mean_drift),
                precision=lag_filter.update(traces.precision, emission.precision_drift),
                shrinkage=lag_filter.update(traces.shrinkage, emission.shrinkage_drift),
            ),
        )
        return new_state, credit_outputs, loss

    def step(
        self,
        state: eqx.nn.State,
        key: KeyArray,
        *,
        observation_content: NormalNP,
        predictor_instrument: NormalNP,
        gain_observation: NormalNP | None,
        goal: JaxRealArray,
    ) -> tuple[eqx.nn.State, EmitterOutputs]:
        """Emit, then credit a goal that is already available at this step.

        See `emit` and `credit_goal`.
        """
        state, emission = self.emit(
            state,
            key,
            observation_content=observation_content,
            predictor_instrument=predictor_instrument,
            gain_observation=gain_observation,
        )
        state, credit_outputs, credit_loss = self.credit_goal(state, emission, goal)
        return state, EmitterOutputs(
            message=emission.message,
            raw_observation=emission.raw_observation,
            noised_instrument=emission.noised_instrument,
            local_variance=emission.local_variance,
            credit_emission=emission.credit_emission,
            credit_outputs=credit_outputs,
            loss=emission.loss + credit_loss,
        )

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
        empty_trace = credit.lag_filter.value.init(observation_projection.zero_trace())
        return cls(
            observation_projection=observation_projection,
            trace_index=eqx.nn.StateIndex(
                _ObservationTraces(mean=empty_trace, precision=empty_trace, shrinkage=empty_trace)
            ),
            instrument_map=instrument_map,
            credit=credit,
            controllable=controllable,
        )


def _project_observation(
    model: GaussianNeuralProjection, content: NormalNP
) -> tuple[ObservationCoordinates, NormalNP]:
    """Project the content while confining observation credit to the map parameters."""
    observation = model.infer(stop_gradient(content))
    moments = observation.to_precision_parametrization()
    return ObservationCoordinates(mean=moments.mean, precision=moments.precision), observation


def _shift_mean(message: NormalNP, displacement: JaxRealArray) -> NormalNP:
    """Shift a Gaussian message's mean by `displacement`, preserving precision."""
    moments = message.to_precision_parametrization()
    return NormalPP(moments.mean + displacement, moments.precision).to_nat()
