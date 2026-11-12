"""SIP emitter circuit: observation map, instrument map, noise top-up, gain."""

from collections.abc import Mapping
from typing import Self

import equinox as eqx
import jax
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


class ObservationModel(eqx.Module):
    """Map observation content to the raw observation and each goal component's curvature.

    The observation map produces the raw observation. The share map divides its precision among
    the goal components, so each component can fit its own curvature while the curvatures sum to
    the precision. See `sip-curvature-shares` in the thesis.
    """

    observation_map: GaussianNeuralProjection
    share_map: GaussianNeuralProjection


class _ObservationCoordinates(eqx.Module):
    """Represent the raw mean, with one entry per channel, and each goal component's curvature.

    The observation pullback accepts cotangents in the same coordinates. To assign credit to one
    coordinate, supply zeros for the others.
    """

    mean: JaxRealArray  # (n,)
    curvature: JaxRealArray  # (m, n)


class _ProjectedObservation(eqx.Module):
    """The raw observation and the curvature of each goal component."""

    raw_observation: NormalNP
    curvature: JaxRealArray  # (m, n)


type _ObservationProjection = EligibilityTraining[
    ObservationModel, NormalNP, _ObservationCoordinates, _ProjectedObservation
]


class _ObservationTraces(eqx.Module):
    """Retain parameter cotangents from the mean pullback and each curvature pullback.

    Every trace follows the observation model's PyTree structure. At each trainable leaf, it
    stores the accumulated cotangent from credit applied to the corresponding output coordinate.
    The precision traces hold one trace for each goal component.
    """

    mean: ObservationModel
    precision: tuple[ObservationModel, ...]


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
    goal_names: tuple[str, ...] = eqx.field(static=True)
    controllable: bool = eqx.field(static=True)

    @property
    def observation_map(self) -> GaussianNeuralProjection:
        return self.observation_projection.model.observation_map

    @property
    def share_map(self) -> GaussianNeuralProjection:
        return self.observation_projection.model.share_map

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

        `goals` supplies a value for each of the emitter's goal components. Each component's
        balanced loss combines its mean, precision, and shrinkage credit, so a gradient transform
        can balance the components without separating a component's credit from its shrinkage.
        The fixed-weight loss trains the baseline and instrument maps.
        """
        if set(goals) != set(self.goal_names):
            msg = f"Expected goals {sorted(self.goal_names)}, got {sorted(goals)}"
            raise ValueError(msg)
        projected, pullback = self.observation_projection.infer(observation_content)
        raw_observation = projected.raw_observation
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
            curvature=projected.curvature,
            goals=jnp.stack([goals[name] for name in self.goal_names]),
        )

        traces = new_state.get(self.trace_index)
        decay = self.credit.trace_decay()
        no_mean = jnp.zeros_like(credit_outputs.mean_contribution)
        no_curvature = jnp.zeros_like(projected.curvature)
        mean_trace = self.observation_projection.accumulate(
            traces.mean,
            pullback,
            _ObservationCoordinates(mean=credit_outputs.mean_contribution, curvature=no_curvature),
            decay=decay,
        )
        precision_traces = tuple(
            self.observation_projection.accumulate(
                trace,
                pullback,
                _ObservationCoordinates(
                    mean=no_mean,
                    curvature=no_curvature.at[j].set(credit_outputs.curvature_contribution),
                ),
                decay=decay,
            )
            for j, trace in enumerate(traces.precision)
        )
        new_state = new_state.set(
            self.trace_index, _ObservationTraces(mean=mean_trace, precision=precision_traces)
        )
        balanced_losses: dict[str, JaxRealArray] = {}
        for j, name in enumerate(self.goal_names):
            (shrinkage_credit,) = pullback(
                _ObservationCoordinates(
                    mean=credit_outputs.shrinkage_cotangent[j], curvature=no_curvature
                )
            )
            balanced_losses[name] = (
                self.observation_projection.get_loss(mean_trace, goals[name])
                + self.observation_projection.get_loss(
                    precision_traces[j], credit_outputs.goal_prediction_error[j]
                )
                + self.observation_projection.get_loss(shrinkage_credit, jnp.asarray(1.0))
            )
        fixed_weight_loss = credit_outputs.goal_prediction_loss + instrument_loss(
            inherited_instrument, raw_observation, gain
        )
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
        observation_model: ObservationModel,
        instrument_map: GaussianNeuralProjection,
        credit: Credit,
        *,
        goal_names: tuple[str, ...],
        controllable: bool,
    ) -> Self:
        """Build an emitter whose goal has the named components.

        The share map must emit one component for each pair of goal component and channel.
        """
        observation_projection: _ObservationProjection = EligibilityTraining(
            observation_model, _project_observation
        )
        zero_trace = observation_projection.zero_trace()
        return cls(
            observation_projection=observation_projection,
            trace_index=eqx.nn.StateIndex(
                _ObservationTraces(mean=zero_trace, precision=tuple(zero_trace for _ in goal_names))
            ),
            instrument_map=instrument_map,
            credit=credit,
            goal_names=goal_names,
            controllable=controllable,
        )


def _project_observation(
    model: ObservationModel, content: NormalNP
) -> tuple[_ObservationCoordinates, _ProjectedObservation]:
    """Project the content while confining observation credit to the map parameters.

    A softmax over goal components, channel by channel, turns the share map's means into the
    shares of the raw precision.
    """
    fixed_content = stop_gradient(content)
    observation = model.observation_map.infer(fixed_content)
    moments = observation.to_precision_parametrization()
    n = moments.mean.shape[0]
    share_logits = model.share_map.infer(fixed_content).mean().reshape(-1, n)
    curvature = jax.nn.softmax(share_logits, axis=0) * moments.precision[jnp.newaxis, :]
    return (
        _ObservationCoordinates(mean=moments.mean, curvature=curvature),
        _ProjectedObservation(raw_observation=observation, curvature=curvature),
    )


def _shift_mean(message: NormalNP, displacement: JaxRealArray) -> NormalNP:
    """Shift a Gaussian message's mean by `displacement`, preserving precision."""
    moments = message.to_precision_parametrization()
    return NormalPP(moments.mean + displacement, moments.precision).to_nat()
