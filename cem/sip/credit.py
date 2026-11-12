"""Training signals for an emitter’s observation projection."""

from typing import Self

import equinox as eqx
import jax.numpy as jnp
from efax import NormalNP
from jax.lax import stop_gradient
from tjax import JaxRealArray

from cem.framework.graph import FixedParameter
from cem.npn.gaussian import GaussianNeuralProjection
from cem.sip.settings import CreditSettings


class CreditTraces(eqx.Module):
    """Retain the predicted penalty and observation precision of past experiments."""

    goal_penalty: JaxRealArray
    obs_precision: JaxRealArray

    @classmethod
    def zeros(cls, n: int) -> Self:
        return cls(jnp.zeros((), dtype=jnp.float64), jnp.zeros(n, dtype=jnp.float64))


class CreditOutputs(eqx.Module):
    """Trace contributions, shrinkage credit, and the loss that trains the goal baseline.

    `shrinkage_cotangent` is shrinkage credit in raw-mean coordinates: the descent cotangent of the
    mean shrinkage loss, with the observation precision trace held fixed. The observation
    projection pulls it back through the current map. See `sip-shrinkage-credit` in the thesis.
    """

    mean_contribution: JaxRealArray
    precision_contribution: JaxRealArray
    goal_prediction_error: JaxRealArray
    goal_prediction_loss: JaxRealArray  # scalar
    shrinkage_cotangent: JaxRealArray


class Credit(eqx.Module):
    """Assign credit to an emitter's experiments when later goals arrive.

    See `sip-credit-circuit` in the thesis.
    """

    trace_decay_rate: FixedParameter[JaxRealArray]
    goal_baseline_map: GaussianNeuralProjection
    trace_index: eqx.nn.StateIndex[CreditTraces]

    @classmethod
    def create(
        cls,
        n: int,
        goal_baseline_map: GaussianNeuralProjection,
        *,
        credit_settings: CreditSettings,
    ) -> Self:
        """Initialize the goal penalty and observation precision traces.

        The baseline map must emit one component for the scalar goal. Each step retains a
        fraction of the previous traces equal to the exponential of the negated decay rate.
        """
        return cls(
            trace_decay_rate=FixedParameter(
                jnp.asarray(credit_settings.trace_decay_rate, dtype=jnp.float64)
            ),
            goal_baseline_map=goal_baseline_map,
            trace_index=eqx.nn.StateIndex(CreditTraces.zeros(n)),
        )

    def trace_decay(self) -> JaxRealArray:
        """Return the fraction of each trace retained at the next emission."""
        return jnp.exp(-self.trace_decay_rate.value)

    def step(
        self,
        state: eqx.nn.State,
        *,
        gain_observation: NormalNP | None,
        noised_instrument: NormalNP,
        observation_content: NormalNP,
        raw_observation: NormalNP,
        goal: JaxRealArray,
    ) -> tuple[eqx.nn.State, CreditOutputs]:
        """Advance the credit traces and compute training signals for this emission.

        The goal prediction combines the current baseline and the penalty accumulated at
        past and current emissions. Hold the arriving goal and the baseline's observation
        content fixed when fitting that prediction. Apply gain once to each new trace contribution.
        Weight shrinkage credit for the current raw mean by the updated observation precision trace.
        """
        traces = state.get(self.trace_index)
        decay = self.trace_decay()

        instrument = noised_instrument.to_precision_parametrization()
        raw = raw_observation.to_precision_parametrization()
        gain = 1.0 if gain_observation is None else gain_observation.mean()
        mean_contribution = gain * instrument.precision * instrument.mean
        precision_contribution = -0.5 * gain * jnp.square(instrument.mean)

        new_traces = CreditTraces(
            goal_penalty=decay * traces.goal_penalty
            + jnp.sum(raw.precision * precision_contribution),
            obs_precision=decay * traces.obs_precision + gain * raw.precision,
        )
        new_state = state.set(self.trace_index, new_traces)

        baseline_message = self.goal_baseline_map.infer(stop_gradient(observation_content))
        baseline = baseline_message.mean().reshape(())
        error = stop_gradient(goal) - (baseline + new_traces.goal_penalty)
        # The goal prediction loss trains only the baseline map; precision credit reaches the
        # penalty through the precision trace instead.
        fixed_penalty_error = stop_gradient(goal - new_traces.goal_penalty) - baseline
        return new_state, CreditOutputs(
            mean_contribution=mean_contribution,
            precision_contribution=precision_contribution,
            goal_prediction_error=error,
            goal_prediction_loss=0.5 * jnp.square(fixed_penalty_error),
            shrinkage_cotangent=-raw.mean * new_traces.obs_precision,
        )
