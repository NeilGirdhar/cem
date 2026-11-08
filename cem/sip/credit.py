"""Training signals for an emitter’s observation projection."""

from typing import Self

import equinox as eqx
import jax.numpy as jnp
from efax import NormalNP
from jax.lax import stop_gradient
from tjax import JaxRealArray

from cem.npn.gaussian import GaussianNPN
from cem.sip.messages import mean_and_precision
from cem.sip.settings import CreditSettings
from cem.structure.graph import FixedParameter


class CreditTraces(eqx.Module):
    """Retain the predicted penalty and observation precision of past experiments."""

    goal_penalty: JaxRealArray
    obs_precision: JaxRealArray

    @classmethod
    def zeros(cls, n: int) -> Self:
        return cls(jnp.zeros((), dtype=jnp.float64), jnp.zeros(n, dtype=jnp.float64))


class CreditOutputs(eqx.Module):
    """Trace contributions, shrinkage credit, and the signals for training the goal baseline."""

    mean_contribution: JaxRealArray
    precision_contribution: JaxRealArray
    goal_prediction_error: JaxRealArray
    shrinkage_cotangent: JaxRealArray
    goal_baseline: JaxRealArray


class Credit(eqx.Module):
    """Assign credit to an emitter's experiments when later goals arrive.

    See `sip-credit-circuit` in the thesis.
    """

    trace_decay_rate: FixedParameter[JaxRealArray]
    goal_baseline_map: GaussianNPN
    trace_index: eqx.nn.StateIndex[CreditTraces]

    @classmethod
    def create(
        cls,
        n: int,
        goal_baseline_map: GaussianNPN,
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
        """
        traces = state.get(self.trace_index)
        decay = self.trace_decay()

        inst_mean, inst_precision = mean_and_precision(noised_instrument)
        raw_mean, raw_precision = mean_and_precision(raw_observation)
        gain = 1.0 if gain_observation is None else mean_and_precision(gain_observation)[0]
        mean_contribution = gain * inst_precision * inst_mean
        precision_contribution = -0.5 * gain * jnp.square(inst_mean)

        new_traces = CreditTraces(
            goal_penalty=decay * traces.goal_penalty
            + jnp.sum(raw_precision * precision_contribution),
            obs_precision=decay * traces.obs_precision + gain * raw_precision,
        )
        new_state = state.set(self.trace_index, new_traces)

        baseline_message = self.goal_baseline_map.infer(
            *mean_and_precision(stop_gradient(observation_content))
        )
        baseline_values, _ = mean_and_precision(baseline_message)
        baseline = baseline_values.reshape(())
        error = stop_gradient(goal) - (baseline + new_traces.goal_penalty)
        return new_state, CreditOutputs(
            mean_contribution=mean_contribution,
            precision_contribution=precision_contribution,
            goal_prediction_error=error,
            shrinkage_cotangent=-raw_mean * new_traces.obs_precision,
            goal_baseline=baseline,
        )
