"""Training signals for an emitter’s observation projection."""

from typing import Self

import equinox as eqx
import jax.numpy as jnp
from efax import NormalNP
from jax.lax import stop_gradient
from tjax import JaxRealArray

from cem.npn.gaussian import GaussianNeuralProjection
from cem.sip.settings import CreditSettings
from cem.structure.graph import FixedParameter


class CreditTraces(eqx.Module):
    """Retain each goal component's predicted penalty and curvature from past experiments."""

    goal_penalty: JaxRealArray  # (m,)
    curvature: JaxRealArray  # (m, n)

    @classmethod
    def zeros(cls, n: int, m: int) -> Self:
        return cls(jnp.zeros(m, dtype=jnp.float64), jnp.zeros((m, n), dtype=jnp.float64))


class CreditOutputs(eqx.Module):
    """Trace contributions, shrinkage credit, and the loss that trains the goal baselines.

    `shrinkage_cotangent` holds each goal component's shrinkage credit in raw-mean coordinates: the
    descent cotangent of its mean shrinkage loss, with its curvature trace held fixed. The
    observation projection pulls it back through the current map. See `sip-shrinkage-credit` in
    the thesis.
    """

    mean_contribution: JaxRealArray  # (n,)
    curvature_contribution: JaxRealArray  # (n,)
    goal_prediction_error: JaxRealArray  # (m,)
    goal_prediction_loss: JaxRealArray  # scalar
    shrinkage_cotangent: JaxRealArray  # (m, n)


class Credit(eqx.Module):
    """Assign credit to an emitter's experiments when later goals arrive.

    Each goal component has its own baseline, penalty trace, and curvature trace, so the
    components can be balanced against each other. See `sip-credit-circuit` in the thesis.
    """

    trace_decay_rate: FixedParameter[JaxRealArray]
    goal_baseline_map: GaussianNeuralProjection
    trace_index: eqx.nn.StateIndex[CreditTraces]

    @classmethod
    def create(
        cls,
        n: int,
        m: int,
        goal_baseline_map: GaussianNeuralProjection,
        *,
        credit_settings: CreditSettings,
    ) -> Self:
        """Initialize the goal penalty and curvature traces for `m` goal components.

        The baseline map must emit one component for each goal component. Each step retains a
        fraction of the previous traces equal to the exponential of the negated decay rate.
        """
        return cls(
            trace_decay_rate=FixedParameter(
                jnp.asarray(credit_settings.trace_decay_rate, dtype=jnp.float64)
            ),
            goal_baseline_map=goal_baseline_map,
            trace_index=eqx.nn.StateIndex(CreditTraces.zeros(n, m)),
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
        curvature: JaxRealArray,
        goals: JaxRealArray,
    ) -> tuple[eqx.nn.State, CreditOutputs]:
        """Advance the credit traces and compute training signals for this emission.

        `curvature` has one row of per-channel curvatures for each goal component, and the rows
        sum to the raw precision. `goals` has one value for each component. Each component's goal
        prediction combines its current baseline and the penalty accumulated at past and current
        emissions. Hold the arriving goals and the baseline's observation content fixed when
        fitting those predictions. Apply gain once to each new trace contribution. Weight each
        component's shrinkage credit for the current raw mean by its updated curvature trace.
        """
        traces = state.get(self.trace_index)
        decay = self.trace_decay()

        instrument = noised_instrument.to_precision_parametrization()
        raw = raw_observation.to_precision_parametrization()
        gain = jnp.ones_like(raw.mean) if gain_observation is None else gain_observation.mean()
        mean_contribution = gain * instrument.precision * instrument.mean
        curvature_contribution = -0.5 * gain * jnp.square(instrument.mean)

        new_traces = CreditTraces(
            goal_penalty=decay * traces.goal_penalty + curvature @ curvature_contribution,
            curvature=decay * traces.curvature + gain[jnp.newaxis, :] * curvature,
        )
        new_state = state.set(self.trace_index, new_traces)

        baseline = self.goal_baseline_map.infer(stop_gradient(observation_content)).mean()
        error = stop_gradient(goals) - (baseline + new_traces.goal_penalty)
        # The goal prediction loss trains only the baseline map; precision credit reaches the
        # penalties through the precision traces instead.
        fixed_penalty_error = stop_gradient(goals - new_traces.goal_penalty) - baseline
        return new_state, CreditOutputs(
            mean_contribution=mean_contribution,
            curvature_contribution=curvature_contribution,
            goal_prediction_error=error,
            goal_prediction_loss=0.5 * jnp.sum(jnp.square(fixed_penalty_error)),
            shrinkage_cotangent=-raw.mean[jnp.newaxis, :] * new_traces.curvature,
        )
