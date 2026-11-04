"""SIP credit circuit: eligibility-based update rule for one emitter.

See `sip-credit-circuit` in `~/src/typst/thesis.typ`.
"""

from typing import Self

import equinox as eqx
import jax.numpy as jnp
from efax import NormalNP
from tjax import JaxRealArray

from cem.npn.gaussian import GaussianNPN
from cem.sip.messages import mean_and_precision, promoting_link_loss
from cem.structure.graph import FixedParameter


class CreditTraces(eqx.Module):
    """Three eligibility traces per channel, held in the Equinox state carry."""

    linear: JaxRealArray  # shape (n,)
    quadratic: JaxRealArray  # shape (n,)
    obs_precision: JaxRealArray  # shape (n,)

    @classmethod
    def zeros(cls, n: int) -> Self:
        zero = jnp.zeros(n, dtype=jnp.float64)
        return cls(linear=zero, quadratic=zero, obs_precision=zero)


class CreditOutputs(eqx.Module):
    """One step of credit assignment."""

    cotangent: JaxRealArray  # shape (n, 2): column 0 mean, column 1 precision
    loss: JaxRealArray  # shape (): baseline-map promoting-link surrogate


class Credit(eqx.Module):
    """Eligibility-based update rule for one emitter's `n` channels.

    Lives inside `Emitter`. Each step reads the current traces from the Equinox
    state carry, advances them, and returns a cotangent for the emitter's
    observation plus a dummy loss that trains the goal-baseline NPN through a
    promoting link.
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
        trace_decay_rate: float = 0.0,
    ) -> Self:
        return cls(
            trace_decay_rate=FixedParameter(jnp.asarray(trace_decay_rate, dtype=jnp.float64)),
            goal_baseline_map=goal_baseline_map,
            trace_index=eqx.nn.StateIndex(CreditTraces.zeros(n)),
        )

    def step(
        self,
        state: eqx.nn.State,
        *,
        gain_observation: NormalNP,
        noised_instrument: NormalNP,
        observation_content: NormalNP,
        raw_observation: NormalNP,
        goal: JaxRealArray,
    ) -> tuple[eqx.nn.State, CreditOutputs]:
        """Advance the credit circuit one time step.

        `goal` has shape `()`. All Gaussian inputs have shape `(n,)` in natural
        parameters, except `observation_content`, whose shape is the baseline
        NPN's input dimension.
        """
        traces = state.get(self.trace_index)
        decay = jnp.exp(-self.trace_decay_rate.value)

        inst_mean, inst_precision = mean_and_precision(noised_instrument)
        raw_mean, raw_precision = mean_and_precision(raw_observation)
        gain, _ = mean_and_precision(gain_observation)

        def advance(old: JaxRealArray, contribution: JaxRealArray) -> JaxRealArray:
            return decay * old + gain * contribution

        new_traces = CreditTraces(
            linear=advance(traces.linear, inst_precision * inst_mean),
            quadratic=advance(traces.quadratic, jnp.square(inst_mean)),
            obs_precision=advance(traces.obs_precision, raw_precision),
        )
        new_state = state.set(self.trace_index, new_traces)

        mean_cotangent = goal * new_traces.linear - raw_mean * new_traces.obs_precision

        oc_mean, oc_precision = mean_and_precision(observation_content)
        baseline_message = self.goal_baseline_map.infer(oc_mean, oc_precision)
        baseline_values, _ = mean_and_precision(baseline_message)
        baseline = baseline_values.reshape(())

        error = goal - baseline + 0.5 * jnp.sum(raw_precision * new_traces.quadratic)
        precision_cotangent = -0.5 * new_traces.quadratic * error
        loss = promoting_link_loss(baseline, error)

        cotangent = jnp.stack([mean_cotangent, precision_cotangent], axis=-1)
        return new_state, CreditOutputs(cotangent=cotangent, loss=loss)
