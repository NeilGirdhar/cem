"""SIP credit circuit: eligibility-based update rule for one emitter."""

from typing import Self

import equinox as eqx
import jax.numpy as jnp
from efax import NormalNP
from tjax import JaxRealArray

from cem.npn.gaussian import GaussianNPN
from cem.sip.messages import mean_and_precision, promoting_link_loss
from cem.sip.settings import CreditSettings
from cem.structure.graph import FixedParameter


class CreditTraces(eqx.Module):
    """Per-channel traces that retain evidence for assigning credit to past experiments."""

    linear: JaxRealArray  # shape (n,)
    quadratic: JaxRealArray  # shape (n,)
    obs_precision: JaxRealArray  # shape (n,)

    @classmethod
    def zeros(cls, n: int) -> Self:
        zero = jnp.zeros(n, dtype=jnp.float64)
        return cls(linear=zero, quadratic=zero, obs_precision=zero)


class CreditOutputs(eqx.Module):
    """Credit for the observation's mean and precision, with a loss for the goal baseline."""

    cotangent: JaxRealArray  # shape (n, 2): column 0 mean, column 1 precision
    loss: JaxRealArray  # shape (): baseline-map promoting-link surrogate


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
        """Initialize credit assignment with three zero traces per emitter channel.

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
        """Update eligibility traces and return credit for the raw observation's mean and precision.

        The traces carry experiments across steps so a later goal can credit them. The mean
        update uses the goal directly. The precision update compares it with a learned baseline
        and accounts for the accumulated squared instrument.

        Supply one scalar goal and one Gaussian component per emitter channel, except that
        `observation_content` has the baseline map's input width. With no gain observation,
        new trace contributions have unit gain.
        """
        traces = state.get(self.trace_index)
        decay = jnp.exp(-self.trace_decay_rate.value)

        inst_mean, inst_precision = mean_and_precision(noised_instrument)
        raw_mean, raw_precision = mean_and_precision(raw_observation)
        gain = None if gain_observation is None else mean_and_precision(gain_observation)[0]

        def advance(old: JaxRealArray, contribution: JaxRealArray) -> JaxRealArray:
            scaled = contribution if gain is None else gain * contribution
            return decay * old + scaled

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
