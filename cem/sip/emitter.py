"""SIP emitter circuit: observation map, instrument map, noise top-up, gain.

See `sip-emitter` in `~/src/typst/thesis.typ`.
"""

from typing import Self

import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
from efax import NormalNP
from tjax import JaxRealArray, KeyArray

from cem.npn.gaussian import GaussianNPN, normal_from_mean_and_presence
from cem.sip.credit import Credit, CreditOutputs
from cem.sip.messages import evidence_scale, mean_and_precision, promoting_link_loss


class EmitterOutputs(eqx.Module):
    """One step of the emitter circuit."""

    observation: NormalNP
    instrument: NormalNP
    raw_observation: NormalNP
    noised_instrument: NormalNP
    local_variance: JaxRealArray
    credit_outputs: CreditOutputs
    loss: JaxRealArray


class Emitter(eqx.Module):
    """One feature's emitter: constructs its observation and instrument.

    Transforms observation content and predictor instruments through neural maps,
    injects fresh noise at controllable features to create experimental variation,
    and evidence-scales both messages by gain.
    """

    observation_map: GaussianNPN
    instrument_map: GaussianNPN
    credit: Credit
    controllable: bool = eqx.field(static=True)

    def step(
        self,
        state: eqx.nn.State,
        key: KeyArray,
        *,
        observation_content: NormalNP,
        predictor_instruments: NormalNP,
        gain_observation: NormalNP | None,
        goal: JaxRealArray,
    ) -> tuple[eqx.nn.State, EmitterOutputs]:
        """Advance the emitter one time step.

        Pass `gain_observation=None` for a recruiter feature at the top of the
        gain chain; evidence scaling is skipped and `Credit` sees unit gain.
        """
        raw_observation = self.observation_map.infer(*mean_and_precision(observation_content))
        inherited_instrument = self.instrument_map.infer(*mean_and_precision(predictor_instruments))
        inherited_mean, inherited_precision = mean_and_precision(inherited_instrument)

        if self.controllable:
            local_variance = jnp.maximum(0.0, 1 / inherited_precision - jnp.square(inherited_mean))
            displacement = jnp.sqrt(local_variance) * jr.normal(key, inherited_mean.shape)
            noised_raw_observation = _shift_mean(raw_observation, displacement)
            noised_instrument = _shift_mean(inherited_instrument, displacement)
        else:
            local_variance = jnp.zeros_like(inherited_mean)
            noised_raw_observation = raw_observation
            noised_instrument = inherited_instrument

        if gain_observation is None:
            observation = noised_raw_observation
            instrument = noised_instrument
        else:
            gain, _ = mean_and_precision(gain_observation)
            observation = evidence_scale(noised_raw_observation, gain)
            instrument = evidence_scale(noised_instrument, gain)

        new_state, credit_outputs = self.credit.step(
            state,
            gain_observation=gain_observation,
            noised_instrument=noised_instrument,
            observation_content=observation_content,
            raw_observation=raw_observation,
            goal=goal,
        )

        raw_moments = jnp.stack(list(mean_and_precision(raw_observation)), axis=-1)
        promoting_loss = promoting_link_loss(raw_moments, credit_outputs.cotangent)
        outputs = EmitterOutputs(
            observation=observation,
            instrument=instrument,
            raw_observation=raw_observation,
            noised_instrument=noised_instrument,
            local_variance=local_variance,
            credit_outputs=credit_outputs,
            loss=credit_outputs.loss + promoting_loss,
        )
        return new_state, outputs

    @classmethod
    def create(
        cls,
        observation_map: GaussianNPN,
        instrument_map: GaussianNPN,
        credit: Credit,
        *,
        controllable: bool,
    ) -> Self:
        return cls(
            observation_map=observation_map,
            instrument_map=instrument_map,
            credit=credit,
            controllable=controllable,
        )


def _shift_mean(message: NormalNP, displacement: JaxRealArray) -> NormalNP:
    """Shift a Gaussian message's mean by `displacement`, preserving precision."""
    mean, precision = mean_and_precision(message)
    return normal_from_mean_and_presence(mean + displacement, precision)
