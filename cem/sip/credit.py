"""Training signals for an emitter’s observation projection."""

from typing import Self

import equinox as eqx
import jax.numpy as jnp
from efax import NormalNP
from jax.lax import stop_gradient
from tjax import JaxRealArray, PyTree
from tjax.filtering import DelayLine, Filter

from cem.framework.graph import FixedParameter
from cem.npn.gaussian import GaussianNeuralProjection


class CreditState(eqx.Module):
    """What the credit circuit retains between steps.

    `goal_penalty` and `emission_weight` are the lag filter's states for the goal penalty of past
    emissions and for a unit drift at each emission, whose value is the weight of the emissions
    that an arriving goal follows. `observation_content` is the content of the latest emission,
    which the goal baseline reads when that emission's goal arrives.
    """

    goal_penalty: PyTree
    emission_weight: PyTree
    observation_content: NormalNP


class CreditEmission(eqx.Module):
    """One emission's contributions, goal penalty, content, and shrinkage cotangent.

    `shrinkage_cotangent` is shrinkage credit in raw-mean coordinates: the descent cotangent of the
    mean shrinkage loss, whose weight is the emission's gain-weighted raw precision. The
    observation projection pulls it back through the map as it ran. See `sip-shrinkage-credit` in
    the thesis.
    """

    mean_contribution: JaxRealArray
    precision_contribution: JaxRealArray
    goal_penalty: JaxRealArray  # scalar
    observation_content: NormalNP
    shrinkage_cotangent: JaxRealArray


class CreditOutputs(eqx.Module):
    """The goal prediction error and the loss that trains the goal baseline."""

    goal_prediction_error: JaxRealArray  # scalar
    goal_prediction_loss: JaxRealArray  # scalar


class Credit(eqx.Module):
    """Assign credit to an emitter's experiments when later goals arrive.

    Each step has two phases. `emit` records the current emission's contributions, and `credit`
    pairs an arriving goal with the emissions that the lag filter retains. A circuit whose goal
    arrives later in the step can run the phases apart. See `sip-credit-circuit` in the thesis.
    """

    lag_filter: FixedParameter[Filter[PyTree]]
    goal_baseline_map: GaussianNeuralProjection
    state_index: eqx.nn.StateIndex[CreditState]

    @classmethod
    def create(cls, goal_baseline_map: GaussianNeuralProjection) -> Self:
        """Start with no retained emissions.

        The baseline map must emit one component for the scalar goal.
        """
        # Read before it is updated, a one-slot delay line holds the previous step's drift, so each
        # goal credits only the emission one step before it. See `eligibility-training-link` in the
        # thesis.
        lag_filter = DelayLine(jnp.ones(1, dtype=jnp.float64))
        absent = jnp.zeros(goal_baseline_map.in_features, dtype=jnp.float64)
        scalar = jnp.zeros((), dtype=jnp.float64)
        initial = CreditState(
            goal_penalty=lag_filter.init(scalar),
            emission_weight=lag_filter.init(scalar),
            observation_content=NormalNP(absent, absent),
        )
        return cls(
            lag_filter=FixedParameter(lag_filter),
            goal_baseline_map=goal_baseline_map,
            state_index=eqx.nn.StateIndex(initial),
        )

    def emit(
        self,
        *,
        gain_observation: NormalNP | None,
        noised_instrument: NormalNP,
        observation_content: NormalNP,
        raw_observation: NormalNP,
    ) -> CreditEmission:
        """Record this emission's contributions and its shrinkage cotangent.

        Apply gain once to each contribution. The shrinkage weight is the emission's gain-weighted
        raw precision.
        """
        instrument = noised_instrument.to_precision_parametrization()
        raw = raw_observation.to_precision_parametrization()
        gain = 1.0 if gain_observation is None else gain_observation.mean()
        mean_contribution = gain * instrument.precision * instrument.mean
        precision_contribution = -0.5 * gain * jnp.square(instrument.mean)
        return CreditEmission(
            mean_contribution=mean_contribution,
            precision_contribution=precision_contribution,
            goal_penalty=jnp.sum(raw.precision * precision_contribution),
            observation_content=observation_content,
            shrinkage_cotangent=-raw.mean * gain * raw.precision,
        )

    def credit(
        self, state: eqx.nn.State, emission: CreditEmission, goal: JaxRealArray
    ) -> tuple[eqx.nn.State, CreditOutputs]:
        """Predict the arriving goal from the retained emissions, then retain this emission.

        The goal prediction is the baseline, read from the content of the emission that the goal
        follows, plus the lag-filtered goal penalty. Hold the arriving goal and that content fixed
        when fitting the prediction. The fit is weighted by the retained emission weight, so a goal
        that follows no emission, such as the first, trains nothing.
        """
        retained = state.get(self.state_index)
        lag_filter = self.lag_filter.value
        goal_penalty = lag_filter.value(retained.goal_penalty)
        emission_weight = lag_filter.value(retained.emission_weight)
        baseline_message = self.goal_baseline_map.infer(stop_gradient(retained.observation_content))
        baseline = baseline_message.mean().reshape(())
        error = stop_gradient(goal) - (baseline + goal_penalty)
        # The goal prediction loss trains only the baseline map; precision credit reaches the
        # penalty through the precision trace instead.
        fixed_penalty_error = stop_gradient(goal - goal_penalty) - baseline
        new_state = state.set(
            self.state_index,
            CreditState(
                goal_penalty=lag_filter.update(retained.goal_penalty, emission.goal_penalty),
                emission_weight=lag_filter.update(
                    retained.emission_weight, jnp.ones((), dtype=jnp.float64)
                ),
                observation_content=emission.observation_content,
            ),
        )
        return new_state, CreditOutputs(
            goal_prediction_error=error,
            goal_prediction_loss=0.5 * emission_weight * jnp.square(fixed_penalty_error),
        )
