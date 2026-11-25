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
from cem.sip.messages import InstrumentedMessage


def predictor_substrate(predictor: InstrumentedMessage) -> NormalNP:
    """Return the predictor observations with their paired instruments' means subtracted.

    The substrate removes the parents' experimental displacements from their observations, so a
    baseline that reads it cannot see the experiment that the feature inherits. Each observation
    keeps its precision, and an absent instrument subtracts nothing. See `sip-credit-architecture`
    in the thesis.
    """
    observation = predictor.observation
    displacement = predictor.instrument.mean()
    # Shifting the mean by -d at fixed precision κ adds -κ d = 2 η₂ d to the first natural
    # parameter, which stays defined for an absent observation.
    return NormalNP(
        observation.mean_times_precision + 2.0 * observation.negative_half_precision * displacement,
        observation.negative_half_precision,
    )


class CreditState(eqx.Module):
    """What the credit circuit retains between steps.

    `quadratic_cost` and `emission_weight` are the lag filter's states for the quadratic cost of
    past emissions and for a unit drift at each emission, whose value is the weight of the
    emissions that an arriving goal follows. `predictor_substrate` is the substrate of the latest
    emission, which both baselines read when that emission's goal arrives.
    """

    quadratic_cost: PyTree
    emission_weight: PyTree
    predictor_substrate: NormalNP


class CreditEmission(eqx.Module):
    """One emission's contributions, quadratic cost, substrate, and shrinkage cotangent.

    `mean_contribution` and `precision_contribution` are the raw-mean and raw-precision cotangents
    that the observation projection pulls back and records. `quadratic_cost` weights the squared
    experimental displacements by raw precision. `shrinkage_cotangent` is shrinkage credit in
    raw-mean coordinates: the descent cotangent of the mean shrinkage loss, whose weight is the
    emission's gain-weighted raw precision. See `sip-eligibility-traces` in the thesis.
    """

    mean_contribution: JaxRealArray
    precision_contribution: JaxRealArray
    quadratic_cost: JaxRealArray  # scalar
    predictor_substrate: NormalNP
    shrinkage_cotangent: JaxRealArray


class CreditOutputs(eqx.Module):
    """The residuals that scale the recorded cotangents and the losses that train the baselines.

    The baseline goal residual scales mean credit and the quadratic goal residual scales precision
    credit. Both are held fixed, so credit does not train the baselines.
    """

    baseline_goal_residual: JaxRealArray  # scalar
    quadratic_goal_residual: JaxRealArray  # scalar
    goal_baseline_loss: JaxRealArray  # scalar
    quadratic_cost_loss: JaxRealArray  # scalar


class Credit(eqx.Module):
    """Assign credit to an emitter's experiments when later goals arrive.

    Each step has two phases. `emit` records the current emission's contributions, and `credit`
    pairs an arriving goal with the emission that the lag filter retains. A circuit whose goal
    arrives later in the step can run the phases apart. The goal baseline predicts the goal, and
    the quadratic cost baseline predicts the experiment's cost, both from the retained emission's
    predictor substrate. See `sip-credit-circuit` in the thesis.
    """

    lag_filter: FixedParameter[Filter[PyTree]]
    goal_baseline_map: GaussianNeuralProjection
    quadratic_cost_baseline_map: GaussianNeuralProjection
    state_index: eqx.nn.StateIndex[CreditState]

    @classmethod
    def create(
        cls,
        goal_baseline_map: GaussianNeuralProjection,
        quadratic_cost_baseline_map: GaussianNeuralProjection,
    ) -> Self:
        """Start with no retained emissions.

        Both baseline maps read the predictor substrate and emit one component for their scalar
        target.
        """
        if quadratic_cost_baseline_map.in_features != goal_baseline_map.in_features:
            msg = "Both baseline maps must read the same predictor substrate"
            raise ValueError(msg)
        # Read before it is updated, a one-slot delay line holds the previous step's drift, so each
        # goal credits only the emission one step before it. See `cotangent-training-link` in the
        # thesis.
        lag_filter = DelayLine(jnp.ones(1, dtype=jnp.float64))
        absent = jnp.zeros(goal_baseline_map.in_features, dtype=jnp.float64)
        scalar = jnp.zeros((), dtype=jnp.float64)
        initial = CreditState(
            quadratic_cost=lag_filter.init(scalar),
            emission_weight=lag_filter.init(scalar),
            predictor_substrate=NormalNP(absent, absent),
        )
        return cls(
            lag_filter=FixedParameter(lag_filter),
            goal_baseline_map=goal_baseline_map,
            quadratic_cost_baseline_map=quadratic_cost_baseline_map,
            state_index=eqx.nn.StateIndex(initial),
        )

    def emit(
        self,
        *,
        gain_observation: NormalNP | None,
        noised_instrument: NormalNP,
        predictor: InstrumentedMessage,
        raw_observation: NormalNP,
    ) -> CreditEmission:
        """Record this emission's contributions, cost, substrate, and shrinkage cotangent.

        Apply gain once to each contribution. The quadratic cost weights the gain-weighted squared
        displacement by raw precision, and the shrinkage weight is the emission's gain-weighted raw
        precision.
        """
        instrument = noised_instrument.to_precision_parametrization()
        raw = raw_observation.to_precision_parametrization()
        gain = 1.0 if gain_observation is None else gain_observation.mean()
        quadratic_contribution = gain * jnp.square(instrument.mean)
        return CreditEmission(
            mean_contribution=gain * instrument.precision * instrument.mean,
            precision_contribution=-0.5 * quadratic_contribution,
            quadratic_cost=0.5 * jnp.sum(raw.precision * quadratic_contribution),
            predictor_substrate=predictor_substrate(predictor),
            shrinkage_cotangent=-gain * raw.precision * raw.mean,
        )

    def credit(
        self, state: eqx.nn.State, emission: CreditEmission, goal: JaxRealArray
    ) -> tuple[eqx.nn.State, CreditOutputs]:
        """Score the arriving goal against the retained emission, then retain this emission.

        The goal baseline is fitted to the goal and the quadratic cost baseline to the retained
        emission's cost, both read from that emission's substrate. Each fit holds its target and
        the substrate fixed and is weighted by the retained emission weight, so a goal that follows
        no emission, such as the first, trains nothing.
        """
        retained = state.get(self.state_index)
        lag_filter = self.lag_filter.value
        quadratic_cost = stop_gradient(lag_filter.value(retained.quadratic_cost))
        emission_weight = lag_filter.value(retained.emission_weight)
        substrate = stop_gradient(retained.predictor_substrate)
        goal_baseline = self.goal_baseline_map.infer(substrate).mean().reshape(())
        cost_baseline = self.quadratic_cost_baseline_map.infer(substrate).mean().reshape(())
        baseline_goal_residual = stop_gradient(goal) - goal_baseline
        cost_residual = quadratic_cost - cost_baseline
        new_state = state.set(
            self.state_index,
            CreditState(
                quadratic_cost=lag_filter.update(retained.quadratic_cost, emission.quadratic_cost),
                emission_weight=lag_filter.update(
                    retained.emission_weight, jnp.ones((), dtype=jnp.float64)
                ),
                predictor_substrate=emission.predictor_substrate,
            ),
        )
        return new_state, CreditOutputs(
            baseline_goal_residual=stop_gradient(baseline_goal_residual),
            quadratic_goal_residual=stop_gradient(baseline_goal_residual + cost_residual),
            goal_baseline_loss=0.5 * emission_weight * jnp.square(baseline_goal_residual),
            quadratic_cost_loss=0.5 * emission_weight * jnp.square(cost_residual),
        )
