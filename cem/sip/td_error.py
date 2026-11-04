"""SIP TD-error circuit: reward prediction with bootstrap TD and confounding witness.

See `sip-td-error-circuit` in `~/src/typst/thesis.typ`.
"""

from collections.abc import Mapping
from typing import Self

import equinox as eqx
import jax
import jax.numpy as jnp
from efax import NormalNP
from tjax import JaxRealArray, RngStream

from cem.npn.gaussian import GaussianNPN, normal_from_mean_and_presence
from cem.perceptron.mlp import MLP
from cem.sip.messages import mean_and_precision, promoting_link_loss, variance_from_precision
from cem.structure.graph import FixedParameter, LearnableParameter


class TDTraces(eqx.Module):
    """One-step delay buffer: the previous step's predictor inputs.

    Held in the Equinox state carry. We re-run the prediction and witness maps
    on these inputs each step to recover the delayed prediction and witness
    with a live gradient to the circuit's parameters.
    """

    predictor_observations: NormalNP
    predictor_instruments: NormalNP

    @classmethod
    def initial(cls, predictor_features: int) -> Self:
        def placeholder(size: int) -> NormalNP:
            return normal_from_mean_and_presence(
                jnp.zeros(size, dtype=jnp.float64),
                jnp.ones(size, dtype=jnp.float64),
            )

        return cls(
            predictor_observations=placeholder(predictor_features),
            predictor_instruments=placeholder(predictor_features),
        )


class TDErrorOutputs(eqx.Module):
    """One step of the TD-error circuit."""

    advantage: JaxRealArray  # shape (n,); outgoing TD-error mean signal
    reconstruction_cost: JaxRealArray  # scalar
    confounding_cost: JaxRealArray  # scalar
    uncertainty_loss: JaxRealArray  # scalar
    witness_loss: JaxRealArray  # scalar
    loss: JaxRealArray  # scalar; combined surrogate for first-pass training


class _Moments(eqx.Module):
    boot_mean: JaxRealArray
    boot_variance: JaxRealArray
    del_mean: JaxRealArray
    del_variance: JaxRealArray


class TDError(eqx.Module):
    """One reward feature's TD-error circuit.

    Delays prediction and witness by one step through the state carry. The
    bootstrap target convolves the current reward observation with the current
    prediction, both stopped. Learning trains the delayed prediction to match
    this target under reconstruction, purification, and an uncertainty penalty,
    and the witness to find instrument-predictable residuals under a unit-norm
    bound. The outgoing advantage is the bootstrap-vs-delayed mean difference.
    """

    prediction_map: GaussianNPN
    witness_map: MLP
    reconstruction_cost_weight: LearnableParameter[JaxRealArray]  # raw (n,)
    confounding_cost_weight: LearnableParameter[JaxRealArray]  # raw (n, 2)
    uncertainty_loss_weight: LearnableParameter[JaxRealArray]  # raw (n,)
    witness_multiplier: LearnableParameter[JaxRealArray]  # raw ()
    discount: FixedParameter[JaxRealArray]  # scalar
    delay_index: eqx.nn.StateIndex[TDTraces]

    @classmethod
    def create(
        cls,
        n: int,
        *,
        predictor_features: int,
        discount: float = 1.0,
        prediction_hidden_features: int | tuple[int, ...] = (),
        witness_hidden_features: int | tuple[int, ...] = (),
        streams: Mapping[str, RngStream],
    ) -> Self:
        return cls(
            prediction_map=GaussianNPN.create(
                in_features=predictor_features,
                out_features=n,
                hidden_features=prediction_hidden_features,
                streams=streams,
            ),
            witness_map=MLP.create(
                in_features=2 * predictor_features,
                out_features=2 * n,
                hidden_features=witness_hidden_features,
                streams=streams,
            ),
            reconstruction_cost_weight=LearnableParameter(jnp.zeros(n, dtype=jnp.float64)),
            confounding_cost_weight=LearnableParameter(jnp.zeros((n, 2), dtype=jnp.float64)),
            uncertainty_loss_weight=LearnableParameter(jnp.zeros(n, dtype=jnp.float64)),
            witness_multiplier=LearnableParameter(jnp.asarray(0.0, dtype=jnp.float64)),
            discount=FixedParameter(jnp.asarray(discount, dtype=jnp.float64)),
            delay_index=eqx.nn.StateIndex(TDTraces.initial(predictor_features)),
        )

    def step(
        self,
        state: eqx.nn.State,
        *,
        reward_observation: NormalNP,
        predictor_observations: NormalNP,
        predictor_instruments: NormalNP,
        streams: Mapping[str, RngStream],
        inference: bool,
    ) -> tuple[eqx.nn.State, TDErrorOutputs]:
        """Advance the TD-error circuit one time step."""
        delayed = state.get(self.delay_index)
        delayed_prediction = self.prediction_map.infer(
            *mean_and_precision(delayed.predictor_observations)
        )
        delayed_witness = self._witness(
            delayed.predictor_instruments, streams=streams, inference=inference
        )
        current_prediction = self.prediction_map.infer(*mean_and_precision(predictor_observations))

        moments = self._moments(reward_observation, current_prediction, delayed_prediction)
        witness_residuals = _witness_residuals(moments)

        # Loss weights are LearnableParameters now, but their adaptive rule
        # lives in the (not-yet-implemented) multi-loss gradient transform.
        # Stop-gradient the readout so single-loss Adam doesn't collapse them.
        r_weight = jax.nn.softplus(jax.lax.stop_gradient(self.reconstruction_cost_weight.value))
        c_weight = jax.nn.softplus(jax.lax.stop_gradient(self.confounding_cost_weight.value))
        u_weight = jax.nn.softplus(jax.lax.stop_gradient(self.uncertainty_loss_weight.value))

        reconstruction_cost = _reconstruction_cost(moments, r_weight)
        confounding_cost = jnp.sum(
            c_weight * witness_residuals * jax.lax.stop_gradient(delayed_witness)
        )
        uncertainty_loss = jnp.sum(u_weight * moments.del_variance)
        witness_loss, multiplier_loss = self._witness_training(witness_residuals, delayed_witness)

        advantage = moments.boot_mean - moments.del_mean

        new_state = state.set(
            self.delay_index,
            TDTraces(
                predictor_observations=predictor_observations,
                predictor_instruments=predictor_instruments,
            ),
        )

        return new_state, TDErrorOutputs(
            advantage=advantage,
            reconstruction_cost=reconstruction_cost,
            confounding_cost=confounding_cost,
            uncertainty_loss=uncertainty_loss,
            witness_loss=witness_loss,
            loss=(
                reconstruction_cost
                + confounding_cost
                + uncertainty_loss
                + witness_loss
                + multiplier_loss
            ),
        )

    def _witness(
        self,
        predictor_instruments: NormalNP,
        *,
        streams: Mapping[str, RngStream],
        inference: bool,
    ) -> JaxRealArray:
        witness_input = jnp.concatenate(
            [
                predictor_instruments.mean_times_precision,
                predictor_instruments.negative_half_precision,
            ]
        )
        return self.witness_map.infer(witness_input, streams=streams, inference=inference).reshape(
            -1, 2
        )

    def _moments(
        self,
        reward_observation: NormalNP,
        current_prediction: NormalNP,
        delayed_prediction: NormalNP,
    ) -> _Moments:
        reward_mean, reward_precision = mean_and_precision(reward_observation)
        cur_mean, cur_precision = mean_and_precision(current_prediction)
        del_mean, del_precision = mean_and_precision(delayed_prediction)

        gamma = self.discount.value
        boot_mean = jax.lax.stop_gradient(reward_mean) + gamma * jax.lax.stop_gradient(cur_mean)
        boot_variance = jax.lax.stop_gradient(variance_from_precision(reward_precision)) + (
            jnp.square(gamma) * jax.lax.stop_gradient(jnp.reciprocal(cur_precision))
        )
        return _Moments(
            boot_mean=boot_mean,
            boot_variance=boot_variance,
            del_mean=del_mean,
            del_variance=jnp.reciprocal(del_precision),
        )

    def _witness_training(
        self,
        witness_residuals: JaxRealArray,
        confounding_witness: JaxRealArray,
    ) -> tuple[JaxRealArray, JaxRealArray]:
        covariance = -jnp.sum(jax.lax.stop_gradient(witness_residuals) * confounding_witness)
        # Unit-gain special case of the thesis's E[||w||²] ≤ E[mean(gain)] bound
        # (sip-norm-constraint): with gain = 1, the RHS is 1.
        constraint_violation = jnp.sum(jnp.square(confounding_witness)) - 1.0
        lambda_value = jax.nn.softplus(self.witness_multiplier.value)
        witness_loss = covariance + jax.lax.stop_gradient(lambda_value) * constraint_violation
        multiplier_loss = promoting_link_loss(
            lambda_value, jax.lax.stop_gradient(constraint_violation)
        )
        return witness_loss, multiplier_loss


def _reconstruction_cost(moments: _Moments, r_weight: JaxRealArray) -> JaxRealArray:
    mean_resid = moments.del_mean - moments.boot_mean
    recon_error = 0.5 * (
        jnp.log(2.0 * jnp.pi * moments.del_variance)
        + (moments.boot_variance + jnp.square(mean_resid)) / moments.del_variance
    )
    return jnp.sum(r_weight * recon_error)


def _witness_residuals(moments: _Moments) -> JaxRealArray:
    mean_resid = moments.del_mean - moments.boot_mean
    var_resid = moments.del_variance - moments.boot_variance - jnp.square(mean_resid)
    return jnp.stack([mean_resid, var_resid], axis=-1)
