"""Check parameter credit from an earlier emission with different inputs."""

import equinox as eqx
import jax
import jax.numpy as jnp
from tjax import JaxRealArray

from cem.framework.graph import EligibilityTraining, FixedParameter, LearnableParameter


class _Map(eqx.Module):
    weight: LearnableParameter[JaxRealArray]
    bias: FixedParameter[JaxRealArray]


def _project(model: _Map, x: JaxRealArray) -> tuple[JaxRealArray, JaxRealArray]:
    output = model.weight.value * x + model.bias.value
    return output, output


def test_drift_is_each_emissions_parameter_cotangent() -> None:
    training = EligibilityTraining(
        _Map(LearnableParameter(jnp.asarray(2.0)), FixedParameter(jnp.asarray(1.0))),
        _project,
    )

    @eqx.filter_jit
    def emit(
        wrapper: EligibilityTraining[_Map, JaxRealArray, JaxRealArray, JaxRealArray],
        x: JaxRealArray,
    ) -> tuple[JaxRealArray, _Map]:
        observation, pullback = wrapper.infer(x)
        return observation, wrapper.drift(pullback, jnp.asarray(1.0))

    observation, first_drift = emit(training, jnp.asarray(4.0))
    _, second_drift = emit(training, jnp.asarray(3.0))

    assert jnp.allclose(observation, 9)
    assert jnp.allclose(first_drift.weight.value, 4)
    assert jnp.allclose(second_drift.weight.value, 3)

    # A later goal credits the drift recorded at the earlier emission, not the current input.
    def loss(weight: JaxRealArray, factor: JaxRealArray) -> JaxRealArray:
        wrapper = eqx.tree_at(lambda box: box.model.weight.value, training, weight)
        return wrapper.get_loss(first_drift, factor)

    weight_grad, factor_grad = jax.grad(loss, argnums=(0, 1))(
        training.model.weight.value, jnp.asarray(2.0)
    )
    assert jnp.allclose(weight_grad, -8)
    assert factor_grad == 0
