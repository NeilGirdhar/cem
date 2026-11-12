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


def test_delayed_credit_uses_each_emissions_parameter_cotangent() -> None:
    training = EligibilityTraining(
        _Map(LearnableParameter(jnp.asarray(2.0)), FixedParameter(jnp.asarray(1.0))),
        _project,
    )

    @eqx.filter_jit
    def step(
        wrapper: EligibilityTraining[_Map, JaxRealArray, JaxRealArray, JaxRealArray],
        trace: _Map,
        x: JaxRealArray,
    ) -> tuple[JaxRealArray, _Map]:
        observation, pullback = wrapper.infer(x)
        trace = wrapper.accumulate(trace, pullback, jnp.asarray(1.0), decay=jnp.asarray(0.5))
        return observation, trace

    zero = training.zero_trace()
    observation, first_trace = step(training, zero, jnp.asarray(4.0))
    _, second_trace = step(training, first_trace, jnp.asarray(3.0))
    _, fresh_trace = step(training, zero, jnp.asarray(3.0))

    assert jnp.allclose(observation, 9)
    assert jnp.allclose(first_trace.weight.value, 4)
    assert jnp.allclose(second_trace.weight.value, 5)
    assert jnp.allclose(fresh_trace.weight.value, 3)

    def loss(weight: JaxRealArray, factor: JaxRealArray) -> JaxRealArray:
        wrapper = eqx.tree_at(lambda box: box.model.weight.value, training, weight)
        return wrapper.get_loss(second_trace, factor)

    weight_grad, factor_grad = jax.grad(loss, argnums=(0, 1))(
        training.model.weight.value, jnp.asarray(2.0)
    )
    assert jnp.allclose(weight_grad, -10)
    assert factor_grad == 0
