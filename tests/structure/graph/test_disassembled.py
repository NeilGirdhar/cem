"""Check parameter partitioning and the routing of loss gradients."""

import equinox as eqx
import jax.numpy as jnp
from jax import tree
from tjax import JaxRealArray
from tjax.gradient import SGD, MTAdam, MTAdamState

from cem.structure.graph import (
    DisGradientTransformation,
    DisModel,
    LossBalancedParameter,
    Model,
    ParameterType,
    SummedGradientParameter,
)


class _Model(Model):
    prediction: LossBalancedParameter[JaxRealArray]
    auxiliary: SummedGradientParameter[JaxRealArray]


def test_parameter_type_partition_round_trip_preserves_type() -> None:
    parameter_type = ParameterType(SummedGradientParameter)
    extracted, remainder = eqx.partition(parameter_type, lambda x: isinstance(x, type))
    round_tripped = eqx.combine(extracted, remainder)

    assert tree.leaves(extracted) == []
    assert round_tripped.t is SummedGradientParameter


def test_disassembled_transform_routes_separate_loss_gradients() -> None:
    prediction_type = ParameterType(LossBalancedParameter)
    auxiliary_type = ParameterType(SummedGradientParameter)
    transform = DisGradientTransformation(
        [(prediction_type, MTAdam[Model](0.5)), (auxiliary_type, SGD[Model](0.5))]
    )
    parameter_types = (prediction_type, auxiliary_type)
    loss_names = ("reference", "auxiliary")

    def partition(prediction: float, auxiliary: float) -> DisModel:
        return DisModel.create(
            _Model(
                LossBalancedParameter(jnp.asarray(prediction), loss_names=loss_names),
                SummedGradientParameter(jnp.asarray(auxiliary)),
            ),
            parameter_types,
        )

    model = partition(0.0, 0.0)
    gradients = {
        "auxiliary": partition(3.0, 8.0),
        "reference": partition(1.0, 4.0),
    }
    state = transform.init(model, loss_ids=loss_names)
    delta, next_state = eqx.filter_jit(transform.update)(gradients, state, model)
    update = delta.assembled()
    assert isinstance(update, _Model)
    assert jnp.isclose(update.prediction.value, -1.0)
    assert jnp.isclose(update.auxiliary.value, -6.0)
    balanced_state = dict(next_state)[prediction_type]
    assert isinstance(balanced_state, MTAdamState)
    assert balanced_state.loss_ids == loss_names
    assert balanced_state.count == 1
    assert jnp.allclose(
        jnp.stack([tree.leaves(balanced_state.gradient_norms[name])[0] for name in loss_names]),
        jnp.asarray([1.0, 3.0]),
    )
