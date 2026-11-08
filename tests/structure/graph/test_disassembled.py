"""Check parameter partitioning and the routing of loss gradients."""

import equinox as eqx
import jax.numpy as jnp
import pytest
from jax import tree
from jax.experimental import checkify
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
from cem.structure.graph.losses import LossGradients


class _Model(Model):
    prediction: LossBalancedParameter[JaxRealArray]
    auxiliary: SummedGradientParameter[JaxRealArray]


def test_parameter_type_partition_round_trip_preserves_type() -> None:
    parameter_type = ParameterType(SummedGradientParameter)
    extracted, remainder = eqx.partition(parameter_type, lambda x: isinstance(x, type))
    round_tripped = eqx.combine(extracted, remainder)

    assert tree.leaves(extracted) == []
    assert round_tripped.t is SummedGradientParameter


def test_multi_loss_group_receives_separate_loss_gradients() -> None:
    prediction_type = ParameterType(LossBalancedParameter)
    auxiliary_type = ParameterType(SummedGradientParameter)
    loss_ids = ("reference", "auxiliary")
    transform = DisGradientTransformation(
        [
            (prediction_type, MTAdam[Model](0.5, loss_ids=loss_ids)),
            (auxiliary_type, SGD[Model](0.5)),
        ]
    )
    parameter_types = (prediction_type, auxiliary_type)

    def partition(prediction: float, auxiliary: float) -> DisModel:
        return DisModel.create(
            _Model(
                LossBalancedParameter(jnp.asarray(prediction)),
                SummedGradientParameter(jnp.asarray(auxiliary)),
            ),
            parameter_types,
        )

    model = partition(0.0, 0.0)
    gradients = LossGradients(
        {"auxiliary": partition(3.0, 8.0), "reference": partition(1.0, 4.0)},
        partition(0.0, 0.0),
    )
    checked = checkify.checkify(transform.update, errors=checkify.user_checks)
    error, (delta, next_state) = eqx.filter_jit(checked)(gradients, transform.init(model), model)
    error.throw()
    update = delta.assembled()
    assert isinstance(update, _Model)
    assert jnp.isclose(update.prediction.value, -1.0)
    assert jnp.isclose(update.auxiliary.value, -6.0)
    balanced_state = dict(next_state)[prediction_type]
    assert isinstance(balanced_state, MTAdamState)
    assert balanced_state.count == 1
    assert jnp.allclose(
        jnp.stack([tree.leaves(balanced_state.gradient_norms[name])[0] for name in loss_ids]),
        jnp.asarray([1.0, 3.0]),
    )


class _OtherBalancedParameter(LossBalancedParameter[JaxRealArray]):
    pass


class _GroupedModel(_Model):
    other: _OtherBalancedParameter


def _grouped_transform() -> DisGradientTransformation:
    return DisGradientTransformation(
        [
            (
                ParameterType(_OtherBalancedParameter),
                MTAdam[Model](0.5, loss_ids=("first", "second")),
            ),
            (ParameterType(LossBalancedParameter), MTAdam[Model](0.5, loss_ids=("third",))),
            (ParameterType(SummedGradientParameter), SGD[Model](0.5)),
        ]
    )


def _grouped_partition(
    transform: DisGradientTransformation, *, other: float, prediction: float, auxiliary: float
) -> DisModel:
    return DisModel.create(
        _GroupedModel(
            prediction=LossBalancedParameter(jnp.asarray(prediction)),
            auxiliary=SummedGradientParameter(jnp.asarray(auxiliary)),
            other=_OtherBalancedParameter(jnp.asarray(other)),
        ),
        tuple(transform.learnable_parameter_types()),
    )


def _checked_update(
    transform: DisGradientTransformation, gradients: LossGradients[DisModel], model: DisModel
) -> DisModel:
    checked = checkify.checkify(transform.update, errors=checkify.user_checks)
    error, (delta, _) = eqx.filter_jit(checked)(gradients, transform.init(model), model)
    error.throw()
    return delta


def test_balancing_groups_receive_only_their_losses() -> None:
    transform = _grouped_transform()
    assert transform.requested_loss_ids() == ("first", "second", "third")
    model = _grouped_partition(transform, other=0.0, prediction=0.0, auxiliary=0.0)
    gradients = LossGradients(
        {
            "first": _grouped_partition(transform, other=1.0, prediction=0.0, auxiliary=1.0),
            "second": _grouped_partition(transform, other=3.0, prediction=0.0, auxiliary=3.0),
            "third": _grouped_partition(transform, other=0.0, prediction=4.0, auxiliary=4.0),
        },
        _grouped_partition(transform, other=0.0, prediction=0.0, auxiliary=5.0),
    )
    update = _checked_update(transform, gradients, model).assembled()
    assert isinstance(update, _GroupedModel)
    assert jnp.isclose(update.other.value, -1.0)
    assert jnp.isclose(update.prediction.value, -0.5)
    assert jnp.isclose(update.auxiliary.value, -6.5)


@pytest.mark.parametrize(
    ("third_leak", "fixed_weight_leak", "source"),
    [(2.0, 0.0, "the 'third' loss"), (0.0, 2.0, "the fixed-weight loss")],
)
def test_balancing_group_rejects_gradient_from_other_losses(
    third_leak: float, fixed_weight_leak: float, source: str
) -> None:
    transform = _grouped_transform()
    model = _grouped_partition(transform, other=0.0, prediction=0.0, auxiliary=0.0)
    gradients = LossGradients(
        {
            "first": _grouped_partition(transform, other=1.0, prediction=0.0, auxiliary=0.0),
            "second": _grouped_partition(transform, other=3.0, prediction=0.0, auxiliary=0.0),
            "third": _grouped_partition(transform, other=third_leak, prediction=4.0, auxiliary=0.0),
        },
        _grouped_partition(transform, other=fixed_weight_leak, prediction=0.0, auxiliary=0.0),
    )
    with pytest.raises(checkify.JaxRuntimeError, match=f"received gradient from {source}"):
        _checked_update(transform, gradients, model)
