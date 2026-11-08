"""Partition models by parameter type so each group can use its own gradient transformation."""

from collections.abc import Iterable, Sequence
from functools import partial, reduce
from typing import Any, Self

import equinox as eqx
import jax.numpy as jnp
from jax import tree
from jax.experimental import checkify
from tjax import JaxArray, JaxBooleanArray, dynamic_tree_all
from tjax.gradient import GradientState, GradientTransformation, MultiLossGradientTransformation

from .losses import LossGradients
from .model import Model
from .parameters import Parameter, is_parameter


class ParameterType(eqx.Module):
    """A parameter class used to select leaves of a model pytree."""

    t: type[Parameter[Any]] = eqx.field(static=True)


type DisGradientState = list[tuple[ParameterType, GradientState]]


def verify_model_has_no_free_parameters(model: Model, *, should_be_empty: bool = False) -> None:
    """Reject model leaves outside parameter wrappers.

    When `should_be_empty` is true, reject wrapped parameters too.
    """
    for path, value in tree.leaves_with_path(model, is_leaf=is_parameter):
        if not should_be_empty and isinstance(value, Parameter):
            continue
        msg = (
            f"Jax tree leaf found in the model at path {path} that is not enclosed in a Parameter."
        )
        raise RuntimeError(msg)


class DisModel(eqx.Module):
    """A model partitioned into disjoint pytrees by parameter type."""

    parameters: list[tuple[ParameterType, Model]]

    @classmethod
    def create(cls, model: Model, parameter_types: Sequence[ParameterType]) -> Self:
        """Partition the model in the given type order.

        The first matching type takes each parameter. Every model leaf must be covered.
        """
        parameters: dict[ParameterType, Model] = {}
        for parameter_type in parameter_types:

            def belongs_to_type(
                x: object, /, *, parameter_type: ParameterType = parameter_type
            ) -> bool:
                return isinstance(x, parameter_type.t)

            parameters[parameter_type], model = eqx.partition(
                model, belongs_to_type, is_leaf=is_parameter
            )
        verify_model_has_no_free_parameters(model, should_be_empty=True)
        return cls(list(parameters.items()))

    def assembled(self) -> Model:
        """Recombine the parameter partitions into a complete model."""

        def f(x: Model, y: Model, /) -> Model:
            return eqx.combine(x, y, is_leaf=is_parameter)

        return reduce(f, (m for _, m in self.parameters))


def _is_zero_where_finite(x: JaxArray, /) -> JaxBooleanArray:
    return jnp.all((x == 0) | ~jnp.isfinite(x))


def _sum_gradients(gradients: Iterable[Model]) -> Model:
    return reduce(partial(tree.map, jnp.add), gradients)


type _ParameterTransform = (
    GradientTransformation[Any, Model] | MultiLossGradientTransformation[Any, Model]
)


def _update_group(
    transform: _ParameterTransform,
    gradients: LossGradients[Model],
    state: GradientState,
    parameters: Model,
) -> tuple[Model, GradientState]:
    """Apply one group's transform to the gradients that reach the group.

    A multi-loss transform receives the gradient of each of its losses after a `checkify` check
    that no other loss reaches the group, so a staged caller must functionalize the check with
    `checkify.checkify`. Non-finite gradients pass the check and are left to the trainer's
    finiteness diagnostics. Any other transform receives the sum of all the gradients.
    """
    if not isinstance(transform, MultiLossGradientTransformation):
        # TODO: Once every parameter that a balanced loss reaches is a balanced parameter, block
        # the balanced gradients here with a `checkify` check that they are zero, instead of
        # summing them.
        total = _sum_gradients(
            [*gradients.balanced_gradients.values(), gradients.fixed_weight_gradient]
        )
        return transform.update(total, state, parameters)
    selected = {name: gradients.balanced_gradients[name] for name in transform.loss_ids}
    unbalanced = {
        **{
            f"the {name!r} loss": gradient
            for name, gradient in gradients.balanced_gradients.items()
            if name not in selected
        },
        "the fixed-weight loss": gradients.fixed_weight_gradient,
    }
    for source, gradient in unbalanced.items():
        checkify.check(
            dynamic_tree_all(tree.map(_is_zero_where_finite, gradient)),
            f"Parameters balanced over {transform.loss_ids} received gradient from {source}",
        )
    return transform.update(selected, state, parameters)


class DisGradientTransformation(eqx.Module):
    """Apply separate gradient transformations to the parameter groups of a model.

    The first matching parameter type determines the transform. An entry with no transform keeps
    its parameters fixed. A multi-loss transform balances the losses it names, and its group must
    receive gradient from those losses alone.
    """

    gradient_transformations: list[tuple[ParameterType, _ParameterTransform | None]]

    def learnable_parameter_types(self) -> Iterable[ParameterType]:
        return (p for p, g in self.gradient_transformations if g is not None)

    def requested_loss_ids(self) -> tuple[str, ...]:
        """Return the distinct loss IDs that the multi-loss transforms balance."""
        return tuple(
            dict.fromkeys(
                name
                for _, transform in self.gradient_transformations
                if isinstance(transform, MultiLossGradientTransformation)
                for name in transform.loss_ids
            )
        )

    def is_learnable(self, x: Parameter[Any]) -> bool:
        assert isinstance(x, Parameter)
        for parameter_type, gradient_transformation in self.gradient_transformations:
            if isinstance(x, parameter_type.t):
                return gradient_transformation is not None
        msg = f"Parameter of type {type(x).__name__} has no corresponding gradient transformation"
        raise TypeError(msg)

    def init(self, model: DisModel) -> DisGradientState:
        """Initialize the optimizer state for each parameter group."""
        dis_model = dict(model.parameters)
        assert {parameter_type.t for parameter_type in dis_model} <= {
            parameter_type.t for parameter_type, _ in self.gradient_transformations
        }
        return [
            (parameter_type, gradient_transformation.init(dis_model[parameter_type]))
            for parameter_type, gradient_transformation in self.gradient_transformations
            if gradient_transformation is not None
        ]

    def update(
        self,
        parameters_bar: LossGradients[DisModel],
        gradient_states: DisGradientState,
        model: DisModel,
    ) -> tuple[DisModel, DisGradientState]:
        """Route each parameter group's gradients to its transform."""
        balanced_gradients = {
            name: dict(gradient.parameters)
            for name, gradient in parameters_bar.balanced_gradients.items()
        }
        fixed_weight_gradients = dict(parameters_bar.fixed_weight_gradient.parameters)
        dis_gs = dict(gradient_states)
        dis_model = dict(model.parameters)
        new_parameters_bar: dict[ParameterType, Model] = {}
        new_dis_gs: dict[ParameterType, GradientState] = {}
        for parameter_type, gradient_transformation in self.gradient_transformations:
            if gradient_transformation is None:
                continue
            group_gradients = LossGradients(
                {name: gradient[parameter_type] for name, gradient in balanced_gradients.items()},
                fixed_weight_gradients[parameter_type],
            )
            update, state = _update_group(
                gradient_transformation,
                group_gradients,
                dis_gs[parameter_type],
                dis_model[parameter_type],
            )
            new_parameters_bar[parameter_type] = update
            new_dis_gs[parameter_type] = state
        return DisModel(list(new_parameters_bar.items())), list(new_dis_gs.items())
