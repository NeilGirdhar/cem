"""Partition models by parameter type so each group can use its own gradient transformation."""

from collections.abc import Iterable, Sequence
from functools import reduce
from typing import Any, Self

import equinox as eqx
from jax import tree
from tjax.gradient import GradientState, GradientTransformation

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


class DisGradientTransformation(eqx.Module):
    """Apply separate gradient transformations to the parameter groups of a model.

    The first matching parameter type determines the transform. An entry with no
    transform keeps its parameters fixed.
    """

    gradient_transformations: list[tuple[ParameterType, GradientTransformation[Any, Model] | None]]

    def learnable_parameter_types(self) -> Iterable[ParameterType]:
        return (p for p, g in self.gradient_transformations if g is not None)

    def is_learnable(self, x: Parameter[Any]) -> bool:
        assert isinstance(x, Parameter)
        for parameter_type, gradient_transformation in self.gradient_transformations:
            if isinstance(x, parameter_type.t):
                return gradient_transformation is not None
        msg = f"Parameter of type {type(x).__name__} has no corresponding gradient transformation"
        raise TypeError(msg)

    def init(self, model: DisModel) -> DisGradientState:
        """Initialize optimizer state for groups that have a gradient transform."""
        dis_model = dict(model.parameters)
        assert {parameter_type.t for parameter_type in dis_model} <= {
            parameter_type.t for parameter_type, _ in self.gradient_transformations
        }
        gradient_states: list[tuple[ParameterType, GradientState]] = []
        for parameter_type, gradient_transformation in self.gradient_transformations:
            if gradient_transformation is None:
                continue
            parameters = dis_model[parameter_type]
            gradient_states.append((parameter_type, gradient_transformation.init(parameters)))
        return gradient_states

    def update(
        self, parameters_bar: DisModel, gradient_states: DisGradientState, model: DisModel
    ) -> tuple[DisModel, DisGradientState]:
        """Transform gradients for the trainable groups and advance their optimizer states."""
        dis_parameters_bar = dict(parameters_bar.parameters)
        dis_gs = dict(gradient_states)
        dis_model = dict(model.parameters)
        new_parameters_bar: dict[ParameterType, Model] = {}
        new_dis_gs: dict[ParameterType, GradientState] = {}
        for parameter_type, gradient_transformation in self.gradient_transformations:
            if gradient_transformation is None:
                continue
            new_parameters_bar[parameter_type], new_dis_gs[parameter_type] = (
                gradient_transformation.update(
                    dis_parameters_bar[parameter_type],
                    dis_gs[parameter_type],
                    dis_model[parameter_type],
                )
            )
        return DisModel(list(new_parameters_bar.items())), list(new_dis_gs.items())
