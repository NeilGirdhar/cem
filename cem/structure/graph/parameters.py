from __future__ import annotations

from collections.abc import Callable, Sequence
from functools import partial
from typing import Any, cast

import equinox as eqx
from jax import tree


class Parameter[A](eqx.Module):
    """A parameter that is either constant or variable (learned during training)."""

    value: A


class LearnableParameter[A](Parameter[A]):
    """A parameter that is updated by training.

    E.g., weights, biases.
    """


class FixedParameter[A](Parameter[A]):
    """A parameter that remains unchanged throughout model training.

    E.g., regularization strength, dropout rate, architectural constants.
    """


def is_parameter(x: object, /) -> bool:
    """Return True if x is a Parameter leaf."""
    return isinstance(x, Parameter)


def apply_to_parameters[T](f: Callable[[Parameter[Any]], Parameter[Any]], x: T) -> T:
    """Replace one type of parameter with another throughout a pytree."""
    return cast("T", eqx.tree_at(_get_parameters, x, replace_fn=partial(_apply_to_parameter, f)))


def _get_parameters(m: object, /) -> Sequence[Any]:
    return tree.leaves(m, is_leaf=is_parameter)


def _apply_to_parameter(f: Callable[[Parameter[Any]], Parameter[Any]], x: object, /) -> object:
    if isinstance(x, Parameter):
        return f(x)
    return x
