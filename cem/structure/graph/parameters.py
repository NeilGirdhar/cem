from collections.abc import Callable, Sequence
from functools import partial
from typing import Any, cast

import equinox as eqx
import jax.numpy as jnp
from jax import tree


class Parameter[A](eqx.Module):
    """A parameter that is either constant or variable (learned during training)."""

    value: A


class LearnableParameter[A](Parameter[A]):
    """A parameter that is updated by training.

    E.g., weights, biases.
    """


class MetaParameter[A](Parameter[A]):
    """A parameter updated by a meta-learning objective, typically at a lower learning rate.

    For example, weights learned to equalize gradient contributions.
    """


class FixedParameter[A](Parameter[A]):
    """A parameter that remains unchanged throughout model training.

    E.g., regularization strength, dropout rate, architectural constants.
    """


def is_parameter(x: object, /) -> bool:
    """Return True if x is a Parameter leaf."""
    return isinstance(x, Parameter)


def count_real_learnable_parameters(x: object, /) -> int:
    """Count trainable real scalar degrees of freedom in a pytree.

    Each complex array element contributes two real degrees of freedom.
    """
    count = 0
    for leaf in tree.leaves(x, is_leaf=is_parameter):
        if not isinstance(leaf, LearnableParameter):
            continue
        value = leaf.value
        multiplier = 2 if jnp.iscomplexobj(value) else 1
        count += multiplier * value.size
    return count


def apply_to_parameters[T](f: Callable[[Parameter[Any]], Parameter[Any]], x: T) -> T:
    """Replace one type of parameter with another throughout a pytree."""
    return cast("T", eqx.tree_at(_get_parameters, x, replace_fn=partial(_apply_to_parameter, f)))


def _get_parameters(m: object, /) -> Sequence[Any]:
    return tree.leaves(m, is_leaf=is_parameter)


def _apply_to_parameter(f: Callable[[Parameter[Any]], Parameter[Any]], x: object, /) -> object:
    if isinstance(x, Parameter):
        return f(x)
    return x
