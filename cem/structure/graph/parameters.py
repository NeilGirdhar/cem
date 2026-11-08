"""Parameter wrappers that let optimizers distinguish model weights from fixed values."""

from collections.abc import Callable, Sequence
from functools import partial
from typing import Any, cast

import equinox as eqx
import jax.numpy as jnp
from jax import tree


class Parameter[A](eqx.Module):
    """A typed wrapper that makes a value a leaf for model partitioning."""

    value: A


class LearnableParameter[A](Parameter[A]):
    """A model parameter updated during training."""


type LearnableParameterFactory[A] = Callable[[A], LearnableParameter[A]]


class SummedGradientParameter[A](LearnableParameter[A]):
    """A parameter trained on the sum of the gradient contributions that reach it."""


class LossBalancedParameter[A](LearnableParameter[A]):
    """A parameter whose loss gradients are to be balanced before the update.

    Each balancing group uses its own subclass, and the group's routing entry names its losses.
    """


class FixedParameter[A](Parameter[A]):
    """A model parameter held fixed during training, such as a dropout rate."""


def is_parameter(x: object, /) -> bool:
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
    """Apply a replacement function to every parameter wrapper in a pytree."""
    return cast("T", eqx.tree_at(_get_parameters, x, replace_fn=partial(_apply_to_parameter, f)))


def _get_parameters(m: object, /) -> Sequence[Any]:
    return tree.leaves(m, is_leaf=is_parameter)


def _apply_to_parameter(f: Callable[[Parameter[Any]], Parameter[Any]], x: object, /) -> object:
    if isinstance(x, Parameter):
        return f(x)
    return x
