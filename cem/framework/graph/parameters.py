"""Parameter wrappers that let optimizers distinguish model weights from fixed values."""

import equinox as eqx
import jax.numpy as jnp
from jax import tree


class Parameter[A](eqx.Module):
    """A typed wrapper that makes a value a leaf for model partitioning."""

    value: A


class LearnableParameter[A](Parameter[A]):
    """A model parameter updated during training."""


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
