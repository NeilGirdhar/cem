"""Train a map on cotangents recorded when its outputs were produced."""

from collections.abc import Callable

import equinox as eqx
import jax
import jax.numpy as jnp
from jax import tree
from tjax import JaxRealArray, PyTree

from .parameters import LearnableParameter, is_parameter
from .surrogate import promoting_link_loss


class EligibilityTraining[M: eqx.Module, X, C: PyTree, Y: PyTree](eqx.Module):
    """Train a map on parameter cotangents recorded when it produced its outputs.

    `M` is the map type. Its floating-point or complex `LearnableParameter` leaves
    define the trace structure.
    `X` is the input type supplied to the map.
    `C` is the pytree of output coordinates used for differentiation. The pullback
    accepts a cotangent with the same structure.
    `Y` is the observation type returned by inference.

    The projection takes a map and an input and returns the coordinates followed by
    the observation. Each inference stream stores its own traces.
    """

    model: M
    project: Callable[[M, X], tuple[C, Y]] = eqx.field(static=True)

    def _partition(self) -> tuple[M, M]:
        return eqx.partition(
            self.model, lambda leaf: isinstance(leaf, LearnableParameter), is_leaf=is_parameter
        )

    def zero_trace(self) -> M:
        """Initialize a trace with one zero cotangent per trainable parameter leaf."""
        trainable, _ = self._partition()
        return tree.map(jnp.zeros_like, trainable)

    def infer(self, x: X) -> tuple[Y, Callable[[C], tuple[M]]]:
        """Return the observation and its pullback in the coordinates supplied by the projection.

        Use the pullback during the current step to record parameter cotangents.
        """
        trainable, fixed = self._partition()

        def project(parameters: M) -> tuple[C, Y]:
            return self.project(eqx.combine(parameters, fixed, is_leaf=is_parameter), x)

        _, pullback, observation = jax.vjp(project, trainable, has_aux=True)
        return observation, pullback

    def accumulate(
        self,
        trace: M,
        pullback: Callable[[C], tuple[M]],
        output_cotangent: C,
        *,
        decay: JaxRealArray,
    ) -> M:
        """Add the current parameter cotangent to the retained fraction of the trace."""
        (cotangent,) = pullback(output_cotangent)
        return tree.map(
            lambda old, new: jax.lax.stop_gradient(decay.astype(old.real.dtype) * old + new),
            trace,
            cotangent,
        )

    def get_loss(self, trace: M, factor: JaxRealArray) -> JaxRealArray:
        """Return a scalar loss whose descent update follows the scaled trace."""
        trainable, _ = self._partition()
        losses = tree.map(
            lambda parameter, cotangent: promoting_link_loss(parameter, factor * cotangent),
            trainable,
            trace,
        )
        return tree.reduce(jnp.add, losses, jnp.asarray(0.0))
