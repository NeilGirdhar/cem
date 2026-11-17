"""Train a map on cotangents recorded when its outputs were produced."""

from collections.abc import Callable

import equinox as eqx
import jax
import jax.numpy as jnp
from jax import tree
from tjax import JaxRealArray, PyTree, cotangent_surrogate_loss, scale_tree

from .parameters import LearnableParameter, is_parameter


class CotangentTraining[M: eqx.Module, X, C: PyTree, Y: PyTree](eqx.Module):
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

    def drift(self, pullback: Callable[[C], tuple[M]], output_cotangent: C) -> M:
        """Return the parameter cotangent that this step adds to the eligibility trace.

        The drift is the output cotangent pulled back through the map as it ran, held fixed so
        that a later goal can credit it. See `cotangent-training-link` in the thesis.
        """
        (cotangent,) = pullback(output_cotangent)
        return jax.lax.stop_gradient(cotangent)

    def get_loss(self, trace: M, factor: JaxRealArray) -> JaxRealArray:
        """Return a scalar loss whose descent update follows the scaled trace.

        The trace holds ascent cotangents, -dL/dX in the thesis's convention, so negate it to give
        tjax the loss derivative it expects. See `cotangent-training-link` in the thesis.
        """
        trainable, _ = self._partition()
        return cotangent_surrogate_loss(trainable, scale_tree(trace, scalar_scale=-factor))
