"""The model interface and its step result for training and inference."""

from collections.abc import Mapping
from typing import Any

import equinox as eqx
from tjax import JaxAbstractClass, JaxArray, RngStream

from .node import NodeConfiguration


class ModelResult(eqx.Module):
    """The loss, node outputs, and next recurrent state from one model step."""

    loss: JaxArray
    configurations: frozendict[str, NodeConfiguration]
    state: Any


class Model(eqx.Module, JaxAbstractClass):
    """A model that maps problem observations to node outputs and a training loss."""

    def initial_state(self) -> object:
        """Return recurrent state for a new episode.

        The default empty mapping suits feedforward models. Recurrent models return a
        pytree of the arrays they carry between time steps.
        """
        return frozendict()

    def infer(
        self,
        observation: object,
        state: object,
        *,
        streams: Mapping[str, RngStream],
        inference: bool,
    ) -> ModelResult:
        """Evaluate one observation and return the model's loss, node outputs, and next state.

        Use the initial state on the first step and the previous result's state thereafter.
        Named random streams supply stochastic operations. Set `inference` to true to use
        evaluation behaviour, such as disabling dropout.
        """
        raise NotImplementedError
