"""The model interface and its step result for training and inference."""

from collections.abc import Mapping
from typing import Any

import equinox as eqx
from tjax import JaxAbstractClass, JaxArray, RngStream

from .node import NodeConfiguration


class ModelResult(eqx.Module):
    """The outcome of evaluating a model on one observation.

    The fixed-weight loss is the part of the training objective that enters with unit weight.
    The balanced losses are named objectives whose gradients a multi-loss transform can balance for
    the parameter groups they reach. A model with a single scalar loss leaves them empty.
    """

    configurations: frozendict[str, NodeConfiguration]
    state: Any
    fixed_weight_loss: JaxArray
    balanced_losses: frozendict[str, JaxArray] = frozendict()

    @property
    def loss(self) -> JaxArray:
        """The training objective: the sum of the fixed-weight and balanced losses."""
        return sum(self.balanced_losses.values(), self.fixed_weight_loss)


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
