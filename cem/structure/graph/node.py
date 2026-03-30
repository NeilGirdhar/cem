from __future__ import annotations

from collections.abc import Mapping
from dataclasses import KW_ONLY
from typing import TYPE_CHECKING, Generic, TypeVar

import equinox as eqx
import jax.numpy as jnp
from tjax import JaxArray, RngStream
from tjax.dataclasses import dataclass

from .batch_loss import BatchLoss
from .module import Module

if TYPE_CHECKING:
    from .model import Model, ModelConfiguration


class NodeMemory(eqx.Module):
    """The memory of a Node.

    NodeMemory contains memory terms that are carried between node inference calls in a
    reinforcement learning trajectory.
    """


class NodeConfiguration(eqx.Module):
    """The configuration of a node.

    The NodeConfiguration is produced by node inference and returned in a reinforcement learning
    trajectory.
    """

    def total_loss(self) -> JaxArray:
        return jnp.zeros(())


_C_co = TypeVar("_C_co", covariant=True, bound=NodeConfiguration, default=NodeConfiguration)


@dataclass
class NodeInferenceResult(Generic[_C_co]):  # noqa: UP046
    configuration: _C_co
    state: eqx.nn.State
    batch_losses: tuple[BatchLoss, ...] = ()


class Node(Module):
    """A node is the fundamental computational unit of a model.

    It is defined in terms of its configuration type and loss type.
    """

    _: KW_ONLY
    name: str = eqx.field(static=True)

    def infer(
        self,
        model: Model,
        model_configuration: ModelConfiguration,
        streams: Mapping[str, RngStream],
        state: eqx.nn.State,
        *,
        use_signal_noise: bool,
        return_samples: bool,
    ) -> NodeInferenceResult:
        """Infer this node.

        Returns:
            loss: The loss scalar.  The backward pass will send a unit cotangent.
            configuration: The configuration, which allows other nodes to see the results earlier
                nodes in the graph.

            The backward pass can send cotangents through the configuration and the memory.
        """
        raise NotImplementedError

    def zero_configuration(self) -> NodeConfiguration:
        raise NotImplementedError

    def set_input(self, field_name: str, new_value: object, state: eqx.nn.State) -> eqx.nn.State:
        raise NotImplementedError

    def get_output(self, model_configuration: ModelConfiguration) -> object:
        raise NotImplementedError
