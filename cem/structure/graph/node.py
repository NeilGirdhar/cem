from __future__ import annotations

from collections.abc import Mapping
from dataclasses import KW_ONLY
from typing import TYPE_CHECKING, Generic, TypeVar, override

import equinox as eqx
import jax.numpy as jnp
from tjax import JaxArray, RngStream
from tjax.dataclasses import dataclass

from .module import Module

if TYPE_CHECKING:
    from .model import Model


class NodeConfiguration(eqx.Module):
    """The configuration of a node.

    The NodeConfiguration is produced by node inference and returned in a reinforcement learning
    trajectory.
    """

    def total_loss(self) -> JaxArray:
        """Return this node's contribution to the scalar model loss."""
        return jnp.zeros(())


_C_co = TypeVar("_C_co", covariant=True, bound=NodeConfiguration, default=NodeConfiguration)


@dataclass
class NodeInferenceResult(Generic[_C_co]):  # noqa: UP046
    """The result of a single node inference step."""

    configuration: _C_co
    state: eqx.nn.State


class Node(Module):
    """A node is the fundamental computational unit of a model.

    It is defined in terms of its configuration type and loss type.
    """

    _: KW_ONLY
    name: str = eqx.field(static=True)
    _output_state_index: eqx.nn.StateIndex = eqx.field(init=False)

    @override
    def __post_init__(self, streams: Mapping[str, RngStream]) -> None:
        super().__post_init__(streams=streams)
        self._output_state_index = eqx.nn.StateIndex(self.zero_configuration())

    def infer(
        self,
        model: Model,
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
        """Return a placeholder configuration matching this node's output shape."""
        raise NotImplementedError

    def set_input(self, field_name: str, new_value: object, state: eqx.nn.State) -> eqx.nn.State:
        """Write one externally supplied input field into the node's state."""
        raise NotImplementedError

    def get_output(self, node_configuration: NodeConfiguration, field_name: str) -> object:
        """Extract one named externally visible output field from the model configuration."""
        msg = f"{type(self).__name__} does not expose output field {field_name!r}"
        raise ValueError(msg)

    def write_output_to_state(
        self, configuration: NodeConfiguration, state: eqx.nn.State
    ) -> eqx.nn.State:
        """Write this node's output configuration into state."""
        return state.set(self._output_state_index, configuration)

    def read_output_from_state(self, state: eqx.nn.State) -> NodeConfiguration:
        """Read this node's most recent output configuration from state."""
        return state.get(self._output_state_index)
