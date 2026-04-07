from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Generic, Self, TypeVar, override

import equinox as eqx
import jax.numpy as jnp
from tjax import JaxArray, RngStream
from tjax.dataclasses import dataclass, field

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


class NodeBase[ConfigurationT: NodeConfiguration = NodeConfiguration](eqx.Module):
    name: str = eqx.field(static=True)
    _output_state_index: eqx.nn.StateIndex

    def infer(
        self,
        model: Model,
        streams: Mapping[str, RngStream],
        state: eqx.nn.State,
        *,
        return_samples: bool,
    ) -> NodeInferenceResult[ConfigurationT]:
        raise NotImplementedError

    def write_output_to_state(
        self, configuration: NodeConfiguration, state: eqx.nn.State
    ) -> eqx.nn.State:
        """Write this node's output configuration into state."""
        return state.set(self._output_state_index, configuration)

    def read_output_from_state(self, state: eqx.nn.State) -> NodeConfiguration:
        """Read this node's most recent output configuration from state."""
        return state.get(self._output_state_index)

    def get_output(self, node_configuration: NodeConfiguration, field_name: str) -> object:
        raise NotImplementedError


class Binding(eqx.Module):
    source_node: str = field(static=True)
    source_field: str = field(static=True)

    def fetch(self, model: Model, state: eqx.nn.State) -> object:
        node = model.get_node(self.source_node)
        return node.get_output(node.read_output_from_state(state), self.source_field)


class Kernel[ConfigurationT: NodeConfiguration](eqx.Module):
    def infer(self, *, streams: Mapping[str, RngStream], **kwargs: list[object]) -> ConfigurationT:
        raise NotImplementedError

    def zero_configuration(self) -> ConfigurationT:
        """Return a placeholder configuration matching this node's output shape."""
        raise NotImplementedError

    def get_output(self, node_configuration: NodeConfiguration, field_name: str) -> object:
        """Extract one named externally visible output field from the node configuration."""
        msg = f"{type(self).__name__} does not expose output field {field_name!r}"
        raise ValueError(msg)


class Node[ConfigurationT: NodeConfiguration = NodeConfiguration](NodeBase[ConfigurationT]):
    """A node is the fundamental computational unit of a model."""

    kernel: Kernel[ConfigurationT]
    _bindings: Mapping[str, tuple[Binding, ...]]

    @classmethod
    def create(
        cls,
        *,
        name: str,
        kernel: Kernel[ConfigurationT],
        bindings: Mapping[str, Sequence[tuple[str, str]]],
        streams: Mapping[str, RngStream],
    ) -> Self:
        del streams
        resolved = {
            field_name: tuple(
                Binding(source_node=source_node, source_field=source_field)
                for source_node, source_field in sources
            )
            for field_name, sources in bindings.items()
        }
        return cls(
            name=name,
            kernel=kernel,
            _output_state_index=eqx.nn.StateIndex(kernel.zero_configuration()),
            _bindings=resolved,
        )

    def infer(
        self,
        model: Model,
        streams: Mapping[str, RngStream],
        state: eqx.nn.State,
        *,
        return_samples: bool,
    ) -> NodeInferenceResult[ConfigurationT]:
        del return_samples
        result = self.kernel.infer(
            streams=streams,
            **{
                name: [binding.fetch(model, state) for binding in bindings]
                for name, bindings in self._bindings.items()
            },
        )
        return NodeInferenceResult(result, state)

    @override
    def get_output(self, node_configuration: NodeConfiguration, field_name: str) -> object:
        """Extract one named externally visible output field from the node configuration."""
        return self.kernel.get_output(node_configuration, field_name)
