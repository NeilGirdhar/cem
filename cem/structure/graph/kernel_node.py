from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Self, override

import equinox as eqx
from tjax import RngStream, frozendict
from tjax.dataclasses import field

from .model import Model
from .node import Node, NodeConfiguration, NodeInferenceResult


class Binding(eqx.Module):
    source_node: str = field(static=True)
    source_field: str = field(static=True)

    def fetch(self, model: Model, state: eqx.nn.State) -> object:
        node = model.get_node(self.source_node)
        return node.get_output(node.read_output_from_state(state), self.source_field)


class NodeWithBindings[ConfigurationT: NodeConfiguration = NodeConfiguration](Node[ConfigurationT]):
    """Node that reads inputs from peer nodes via named bindings.

    Provides :meth:`resolve_bindings` (static constructor helper) and
    :meth:`gather_bound_inputs` (runtime fetch).  Subclasses supply the ``infer``
    implementation that uses these inputs.
    """

    _bindings: Mapping[str, tuple[Binding, ...]] = eqx.field(kw_only=True)

    @staticmethod
    def resolve_bindings(
        bindings: Mapping[str, Sequence[tuple[str, str]]],
    ) -> frozendict[str, tuple[Binding, ...]]:
        return frozendict(
            {
                field_name: tuple(
                    Binding(source_node=source_node, source_field=source_field)
                    for source_node, source_field in sources
                )
                for field_name, sources in bindings.items()
            }
        )

    def gather_bound_inputs(self, model: Model, state: eqx.nn.State) -> Mapping[str, list[object]]:
        return {
            name: [binding.fetch(model, state) for binding in bindings]
            for name, bindings in self._bindings.items()
        }


class Kernel[ConfigurationT: NodeConfiguration](eqx.Module):
    def infer(
        self,
        *,
        streams: Mapping[str, RngStream],
        inference: bool,
        inputs: Mapping[str, list[object]],
    ) -> ConfigurationT:
        raise NotImplementedError

    def zero_configuration(self) -> ConfigurationT:
        """Return a placeholder configuration matching this node's output shape."""
        raise NotImplementedError

    def get_output(self, node_configuration: NodeConfiguration, field_name: str) -> object:
        """Extract one named externally visible output field from the node configuration."""
        msg = f"{type(self).__name__} does not expose output field {field_name!r}"
        raise ValueError(msg)


class KernelNode[ConfigurationT: NodeConfiguration = NodeConfiguration](
    NodeWithBindings[ConfigurationT]
):
    """Graph node implementation backed by a Kernel and binding-based inputs."""

    kernel: Kernel[ConfigurationT] = eqx.field(kw_only=True)

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
        return cls(
            name=name,
            kernel=kernel,
            _output_state_index=eqx.nn.StateIndex(kernel.zero_configuration()),
            _bindings=cls.resolve_bindings(bindings),
        )

    def infer(
        self,
        model: Model,
        streams: Mapping[str, RngStream],
        state: eqx.nn.State,
        *,
        inference: bool,
    ) -> NodeInferenceResult[ConfigurationT]:
        result = self.kernel.infer(
            streams=streams,
            inference=inference,
            inputs=self.gather_bound_inputs(model, state),
        )
        return NodeInferenceResult(result, state)

    @override
    def get_output(self, node_configuration: NodeConfiguration, field_name: str) -> object:
        """Extract one named externally visible output field from the node configuration."""
        return self.kernel.get_output(node_configuration, field_name)
