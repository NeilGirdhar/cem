from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Self, override

import equinox as eqx
from tjax import RngStream, frozendict

from .node import Node, NodeConfiguration, NodeInferenceResult

if TYPE_CHECKING:
    from .model import Model


class InputNodeConfiguration[ValueT](NodeConfiguration):
    """Holds the environment-provided input values for an input node."""

    values: frozendict[str, ValueT]


class InputNode[ValueT](Node[InputNodeConfiguration[ValueT]]):
    """Dummy node that holds environment-provided inputs as named state slots.

    Other nodes read from it via Binding(source_node=<name>, source_field=<field_name>).
    """

    _state_indices: frozendict[str, eqx.nn.StateIndex] = eqx.field(static=True)

    @property
    def field_names(self) -> tuple[str, ...]:
        return tuple(self._state_indices)

    @classmethod
    def create(
        cls,
        name: str,
        field_defaults: Mapping[str, ValueT],
    ) -> Self:
        state_indices = frozendict(
            {field: eqx.nn.StateIndex(v) for field, v in field_defaults.items()}
        )
        zero_config = InputNodeConfiguration(values=frozendict(field_defaults))
        return cls(
            name=name,
            _state_indices=state_indices,
            _output_state_index=eqx.nn.StateIndex(zero_config),
        )

    def gather_local_inputs(self, state: eqx.nn.State) -> Mapping[str, ValueT]:
        return {f: state.get(idx) for f, idx in self._state_indices.items()}

    def infer(
        self,
        model: Model,
        streams: Mapping[str, RngStream],
        state: eqx.nn.State,
        *,
        inference: bool,
    ) -> NodeInferenceResult[InputNodeConfiguration[ValueT]]:
        """Read all field values from state and package them as an InputNodeConfiguration.

        The model, streams, and inference arguments are unused; input nodes
        have no computation — they simply expose values written by :meth:`set_input`.
        """
        del model, streams, inference
        values = frozendict(self.gather_local_inputs(state))
        return NodeInferenceResult(InputNodeConfiguration(values=values), state)

    def set_input(self, field_name: str, new_value: ValueT, state: eqx.nn.State) -> eqx.nn.State:
        """Write one environment-provided value into this node's state."""
        return state.set(self._state_indices[field_name], new_value)

    @override
    def get_output(self, node_configuration: NodeConfiguration, field_name: str) -> ValueT:
        assert isinstance(node_configuration, InputNodeConfiguration)
        return node_configuration.values[field_name]
