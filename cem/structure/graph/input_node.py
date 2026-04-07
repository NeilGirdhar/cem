from __future__ import annotations

from collections.abc import Mapping
from typing import Self, override

import equinox as eqx
from tjax import RngStream, frozendict

from .node import NodeBase, NodeConfiguration, NodeInferenceResult


class InputNodeConfiguration(NodeConfiguration):
    """Holds the environment-provided input values for an input node."""

    values: frozendict[str, object]


class InputNode(NodeBase[InputNodeConfiguration]):
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
        field_defaults: Mapping[str, object],
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

    def infer(
        self,
        model: object,
        streams: Mapping[str, RngStream],
        state: eqx.nn.State,
        *,
        use_signal_noise: bool,
        return_samples: bool,
    ) -> NodeInferenceResult[InputNodeConfiguration]:
        del model, streams, use_signal_noise, return_samples
        values = frozendict({f: state.get(idx) for f, idx in self._state_indices.items()})
        return NodeInferenceResult(InputNodeConfiguration(values=values), state)

    def set_input(self, field_name: str, new_value: object, state: eqx.nn.State) -> eqx.nn.State:
        """Write one environment-provided value into this node's state."""
        return state.set(self._state_indices[field_name], new_value)

    @override
    def get_output(self, node_configuration: NodeConfiguration, field_name: str) -> object:
        assert isinstance(node_configuration, InputNodeConfiguration)
        return node_configuration.values[field_name]
