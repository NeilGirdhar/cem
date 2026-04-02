from __future__ import annotations

from collections.abc import Mapping
from typing import Self, override

import equinox as eqx
from tjax import RngStream

from .node import NodeBase, NodeConfiguration, NodeInferenceResult


class InputNodeConfiguration(NodeConfiguration):
    """Holds the environment-provided input values for the 'input' node."""

    values: tuple[object, ...]  # one value per field, in field_names order


class InputNode(NodeBase[InputNodeConfiguration]):
    """Dummy node that holds environment-provided inputs as named state slots.

    Created automatically by Model and named 'input'. Other nodes read from it
    via Binding(source_node='input', source_field=<field_name>).
    """

    field_names: tuple[str, ...] = eqx.field(static=True)
    _state_indices: tuple[eqx.nn.StateIndex, ...]

    @classmethod
    def create(
        cls,
        field_defaults: Mapping[str, object],
    ) -> Self:
        field_names = tuple(field_defaults.keys())
        defaults = tuple(field_defaults.values())
        zero_config = InputNodeConfiguration(values=defaults)
        return cls(
            name="input",
            field_names=field_names,
            _state_indices=tuple(eqx.nn.StateIndex(v) for v in defaults),
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
        values = tuple(state.get(idx) for idx in self._state_indices)
        return NodeInferenceResult(InputNodeConfiguration(values=values), state)

    def set_input(self, field_name: str, new_value: object, state: eqx.nn.State) -> eqx.nn.State:
        """Write one environment-provided value into this node's state."""
        idx = self.field_names.index(field_name)
        return state.set(self._state_indices[idx], new_value)

    @override
    def get_output(self, node_configuration: NodeConfiguration, field_name: str) -> object:
        assert isinstance(node_configuration, InputNodeConfiguration)
        idx = self.field_names.index(field_name)
        return node_configuration.values[idx]
