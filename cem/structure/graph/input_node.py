from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, Self, override

import equinox as eqx
import jax.numpy as jnp
from efax import Flattener, NaturalParametrization
from tjax import JaxRealArray, RngStream, frozendict

from .node import Node, NodeConfiguration, NodeInferenceResult

if TYPE_CHECKING:
    from .model import Model


class InputConfiguration[ValueT](NodeConfiguration):
    """Holds the environment-provided input values for an input node."""

    values: frozendict[str, ValueT]


class InputNode[ValueT](Node[InputConfiguration[ValueT]]):
    """Dummy node that holds environment-provided inputs as named state slots.

    Other nodes read from it via Binding(source_node=<name>, source_field=<field_name>).
    """

    _state_indices: frozendict[str, eqx.nn.StateIndex]

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
        zero_config = InputConfiguration(values=frozendict(field_defaults))
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
    ) -> NodeInferenceResult[InputConfiguration[ValueT]]:
        """Read all field values from state and package them as an InputConfiguration.

        The model, streams, and inference arguments are unused; input nodes
        have no computation — they simply expose values written by :meth:`set_input`.
        """
        del model, streams, inference
        values = frozendict(self.gather_local_inputs(state))
        return NodeInferenceResult(InputConfiguration(values=values), state)

    def set_input(self, field_name: str, new_value: ValueT, state: eqx.nn.State) -> eqx.nn.State:
        """Write one environment-provided value into this node's state."""
        return state.set(self._state_indices[field_name], new_value)

    @override
    def get_output(self, node_configuration: NodeConfiguration, field_name: str) -> ValueT:
        assert isinstance(node_configuration, InputConfiguration)
        return node_configuration.values[field_name]


class FlatEncodedInputNode[MessageT](InputNode[MessageT]):
    """InputNode base for nodes that store flat distribution encodings per field.

    Provides a :class:`efax.Flattener` per field (``mapped_to_plane=True``) and helpers
    for building state indices from flat distribution arrays.  Subclasses supply the
    ``infer`` override that converts the stored flat arrays to domain-specific messages.

    Attributes:
        _flatteners: One per field, used to unflatten stored flat arrays back to
            :class:`efax.NaturalParametrization` instances.
    """

    _flatteners: frozendict[str, Flattener[Any]]

    @staticmethod
    def _make_state_indices(
        flat_defaults: Mapping[str, JaxRealArray],
    ) -> frozendict[str, eqx.nn.StateIndex]:
        """Build a StateIndex per field from a mapping of default flat arrays."""
        return frozendict({field: eqx.nn.StateIndex(v) for field, v in flat_defaults.items()})

    @staticmethod
    def _prepare_flat_fields(
        field_defaults: Mapping[str, NaturalParametrization[Any, Any]],
    ) -> tuple[dict[str, Flattener[Any]], dict[str, JaxRealArray]]:
        """Flatten each prior distribution to build flatteners and flat default arrays.

        Args:
            field_defaults: Mapping from field name to prior distribution.

        Returns:
            A pair ``(flatteners, flat_defaults)`` where each value uses
            ``mapped_to_plane=True`` coordinates.
        """
        flatteners: dict[str, Flattener[Any]] = {}
        flat_defaults: dict[str, JaxRealArray] = {}
        for field_name, dist in field_defaults.items():
            flattener, flat = Flattener.flatten(dist, mapped_to_plane=True)
            flatteners[field_name] = flattener
            flat_defaults[field_name] = flat
        return flatteners, flat_defaults

    @staticmethod
    def _split_by_field_sizes(
        data: JaxRealArray,
        field_sizes: frozendict[str, int],
    ) -> dict[str, JaxRealArray]:
        """Split a concatenated flat array into per-field slices."""
        running, split_points = 0, []
        for s in list(field_sizes.values())[:-1]:
            running += s
            split_points.append(running)
        return dict(zip(field_sizes, jnp.split(data, split_points, axis=-1), strict=True))
