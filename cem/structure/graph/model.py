from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import InitVar
from functools import reduce
from operator import add
from typing import TYPE_CHECKING, Any, overload, override

import equinox as eqx
import jax.numpy as jnp
from tjax import JaxArray, RngStream

if TYPE_CHECKING:
    from cem.structure.problem.creator import ModelCreator

from .module import Module
from .node import Node, NodeConfiguration

ModelConfiguration = Mapping[str, NodeConfiguration]


class ModelInferenceResult(eqx.Module):
    loss: JaxArray
    state: eqx.nn.State


class Model(Module):
    """The entire model.

    The model is stored as a dictionary of nodes. Inference proceeds in alphabetical order
    by node name.
    """

    creator: InitVar[ModelCreator[Any]]
    _nodes: dict[str, Node] = eqx.field(static=True, init=False)
    _input_routing: tuple[tuple[str, tuple[str, str]], ...] = eqx.field(static=True, init=False)
    _output_routing: tuple[tuple[str, tuple[str, str]], ...] = eqx.field(static=True, init=False)

    @override
    def __post_init__(
        self,
        streams: Mapping[str, RngStream],
        creator: ModelCreator[Any],
    ) -> None:
        super().__post_init__(streams=streams)
        assert "example" in streams
        self._nodes = creator.create_model(streams=streams)
        self._input_routing = tuple(creator.input_routing().items())
        self._output_routing = tuple(creator.output_routing().items())

    # Accessing methods ----------------------------------------------------------------------------
    @overload
    def get_node(self, node_name: str) -> Node: ...

    @overload
    def get_node(self, node_name: str, node_type: None) -> Node: ...

    @overload
    def get_node[NT: Node](self, node_name: str, node_type: type[NT]) -> NT: ...

    def get_node(self, node_name: str, node_type: type[Node] | None = None) -> Node:
        if node_name not in self._nodes:
            msg = f"{node_name} not in model"
            raise ValueError(msg)
        node = self._nodes[node_name]
        if node_type is not None and not isinstance(node, node_type):
            msg = f"{node_name} has type {type(node).__name__} instead of {node_type.__name__}"
            raise TypeError(msg)
        return node

    def ordered_nodes(self) -> Iterable[tuple[str, Node]]:
        return sorted(self._nodes.items())

    # Operation methods ----------------------------------------------------------------------------
    def infer_one_time_step(
        self,
        streams: Mapping[str, RngStream],
        state: eqx.nn.State,
        *,
        use_signal_noise: bool,
        return_samples: bool,
    ) -> ModelInferenceResult:
        model_losses: list[JaxArray] = []

        for _, node in self.ordered_nodes():
            inference_result = node.infer(
                self,
                streams,
                state,
                use_signal_noise=use_signal_noise,
                return_samples=return_samples,
            )
            configuration = inference_result.configuration
            state = inference_result.state
            state = node.write_output_to_state(configuration, state)
            model_losses.append(configuration.total_loss())
        total_loss = reduce(add, model_losses) if model_losses else jnp.asarray(0.0)
        return ModelInferenceResult(total_loss, state)

    def set_input(self, field_values: Mapping[str, Any], state: eqx.nn.State) -> eqx.nn.State:
        """Write the input into the state."""
        routing = dict(self._input_routing)
        for field_name, value in field_values.items():
            route = routing.get(field_name)
            if route is None:
                msg = f"Input field {field_name!r} is not declared in input_routing()"
                raise ValueError(msg)
            node_name, node_field_name = route
            node = self.get_node(node_name)
            state = node.set_input(node_field_name, value, state)
        return state

    def configuration_from_state(self, state: eqx.nn.State) -> ModelConfiguration:
        """Reconstruct model configuration by reading each node's most recent output from state."""
        return {
            node_name: node.read_output_from_state(state)
            for node_name, node in self.ordered_nodes()
        }

    def get_output(self, state: eqx.nn.State) -> dict[str, Any]:
        """Read the output from state."""
        retval: dict[str, Any] = {}
        for field_name, (node_name, node_field_name) in self._output_routing:
            node = self.get_node(node_name)
            node_configuration = node.read_output_from_state(state)
            retval[field_name] = node.get_output(node_configuration, node_field_name)
        return retval
