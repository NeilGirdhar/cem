from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import InitVar
from functools import reduce
from operator import add
from typing import TYPE_CHECKING, Any, overload, override

import equinox as eqx
import jax.numpy as jnp
import networkx as nx
from tjax import JaxArray, RngStream

if TYPE_CHECKING:
    from cem.structure.problem.creator import ModelCreator

from .edge import Edge
from .editable import EditableModel
from .module import Module
from .node import Node, NodeConfiguration

ModelConfiguration = Mapping[str, NodeConfiguration]


class ModelInferenceResult(eqx.Module):
    loss: JaxArray
    state: eqx.nn.State


class Model(Module):
    """The entire model.

    The model is stored in a directed acyclic graph whose nodes are Node objects and whose directed
    edges between nodes are Edge objects.  The directed edges represent the order in which inference
    proceeds.
    """

    creator: InitVar[ModelCreator[Any]]
    network: nx.DiGraph[str] = eqx.field(init=False)
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
        model = EditableModel()
        creator.create_model(model, streams=streams)
        self.network = model.network
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
        node_dict = self.network.nodes[node_name]
        if "node" not in node_dict:
            raise RuntimeError
        node = node_dict["node"]
        if not isinstance(node, Node):
            msg = f"{node_name} has type {type(node).__name__} instead of Node"
            raise TypeError(msg)
        if node_type is not None and not isinstance(node, node_type):
            msg = f"{node_name} has type {type(node).__name__} instead of {node_type.__name__}"
            raise TypeError(msg)
        return node

    def ordered_nodes(self) -> Iterable[tuple[str, Node]]:
        def filter_edge(source: str, target: str, /) -> bool:
            edge = self.get_edge(source, target, Edge)
            return edge.defines_order()

        view = nx.subgraph_view(self.network, filter_edge=filter_edge)
        for node_name in nx.lexicographical_topological_sort(view):
            yield node_name, self.get_node(node_name)

    def ordered_edges(self) -> Iterable[tuple[str, str, Edge]]:
        ordered_nodes = [name for name, _ in self.ordered_nodes()]
        for x in ordered_nodes:
            for source, target, edge_dict in self.network.edges(x, data=True):
                yield source, target, edge_dict["edge"]

    def get_edge[ET: Edge](self, source: str, target: str, edge_type: type[ET]) -> ET:
        edge_dict = self.network.edges[source, target]
        if "edge" not in edge_dict:
            raise RuntimeError
        edge = edge_dict["edge"]
        if not isinstance(edge, edge_type):
            msg = (
                f"Edge {source}-{target} has type {type(edge).__name__} "
                f"instead of {edge_type.__name__}"
            )
            raise TypeError(msg)
        return edge

    def get_incoming[ET: Edge](
        self, node_name: str, edge_type: type[ET]
    ) -> Iterable[tuple[str, ET]]:
        for source_name, _, edge_dict in self.network.in_edges(node_name, data=True):
            edge = edge_dict["edge"]
            if not isinstance(edge, edge_type):
                continue
            yield source_name, edge

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
