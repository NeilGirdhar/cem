from __future__ import annotations

from collections.abc import Iterable, Mapping
from functools import reduce
from operator import add
from typing import Any, Self, overload

import equinox as eqx
import jax.numpy as jnp
from tjax import JaxArray, RngStream, frozendict

from .input_node import InputNode
from .node import Node, NodeBase, NodeConfiguration

ModelConfiguration = Mapping[str, NodeConfiguration]


class ModelInferenceResult(eqx.Module):
    loss: JaxArray
    state: eqx.nn.State


class Model(eqx.Module):
    """The entire model.

    The model is stored as a dictionary of nodes. Inference proceeds in alphabetical order
    by node name. One or more InputNodes hold environment-provided inputs; the model routes
    each field to the correct node automatically via ``_input_routing``.
    """

    _nodes: frozendict[str, NodeBase] = eqx.field(static=True)
    _output_routing: frozendict[str, tuple[str, str]] = eqx.field(static=True)
    _input_routing: frozendict[str, str] = eqx.field(static=True)

    @classmethod
    def create(
        cls,
        nodes: frozendict[str, NodeBase],
        output_routing: frozendict[str, tuple[str, str]],
    ) -> Self:
        """Construct a Model from a pre-built node dictionary.

        Any nodes that are :class:`~cem.structure.graph.input_node.InputNode` instances
        are automatically registered for input routing.  Field names must be unique across
        all InputNodes::

            zero = jnp.zeros(1, dtype=jnp.complex128)
            input_node = InputNode.create("input", field_defaults={"x": zero, "y": zero})
            model = Model.create(
                frozendict({"input": input_node, "encoder": encoder_node}),
                frozendict({"z": ("encoder", "latent")}),
            )

        Args:
            nodes: All nodes in the model, keyed by name.
            output_routing: Maps each externally visible output field name to
                ``(node_name, node_field_name)``.

        Returns:
            A new Model instance.

        """
        return cls(
            _nodes=nodes,
            _output_routing=output_routing,
            _input_routing=Model.build_input_routing(nodes),
        )

    @staticmethod
    def build_input_routing(nodes: frozendict[str, NodeBase]) -> frozendict[str, str]:
        routing: dict[str, str] = {}
        for node_name, node in nodes.items():
            if isinstance(node, InputNode):
                for field_name in node.field_names:
                    if field_name in routing:
                        msg = (
                            f"Input field '{field_name}' declared in both"
                            f" '{routing[field_name]}' and '{node_name}'"
                        )
                        raise ValueError(msg)
                    routing[field_name] = node_name
        return frozendict(routing)

    # Accessing methods ----------------------------------------------------------------------------
    @overload
    def get_node(self, node_name: str) -> NodeBase: ...

    @overload
    def get_node(self, node_name: str, node_type: None) -> NodeBase: ...

    @overload
    def get_node[NT: Node](self, node_name: str, node_type: type[NT]) -> NT: ...

    def get_node(self, node_name: str, node_type: type[Node] | None = None) -> NodeBase:
        """Retrieve a node by name, with optional type narrowing.

        Args:
            node_name: The name of the node to retrieve.
            node_type: If provided, asserts the node is an instance of this type and
                returns it as that type.

        Returns:
            The requested node.

        Raises:
            ValueError: If ``node_name`` is not in the model.
            TypeError: If ``node_type`` is provided and the node is not of that type.
        """
        if node_name not in self._nodes:
            msg = f"{node_name} not in model"
            raise ValueError(msg)
        node = self._nodes[node_name]
        if node_type is not None and not isinstance(node, node_type):
            msg = f"{node_name} has type {type(node).__name__} instead of {node_type.__name__}"
            raise TypeError(msg)
        return node

    def ordered_nodes(self) -> Iterable[tuple[str, NodeBase]]:
        """Return all nodes sorted alphabetically by name.

        Inference always processes nodes in this order, so bindings from node A to node B
        are valid only when A precedes B alphabetically.
        """
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
        """Run one forward pass through every node in alphabetical order.

        Each node reads its inputs from ``state`` (written by earlier nodes or
        :meth:`set_input`), produces a configuration, and writes it back to state.
        The total loss is the sum of each node's :meth:`~NodeConfiguration.total_loss`.

        Args:
            streams: RNG streams passed to each node's ``infer`` method.
            state: Current model state. Updated in-place across nodes.
            use_signal_noise: Passed through to each node's ``infer``.
            return_samples: Passed through to each node's ``infer``.

        Returns:
            A :class:`ModelInferenceResult` with the summed scalar loss and updated state.
        """
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
        """Write environment-provided values into the appropriate InputNode state slots.

        Routes each field to whichever InputNode declared it.  Must be called before
        :meth:`infer_one_time_step` whenever the environment observation changes.

        Args:
            field_values: Mapping from input field name to the new value.
            state: Current model state.

        Returns:
            Updated state with the new input values written in.
        """
        for field_name, value in field_values.items():
            node_name = self._input_routing[field_name]
            node = self._nodes[node_name]
            assert isinstance(node, InputNode)
            state = node.set_input(field_name, value, state)
        return state

    def configuration_from_state(self, state: eqx.nn.State) -> ModelConfiguration:
        """Reconstruct the full model configuration from state.

        Reads each node's most recently written output configuration.  Useful after
        inference to inspect intermediate activations and losses.

        Args:
            state: Model state from which to read.

        Returns:
            Dict mapping each node name to its :class:`~NodeConfiguration`.
        """
        return {
            node_name: node.read_output_from_state(state)
            for node_name, node in self.ordered_nodes()
        }

    def get_output(self, state: eqx.nn.State) -> dict[str, Any]:
        """Read the externally visible outputs declared in ``output_routing``.

        Args:
            state: Model state from which to read.

        Returns:
            Dict mapping each output field name (as declared in ``output_routing``) to
            the corresponding value extracted from the appropriate node's configuration.
        """
        retval: dict[str, Any] = {}
        for field_name, (node_name, node_field_name) in self._output_routing.items():
            node = self.get_node(node_name)
            node_configuration = node.read_output_from_state(state)
            retval[field_name] = node.get_output(node_configuration, node_field_name)
        return retval
