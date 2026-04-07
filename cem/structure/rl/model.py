from __future__ import annotations

from typing import Any, Self

import equinox as eqx
from tjax import frozendict

from cem.structure.graph import Node
from cem.structure.graph.model import Model
from cem.structure.problem.data_source import ProblemState


class RLModel[ProblemStateT: ProblemState](Model):
    """Model subclass for reinforcement learning, adding action and reward routing.

    Extends :class:`~cem.structure.graph.model.Model` with two additional routing tables
    that map externally requested action and reward field names to nodes.  The caller is
    responsible for including an ``'input'`` node in ``nodes``; see :meth:`create_rl`.
    """

    _reward_routing: frozendict[str, tuple[str, str]] = eqx.field(static=True)
    _action_routing: frozendict[str, tuple[str, str]] = eqx.field(static=True)

    @classmethod
    def create_rl(
        cls,
        nodes: frozendict[str, Node],
        output_routing: frozendict[str, tuple[str, str]],
        reward_routing: frozendict[str, tuple[str, str]],
        action_routing: frozendict[str, tuple[str, str]],
    ) -> Self:
        """Construct an RLModel from a pre-built node dictionary.

        The caller is responsible for creating and including the ``'input'`` node.
        Use :class:`~cem.structure.graph.input_node.InputNode` for this::

            zero = jnp.zeros(1, dtype=jnp.complex128)
            input_node = InputNode.create("input", field_defaults={"obs": zero})
            model = RLModel.create_rl(
                frozendict({"input": input_node, "policy": policy_node}),
                output_routing=frozendict({}),
                reward_routing=frozendict({"r": ("policy", "reward")}),
                action_routing=frozendict({"a": ("policy", "action")}),
            )

        Args:
            nodes: All nodes in the model, keyed by name. Must include an ``'input'`` node.
            output_routing: Maps each externally visible output field name to
                ``(node_name, node_field_name)``.
            reward_routing: Maps each externally requested reward field name to
                ``(node_name, node_field_name)``.
            action_routing: Maps each externally requested action field name to
                ``(node_name, node_field_name)``.

        Returns:
            A new RLModel instance.
        """
        return cls(
            _nodes=nodes,
            _output_routing=output_routing,
            _input_routing=Model.build_input_routing(nodes),
            _reward_routing=reward_routing,
            _action_routing=action_routing,
        )

    def get_reward(self, state: eqx.nn.State) -> dict[str, Any]:
        """Read the reward fields declared in ``reward_routing``.

        Args:
            state: Model state from which to read.

        Returns:
            Dict mapping each reward field name to the value from the appropriate node.
        """
        retval: dict[str, Any] = {}
        for field_name, (node_name, node_field_name) in self._reward_routing.items():
            node = self.get_node(node_name)
            node_configuration = node.read_output_from_state(state)
            retval[field_name] = node.get_output(node_configuration, node_field_name)
        return retval

    def get_action(self, state: eqx.nn.State) -> dict[str, Any]:
        """Read the action fields declared in ``action_routing``.

        Args:
            state: Model state from which to read.

        Returns:
            Dict mapping each action field name to the value from the appropriate node.
        """
        retval: dict[str, Any] = {}
        for field_name, (node_name, node_field_name) in self._action_routing.items():
            node = self.get_node(node_name)
            node_configuration = node.read_output_from_state(state)
            retval[field_name] = node.get_output(node_configuration, node_field_name)
        return retval
