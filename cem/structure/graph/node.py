from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Generic, TypeVar

import equinox as eqx
import jax.numpy as jnp
from tjax import JaxArray, RngStream, frozendict
from tjax.dataclasses import dataclass

if TYPE_CHECKING:
    from .model import Model


class NodeConfiguration(eqx.Module):
    """The output produced by a node during one inference step."""

    def total_loss(self) -> JaxArray:
        """Return this node's contribution to the scalar model loss."""
        return jnp.zeros(())


class TargetConfiguration(NodeConfiguration):
    """NodeConfiguration mixin for nodes that compute a per-field loss."""

    loss: frozendict[str, JaxArray]

    def total_loss(self) -> JaxArray:
        return sum((jnp.sum(v) for v in self.loss.values()), start=jnp.asarray(0.0))


_C_co = TypeVar("_C_co", covariant=True, bound=NodeConfiguration, default=NodeConfiguration)


@dataclass
class NodeInferenceResult(Generic[_C_co]):  # noqa: UP046
    """The result of a single node inference step."""

    configuration: _C_co
    state: eqx.nn.State


class Node[ConfigurationT: NodeConfiguration = NodeConfiguration](eqx.Module):
    name: str = eqx.field(static=True)
    _output_state_index: eqx.nn.StateIndex

    def infer(
        self,
        model: Model,
        streams: Mapping[str, RngStream],
        state: eqx.nn.State,
        *,
        inference: bool,
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
