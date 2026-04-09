from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Self, override

import equinox as eqx
import jax
from efax import Flattener, NaturalParametrization
from tjax import JaxRealArray, RngStream, frozendict

from cem.phasor.message import PhasorMessage
from cem.structure.graph.input_node import FlatEncodedInputNode, InputConfiguration
from cem.structure.graph.model import Model
from cem.structure.graph.node import NodeInferenceResult


class PhasorInputConfiguration(InputConfiguration[PhasorMessage]):
    """Holds the environment-provided input values for an input node."""


class PhasorInputNode(FlatEncodedInputNode[PhasorMessage]):
    """InputNode whose fields hold PhasorMessage encodings of distributions.

    Callers create this node by supplying one ``NaturalParametrization`` per field as the
    prior distribution.  Each prior is encoded as a
    :class:`~cem.phasor.message.PhasorMessage` via the characteristic-function encoding and
    stored as the initial field state.

    When the environment provides a new observation for a field it passes a ``JaxRealArray``
    produced by ``Flattener.flatten(dist, mapped_to_plane=True)``.  :meth:`set_input`
    unflattens that array back to a distribution and re-encodes it as a ``PhasorMessage``.

    Attributes:
        _flatteners: One per field (``mapped_to_plane=True``), used to unflatten flat-array
            inputs in :meth:`set_input`.
        frequencies: Geometric frequency grid forwarded to
            :meth:`~cem.phasor.message.PhasorMessage.from_distribution`.
    """

    frequencies: JaxRealArray

    @classmethod
    def _prepare_phasor_fields(
        cls,
        field_defaults: Mapping[str, NaturalParametrization[Any, Any]],
        frequencies: JaxRealArray,
    ) -> tuple[dict[str, Flattener[Any]], dict[str, JaxRealArray], dict[str, PhasorMessage]]:
        flatteners, flat_defaults = cls._prepare_flat_fields(field_defaults)
        phasor_defaults: dict[str, PhasorMessage] = {
            field_name: PhasorMessage.from_distribution(dist, frequencies)
            for field_name, dist in field_defaults.items()
        }
        return flatteners, flat_defaults, phasor_defaults

    @classmethod
    def create(  # ty: ignore
        cls,
        name: str,
        field_defaults: Mapping[str, NaturalParametrization[Any, Any]],
        frequencies: JaxRealArray,
    ) -> Self:
        """Construct a PhasorInputNode from distribution priors and a frequency grid.

        For each field, builds an :class:`efax.Flattener` (``mapped_to_plane=True``) from the
        prior distribution and encodes that prior as a
        :class:`~cem.phasor.message.PhasorMessage` for the initial state.

        Args:
            name: Node name, used for graph routing.
            field_defaults: Mapping from field name to the prior distribution.
            frequencies: Geometric frequency grid, shape ``(m,)``.  Use
                :func:`~cem.phasor.frequency.geometric_frequencies` to generate a standard grid.

        Returns:
            A new :class:`PhasorInputNode` whose state slots are initialised with
            flat real arrays encoding the supplied priors (consistent with what
            :meth:`~cem.structure.graph.input_node.InputNode.set_input` writes).
        """
        assert frequencies.ndim == 1
        flatteners, flat_defaults, phasor_defaults = cls._prepare_phasor_fields(
            field_defaults, frequencies
        )
        state_indices = cls._make_state_indices(flat_defaults)
        zero_config = PhasorInputConfiguration(values=frozendict(phasor_defaults))
        return cls(
            name=name,
            _state_indices=state_indices,
            _output_state_index=eqx.nn.StateIndex(zero_config),
            _flatteners=frozendict(flatteners),
            frequencies=frequencies,
        )

    @override
    def infer(
        self,
        model: Model,
        streams: Mapping[str, RngStream],
        state: eqx.nn.State,
        *,
        inference: bool,
    ) -> NodeInferenceResult[PhasorInputConfiguration]:
        del model, streams, inference
        values = self.gather_local_inputs(state)
        phasors = {}
        for field_name, value in values.items():
            assert isinstance(value, jax.Array)
            distribution = self._flatteners[field_name].unflatten(value)
            phasors[field_name] = PhasorMessage.from_distribution(distribution, self.frequencies)
        return NodeInferenceResult(
            PhasorInputConfiguration(values=frozendict(phasors)),
            state,
        )
