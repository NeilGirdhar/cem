from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Self, override

import equinox as eqx
import jax
from efax import Flattener, NaturalParametrization
from tjax import JaxRealArray, frozendict

from cem.phasor.message import PhasorMessage
from cem.structure.graph.input_node import InputNode, InputNodeConfiguration


class PhasorInputNode(InputNode):
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

    _flatteners: frozendict[str, Flattener[Any]] = eqx.field(static=True)
    frequencies: JaxRealArray

    @classmethod
    def create(  # type: ignore[override]  # ty: ignore
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
            ``PhasorMessage`` encodings of the supplied priors.
        """
        assert frequencies.ndim == 1
        flatteners: dict[str, Flattener[Any]] = {}
        phasor_defaults: dict[str, PhasorMessage] = {}
        for field_name, dist in field_defaults.items():
            flattener, _ = Flattener.flatten(dist, mapped_to_plane=True)
            flatteners[field_name] = flattener
            phasor_defaults[field_name] = PhasorMessage.from_distribution(dist, frequencies)

        state_indices = frozendict(
            {field: eqx.nn.StateIndex(v) for field, v in phasor_defaults.items()}
        )
        zero_config = InputNodeConfiguration(values=frozendict(phasor_defaults))
        return cls(
            name=name,
            _state_indices=state_indices,
            _output_state_index=eqx.nn.StateIndex(zero_config),
            _flatteners=frozendict(flatteners),
            frequencies=frequencies,
        )

    @override
    def set_input(self, field_name: str, new_value: object, state: eqx.nn.State) -> eqx.nn.State:
        """Update a field from a flat real-valued distribution encoding.

        Unflattens ``new_value`` from ``mapped_to_plane=True`` coordinates back to a
        ``NaturalParametrization``, encodes it as a
        :class:`~cem.phasor.message.PhasorMessage`, and writes the result into state.

        Args:
            field_name: Name of the field to update.
            new_value: Flat ``JaxRealArray`` in ``mapped_to_plane=True`` coordinates, as
                produced by ``Flattener.flatten(dist, mapped_to_plane=True)[1]``.
            state: Current model state.

        Returns:
            Updated state with the field set to a ``PhasorMessage`` encoding of the
            unflattened distribution.
        """
        assert isinstance(new_value, jax.Array)
        dist = self._flatteners[field_name].unflatten(new_value)
        phasor = PhasorMessage.from_distribution(dist, self.frequencies)
        return super().set_input(field_name, phasor, state)
