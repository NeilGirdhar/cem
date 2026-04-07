from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Self, override

import equinox as eqx
import jax
from efax import Flattener, NaturalParametrization
from tjax import JaxRealArray

from cem.phasor_calculus.message import PhasorMessage
from cem.structure.graph.input_node import InputNode, InputNodeConfiguration


class PhasorInputNode(InputNode):
    """InputNode whose fields hold PhasorMessage encodings of distributions.

    Callers create this node by supplying one ``NaturalParametrization`` per field as the
    prior distribution.  Each prior is encoded as a
    :class:`~cem.phasor_calculus.message.PhasorMessage` via the characteristic-function encoding and
    stored as the initial field state.

    When the environment provides a new observation for a field it passes a ``JaxRealArray``
    produced by ``Flattener.flatten(dist, mapped_to_plane=True)``.  :meth:`set_input`
    unflattens that array back to a distribution and re-encodes it as a ``PhasorMessage``.

    Attributes:
        _flatteners: One per field, built from the prior distribution to record the fixed
            structure needed to unflatten future flat-array inputs.
        frequencies: Geometric frequency grid forwarded to
            :meth:`~cem.phasor_calculus.message.PhasorMessage.from_distribution`.
    """

    _flatteners: tuple[Flattener[Any], ...] = eqx.field(static=True)
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
        :class:`~cem.phasor_calculus.message.PhasorMessage` for the initial state.

        Args:
            name: Node name, used for graph routing.
            field_defaults: Mapping from field name to the prior distribution.
            frequencies: Geometric frequency grid, shape ``(m,)``.  Use
                :func:`~cem.phasor_calculus.message.geometric_frequencies` to generate a standard
                grid.

        Returns:
            A new :class:`PhasorInputNode` whose state slots are initialised with
            ``PhasorMessage`` encodings of the supplied priors.
        """
        assert frequencies.ndim == 1
        flatteners: list[Flattener[Any]] = []
        phasor_defaults: dict[str, PhasorMessage] = {}
        for field_name, dist in field_defaults.items():
            flattener, _ = Flattener.flatten(dist, mapped_to_plane=True)
            flatteners.append(flattener)
            phasor_defaults[field_name] = PhasorMessage.from_distribution(dist, frequencies)

        field_names = tuple(phasor_defaults.keys())
        defaults = tuple(phasor_defaults.values())
        zero_config = InputNodeConfiguration(values=defaults)
        return cls(
            name=name,
            field_names=field_names,
            _state_indices=tuple(eqx.nn.StateIndex(v) for v in defaults),
            _output_state_index=eqx.nn.StateIndex(zero_config),
            _flatteners=tuple(flatteners),
            frequencies=frequencies,
        )

    @override
    def set_input(self, field_name: str, new_value: object, state: eqx.nn.State) -> eqx.nn.State:
        """Update a field from a flat real-valued distribution encoding.

        Unflattens ``new_value`` from ``mapped_to_plane=True`` coordinates back to a
        ``NaturalParametrization``, encodes it as a
        :class:`~cem.phasor_calculus.message.PhasorMessage`, and writes the result into state.

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
        idx = self.field_names.index(field_name)
        dist = self._flatteners[idx].unflatten(new_value)
        phasor = PhasorMessage.from_distribution(dist, self.frequencies)
        return state.set(self._state_indices[idx], phasor)
