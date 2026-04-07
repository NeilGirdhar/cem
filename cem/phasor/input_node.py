from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Self, override

import equinox as eqx
import jax
from efax import ExpectationParametrization, Flattener, NaturalParametrization
from tjax import JaxRealArray, RngStream, frozendict

from cem.phasor.message import PhasorMessage
from cem.structure.graph.input_node import InputConfiguration, InputNode
from cem.structure.graph.model import Model
from cem.structure.graph.node import NodeInferenceResult


class PhasorInputConfiguration(InputConfiguration[PhasorMessage]):
    """Holds the environment-provided input values for an input node."""

    observed_distributions: frozendict[str, ExpectationParametrization]


class PhasorInputNode(InputNode[PhasorMessage]):
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

    _flatteners: frozendict[str, Flattener[Any]]
    frequencies: JaxRealArray

    @staticmethod
    def _prepare_phasor_fields(
        field_defaults: Mapping[str, NaturalParametrization[Any, Any]],
        frequencies: JaxRealArray,
    ) -> tuple[dict[str, Flattener[Any]], dict[str, JaxRealArray], dict[str, PhasorMessage]]:
        flatteners: dict[str, Flattener[Any]] = {}
        flat_defaults: dict[str, JaxRealArray] = {}
        phasor_defaults: dict[str, PhasorMessage] = {}
        for field_name, dist in field_defaults.items():
            flattener, flat = Flattener.flatten(dist, mapped_to_plane=True)
            flatteners[field_name] = flattener
            flat_defaults[field_name] = flat
            phasor_defaults[field_name] = PhasorMessage.from_distribution(dist, frequencies)
        return flatteners, flat_defaults, phasor_defaults

    @staticmethod
    def _make_state_indices(
        flat_defaults: Mapping[str, JaxRealArray],
    ) -> frozendict[str, eqx.nn.StateIndex]:
        return frozendict({field: eqx.nn.StateIndex(v) for field, v in flat_defaults.items()})

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
        zero_config = PhasorInputConfiguration(
            values=frozendict(phasor_defaults),
            observed_distributions=frozendict({f: d.to_exp() for f, d in field_defaults.items()}),
        )
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
        distributions = {}
        phasors = {}
        for field_name, value in values.items():
            assert isinstance(value, jax.Array)
            distribution = self._flatteners[field_name].unflatten(value)
            distributions[field_name] = distribution.to_exp()
            phasors[field_name] = PhasorMessage.from_distribution(distribution, self.frequencies)
        return NodeInferenceResult(
            PhasorInputConfiguration(
                values=frozendict(phasors), observed_distributions=frozendict(distributions)
            ),
            state,
        )
