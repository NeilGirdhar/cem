from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Self, override

import equinox as eqx
from efax import NaturalParametrization
from tjax import JaxRealArray, RngStream, frozendict

from cem.structure.graph.input_node import FlatEncodedInputNode, InputConfiguration
from cem.structure.graph.model import Model
from cem.structure.graph.node import NodeInferenceResult


class PerceptronInputConfiguration(InputConfiguration[JaxRealArray]):
    """Holds flat distribution encodings for an input node."""


class PerceptronInputNode(FlatEncodedInputNode[JaxRealArray]):
    """InputNode whose fields hold flat real-valued distribution encodings.

    Each field stores a ``JaxRealArray`` in ``mapped_to_plane=True`` coordinates, i.e. the
    output of ``Flattener.flatten(dist, mapped_to_plane=True)[1]``.  :meth:`set_input`
    writes new flat arrays directly with no encoding step.

    Attributes:
        _flatteners: One per field (``mapped_to_plane=True``), used to unflatten stored
            flat arrays back to distributions in :meth:`infer`.
    """

    @classmethod
    def create(  # ty: ignore
        cls,
        name: str,
        field_defaults: Mapping[str, NaturalParametrization[Any, Any]],
    ) -> Self:
        """Construct a PerceptronInputNode from distribution priors.

        Flattens each prior (``mapped_to_plane=True``) to obtain the initial flat value.

        Args:
            name: Node name, used for graph routing.
            field_defaults: Mapping from field name to the prior distribution.

        Returns:
            A new :class:`PerceptronInputNode` whose state slots are initialised with flat
            real arrays encoding the supplied priors.
        """
        flatteners, flat_defaults = cls._prepare_flat_fields(field_defaults)
        state_indices = cls._make_state_indices(flat_defaults)
        zero_config = PerceptronInputConfiguration(values=frozendict(flat_defaults))
        return cls(
            name=name,
            _state_indices=state_indices,
            _output_state_index=eqx.nn.StateIndex(zero_config),
            _flatteners=frozendict(flatteners),
        )

    @override
    def infer(
        self,
        model: Model,
        streams: Mapping[str, RngStream],
        state: eqx.nn.State,
        *,
        inference: bool,
    ) -> NodeInferenceResult[PerceptronInputConfiguration]:
        del model, streams, inference
        values = self.gather_local_inputs(state)
        return NodeInferenceResult(
            PerceptronInputConfiguration(values=frozendict(values)),
            state,
        )
