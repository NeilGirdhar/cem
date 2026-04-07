from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Self

import equinox as eqx
from efax import Flattener, NaturalParametrization
from tjax import JaxRealArray, frozendict

from cem.structure.graph.input_node import InputNode, InputNodeConfiguration


class PerceptronInputNode(InputNode[JaxRealArray]):
    """InputNode whose fields hold flat real-valued distribution encodings.

    Each field stores a ``JaxRealArray`` in ``mapped_to_plane=True`` coordinates, i.e. the
    output of ``Flattener.flatten(dist, mapped_to_plane=True)[1]``.  :meth:`set_input`
    writes new flat arrays directly with no encoding step.
    """

    @classmethod
    def create(  # type: ignore[override]  # ty: ignore
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
        flat_defaults: dict[str, JaxRealArray] = {}
        for field_name, dist in field_defaults.items():
            _, flat = Flattener.flatten(dist, mapped_to_plane=True)
            flat_defaults[field_name] = flat

        state_indices = frozendict(
            {field: eqx.nn.StateIndex(v) for field, v in flat_defaults.items()}
        )
        zero_config: InputNodeConfiguration[JaxRealArray] = InputNodeConfiguration(
            values=frozendict(flat_defaults)
        )
        return cls(
            name=name,
            _state_indices=state_indices,
            _output_state_index=eqx.nn.StateIndex(zero_config),
        )
