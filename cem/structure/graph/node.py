from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import equinox as eqx
import jax.numpy as jnp
from efax import ExpectationParametrization, Flattener, HasEntropyEP, NaturalParametrization
from tjax import JaxArray, JaxRealArray, frozendict

from cem.structure.graph.parameters import FixedParameter


class NodeConfiguration(eqx.Module):
    """The output produced by a node during one inference step."""

    def total_loss(self) -> JaxArray:
        """Return this node's contribution to the scalar model loss."""
        return jnp.zeros(())


class TargetConfiguration(NodeConfiguration):
    """NodeConfiguration mixin for nodes that compute a per-field loss."""

    loss: frozendict[str, JaxArray]
    observed_distributions: frozendict[str, ExpectationParametrization]
    predicted_distributions: frozendict[str, HasEntropyEP]

    def total_loss(self) -> JaxArray:
        return sum((jnp.sum(v) for v in self.loss.values()), start=jnp.asarray(0.0))


class TargetNode(eqx.Module):
    """Base for nodes that compute cross-entropy loss against observed distributions.

    Holds the per-field flatteners and field sizes used by both perceptron and phasor
    target nodes.  Subclasses provide ``infer`` with their specific prediction type.

    Attributes:
        _flatteners: Per-field Flattener (``mapped_to_plane=True``) used to reconstruct
            distributions from flat observed encodings.
        field_sizes: Per-field size of the prediction along the last axis, used to split
            the incoming concatenated prediction.
    """

    _flatteners: FixedParameter[frozendict[str, Flattener[Any]]]
    field_sizes: frozendict[str, int] = eqx.field(static=True)

    @classmethod
    def _build_flatteners(
        cls,
        field_defaults: Mapping[str, NaturalParametrization[Any, Any]],
    ) -> tuple[FixedParameter[frozendict[str, Flattener[Any]]], frozendict[str, int]]:
        """Create flatteners and flat sizes from distribution priors.

        Returns:
            Tuple of ``(flatteners_param, flat_sizes)`` where ``flat_sizes[field]`` is
            the total number of flat elements for that field.
        """
        flatteners: dict[str, Flattener[Any]] = {}
        flat_sizes: dict[str, int] = {}
        for field_name, dist in field_defaults.items():
            flattener, flat = Flattener.flatten(dist, mapped_to_plane=True)
            flatteners[field_name] = flattener
            flat_sizes[field_name] = flat.size
        return FixedParameter(frozendict(flatteners)), frozendict(flat_sizes)

    @staticmethod
    def _split_by_field_sizes(
        data: JaxRealArray,
        field_sizes: frozendict[str, int],
    ) -> dict[str, JaxRealArray]:
        running, split_points = 0, []
        for s in list(field_sizes.values())[:-1]:
            running += s
            split_points.append(running)
        return dict(zip(field_sizes, jnp.split(data, split_points, axis=-1), strict=True))

    def _unflatten_observed(
        self,
        field_name: str,
        flat: JaxRealArray,
    ) -> NaturalParametrization[Any, Any]:
        return self._flatteners.value[field_name].unflatten(flat)
