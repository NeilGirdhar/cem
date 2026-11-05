"""Node outputs and the shared interface for scoring predicted distributions."""

from collections.abc import Iterator, Mapping
from typing import Any

import equinox as eqx
import jax.numpy as jnp
from efax import ExpectationParametrization, Flattener, HasEntropyEP, NaturalParametrization
from tjax import JaxArray, JaxRealArray

from cem.structure.graph.parameters import FixedParameter


class NodeConfiguration(eqx.Module):
    """The output produced by a node during one inference step."""

    def total_loss(self) -> JaxArray:
        """Return zero for a node that defines no loss of its own."""
        return jnp.zeros(())


class TargetConfiguration(NodeConfiguration):
    """Observed and predicted distributions with a loss for each target field."""

    loss: frozendict[str, JaxArray]
    observed_distributions: frozendict[str, ExpectationParametrization]
    predicted_distributions: frozendict[str, HasEntropyEP]

    def total_loss(self) -> JaxArray:
        """Sum losses across fields and all elements of each field."""
        return sum((jnp.sum(v) for v in self.loss.values()), start=jnp.asarray(0.0))


class TargetNode(eqx.Module):
    """Base for nodes that score predictions against observed distributions.

    A target node records each field's distribution family and encoded width so it
    can pair predictions with the corresponding observations.
    """

    _flatteners: FixedParameter[frozendict[str, Flattener[Any]]]
    field_sizes: frozendict[str, int] = eqx.field(static=True)

    @classmethod
    def _build_flatteners(
        cls,
        field_defaults: Mapping[str, NaturalParametrization[Any, Any]],
    ) -> tuple[FixedParameter[frozendict[str, Flattener[Any]]], frozendict[str, int]]:
        """Derive each target's plane-coordinate encoding and flat width from its prior."""
        flatteners: dict[str, Flattener[Any]] = {}
        flat_sizes: dict[str, int] = {}
        for field_name, dist in field_defaults.items():
            flattener, flat = Flattener.flatten(dist, mapped_to_plane=True)
            flatteners[field_name] = flattener
            flat_sizes[field_name] = flat.size
        return FixedParameter(frozendict(flatteners)), frozendict(flat_sizes)

    @staticmethod
    def _split_by_field_sizes(
        data: JaxArray,
        field_sizes: frozendict[str, int],
    ) -> dict[str, JaxArray]:
        """Split the last axis in the order of the declared target fields."""
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

    def _iter_target_fields(
        self,
        flat_observed: frozendict[str, JaxRealArray],
        prediction: JaxArray,
    ) -> Iterator[tuple[str, JaxArray, NaturalParametrization[Any, Any], HasEntropyEP]]:
        """Pair each field prediction with its observed natural and expectation parameters."""
        for field_name, field_prediction in self._split_by_field_sizes(
            prediction, self.field_sizes
        ).items():
            observed_np = self._unflatten_observed(field_name, flat_observed[field_name])
            observed_exp = observed_np.to_exp()
            assert isinstance(observed_exp, HasEntropyEP)
            yield field_name, field_prediction, observed_np, observed_exp
