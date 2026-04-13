from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Self

import equinox as eqx
import jax.numpy as jnp
from efax import ExpectationParametrization, Flattener, HasEntropyEP, NaturalParametrization
from tjax import JaxArray, JaxRealArray, frozendict

from cem.perceptron.input_node import PerceptronInputConfiguration
from cem.structure.graph import FixedParameter
from cem.structure.graph.node import TargetConfiguration


class PerceptronTargetConfiguration(PerceptronInputConfiguration, TargetConfiguration):
    """Scored perceptron targets, keyed by field name."""

    observed_distributions: frozendict[str, ExpectationParametrization]
    predicted_distributions: frozendict[str, HasEntropyEP]


def _split_by_field_sizes(
    data: JaxRealArray,
    field_sizes: frozendict[str, int],
) -> dict[str, JaxRealArray]:
    running, split_points = 0, []
    for s in list(field_sizes.values())[:-1]:
        running += s
        split_points.append(running)
    return dict(zip(field_sizes, jnp.split(data, split_points, axis=-1), strict=True))


class PerceptronTargetNode(eqx.Module):
    """Computes cross-entropy loss between observed and predicted perceptron distributions.

    Attributes:
        field_sizes: Per-field flat dimension, used to split the incoming concatenated
            prediction.  Fields are split in insertion order of ``field_defaults`` passed
            to :meth:`create`.
    """

    _flatteners: FixedParameter[frozendict[str, Flattener[Any]]]
    field_sizes: frozendict[str, int] = eqx.field(static=True)

    @classmethod
    def create(
        cls,
        field_defaults: Mapping[str, NaturalParametrization[Any, Any]],
    ) -> Self:
        """Construct a PerceptronTargetNode from distribution priors.

        Args:
            field_defaults: Per-field prior distributions, in the order they will be
                split from the incoming concatenated prediction.

        Returns:
            A new :class:`PerceptronTargetNode`.
        """
        assert len(field_defaults) > 0, "PerceptronTargetNode requires at least one field"
        flatteners: dict[str, Flattener[Any]] = {}
        flat_defaults: dict[str, JaxRealArray] = {}
        for field_name, dist in field_defaults.items():
            flattener, flat = Flattener.flatten(dist, mapped_to_plane=True)
            flatteners[field_name] = flattener
            flat_defaults[field_name] = flat
        field_sizes = frozendict({field: flat.shape[-1] for field, flat in flat_defaults.items()})
        return cls(_flatteners=FixedParameter(frozendict(flatteners)), field_sizes=field_sizes)

    def infer(
        self,
        flat_observed: frozendict[str, JaxRealArray],
        prediction: JaxRealArray,
    ) -> PerceptronTargetConfiguration:
        """Compute cross-entropy loss between observed distributions and a prediction.

        Args:
            flat_observed: Per-field flat distribution encodings
                (``mapped_to_plane=True`` coordinates).
            prediction: Concatenated flat predictions for all fields.

        Returns:
            A :class:`PerceptronTargetConfiguration` with per-field losses and distributions.
        """
        field_values = _split_by_field_sizes(prediction, self.field_sizes)

        losses: dict[str, JaxArray] = {}
        observed_distributions: dict[str, ExpectationParametrization] = {}
        predicted_distributions: dict[str, HasEntropyEP] = {}

        for field_name, y_hat in field_values.items():
            observed_np = self._flatteners.value[field_name].unflatten(flat_observed[field_name])
            observed_exp = observed_np.to_exp()
            assert isinstance(observed_exp, HasEntropyEP)
            predicted_np = self._flatteners.value[field_name].unflatten(y_hat)
            predicted_exp = predicted_np.to_exp()
            assert isinstance(predicted_exp, type(observed_exp))
            observed_distributions[field_name] = observed_exp
            losses[field_name] = observed_exp.cross_entropy(predicted_np)
            predicted_distributions[field_name] = predicted_exp

        return PerceptronTargetConfiguration(
            values=flat_observed,
            observed_distributions=frozendict(observed_distributions),
            loss=frozendict(losses),
            predicted_distributions=frozendict(predicted_distributions),
        )
