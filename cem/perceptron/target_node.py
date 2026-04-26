from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Self

from efax import ExpectationParametrization, HasEntropyEP, NaturalParametrization
from tjax import JaxArray, JaxRealArray, frozendict

from cem.perceptron.input_node import PerceptronInputConfiguration
from cem.structure.graph.node import TargetConfiguration, TargetNode


class PerceptronTargetConfiguration(PerceptronInputConfiguration, TargetConfiguration):
    """Scored perceptron targets, keyed by field name."""


class PerceptronTargetNode(TargetNode):
    """Computes cross-entropy loss between observed and predicted perceptron distributions.

    Attributes:
        field_sizes: Total number of prediction values per field, used to split the
            incoming concatenated prediction.  Fields are split in insertion order of
            ``field_defaults`` passed to :meth:`create`.
    """

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
        flatteners_param, field_sizes = cls._build_flatteners(field_defaults)
        return cls(_flatteners=flatteners_param, field_sizes=field_sizes)

    def infer(
        self,
        flat_observed: frozendict[str, JaxRealArray],
        prediction: JaxRealArray,
    ) -> PerceptronTargetConfiguration:
        """Compute cross-entropy loss between observed distributions and a prediction.

        Args:
            flat_observed: Per-field flat distribution encodings in any shape with the
                right total number of elements.
            prediction: Concatenated flat predictions for all fields, shape ``(..., total)``.

        Returns:
            A :class:`PerceptronTargetConfiguration` with per-field losses and distributions.
        """
        field_values = self._split_by_field_sizes(prediction, self.field_sizes)

        losses: dict[str, JaxArray] = {}
        observed_distributions: dict[str, ExpectationParametrization] = {}
        predicted_distributions: dict[str, HasEntropyEP] = {}

        for field_name, y_hat in field_values.items():
            flattener = self._flatteners.value[field_name]
            observed_np = self._unflatten_observed(field_name, flat_observed[field_name])
            observed_exp = observed_np.to_exp()
            assert isinstance(observed_exp, HasEntropyEP)
            predicted_np = flattener.unflatten(y_hat)
            predicted_exp = predicted_np.to_exp()
            assert isinstance(predicted_exp, type(observed_exp))
            observed_distributions[field_name] = observed_exp
            losses[field_name] = observed_exp.cross_entropy(predicted_np)
            predicted_distributions[field_name] = predicted_exp

        return PerceptronTargetConfiguration(
            values=flat_observed,
            loss=frozendict(losses),
            observed_distributions=frozendict(observed_distributions),
            predicted_distributions=frozendict(predicted_distributions),
        )
