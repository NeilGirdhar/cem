"""Score perceptron outputs as parameters of the observed distribution families."""

from collections.abc import Mapping
from typing import Any, Self

from efax import ExpectationParametrization, HasEntropyEP, NaturalParametrization
from tjax import JaxArray, JaxRealArray

from cem.framework.graph.node import TargetConfiguration, TargetNode
from cem.perceptron.input_node import PerceptronInputConfiguration


class PerceptronTargetConfiguration(PerceptronInputConfiguration, TargetConfiguration):
    """Flat target encodings, decoded distributions, and KL losses keyed by field."""


class PerceptronTargetNode(TargetNode):
    """Score a perceptron's predicted distributions against observed distributions."""

    @classmethod
    def create(
        cls,
        field_defaults: Mapping[str, NaturalParametrization[Any, Any]],
    ) -> Self:
        """Configure the distribution family and encoded width of each target field.

        Provide at least one prior. Their insertion order determines the prediction layout.
        """
        assert len(field_defaults) > 0, "PerceptronTargetNode requires at least one field"
        flatteners_param, field_sizes = cls._build_flatteners(field_defaults)
        return cls(_flatteners=flatteners_param, field_sizes=field_sizes)

    def infer(
        self,
        flat_observed: frozendict[str, JaxRealArray],
        prediction: JaxRealArray,
    ) -> PerceptronTargetConfiguration:
        """Return the observed and predicted distributions and their KL losses by field.

        Split the prediction along its last axis using the field widths and order.
        Interpret each predicted slice and observed encoding in the same distribution family.
        """
        losses: dict[str, JaxArray] = {}
        observed_distributions: dict[str, ExpectationParametrization] = {}
        predicted_distributions: dict[str, HasEntropyEP] = {}

        for field_name, y_hat, observed_np, observed_exp in self._iter_target_fields(
            flat_observed, prediction
        ):
            flattener = self._flatteners.value[field_name]
            predicted_np = flattener.unflatten(y_hat)
            predicted_exp = predicted_np.to_exp()
            assert isinstance(predicted_exp, type(observed_exp))
            observed_distributions[field_name] = observed_exp
            losses[field_name] = observed_exp.kl_divergence(predicted_np, self_nat=observed_np)
            predicted_distributions[field_name] = predicted_exp

        return PerceptronTargetConfiguration(
            values=flat_observed,
            loss=frozendict(losses),
            observed_distributions=frozendict(observed_distributions),
            predicted_distributions=frozendict(predicted_distributions),
        )
