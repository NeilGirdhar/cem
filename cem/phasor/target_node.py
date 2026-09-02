from collections.abc import Mapping
from typing import Any, Self

import jax.numpy as jnp
from efax import ExpectationParametrization, Flattener, HasEntropyEP, NaturalParametrization
from jax.lax import stop_gradient
from tjax import JaxArray, JaxRealArray, copy_cotangent, frozendict

from cem.phasor.input_node import PhasorInputConfiguration
from cem.phasor.loss import phasor_reconstruction_loss_and_score
from cem.phasor.message import JaxComplexArray
from cem.structure.graph import FixedParameter
from cem.structure.graph.node import TargetConfiguration, TargetNode
from cem.transforms import decode_observation_phasors, encode_observation_phasors


class PhasorTargetConfiguration(PhasorInputConfiguration, TargetConfiguration):
    """A scored one-phasor target for each expectation parameter."""

    score: JaxComplexArray
    reconstruction_loss: frozendict[str, JaxArray]

    def total_reconstruction_loss(self) -> JaxArray:
        """Sum the phasor reconstruction objective across fields."""
        return sum(
            (jnp.sum(value) for value in self.reconstruction_loss.values()),
            start=jnp.asarray(0.0),
        )


class PhasorTargetNode(TargetNode):
    """Score one predicted phasor per observation expectation parameter."""

    _expectation_flatteners: FixedParameter[frozendict[str, Flattener[Any]]]

    @classmethod
    def create(
        cls,
        field_defaults: Mapping[str, NaturalParametrization[Any, Any]],
    ) -> Self:
        """Create a target node from the prior distribution for each field."""
        if not field_defaults:
            msg = "PhasorTargetNode requires at least one field"
            raise ValueError(msg)

        natural_flatteners, _ = cls._build_flatteners(field_defaults)
        expectation_flatteners: dict[str, Flattener[Any]] = {}
        field_sizes: dict[str, int] = {}
        for field_name, distribution in field_defaults.items():
            flattener, flat = Flattener.flatten(
                distribution.to_exp(),
                mapped_to_plane=True,
            )
            expectation_flatteners[field_name] = flattener
            field_sizes[field_name] = flat.size

        return cls(
            _flatteners=natural_flatteners,
            field_sizes=frozendict(field_sizes),
            _expectation_flatteners=FixedParameter(frozendict(expectation_flatteners)),
        )

    def infer(
        self,
        flat_observed: frozendict[str, JaxRealArray],
        prediction: JaxComplexArray,
    ) -> PhasorTargetConfiguration:
        """Score and decode concatenated one-phasor predictions."""
        phasors: dict[str, JaxComplexArray] = {}
        scores: list[JaxComplexArray] = []
        losses: dict[str, JaxArray] = {}
        reconstruction_losses: dict[str, JaxArray] = {}
        observed_distributions: dict[str, ExpectationParametrization] = {}
        predicted_distributions: dict[str, HasEntropyEP] = {}

        predictions = self._split_by_field_sizes(prediction, self.field_sizes)
        for field_name, predicted_phasors in predictions.items():
            observed_np = self._unflatten_observed(field_name, flat_observed[field_name])
            observed_exp = observed_np.to_exp()
            assert isinstance(observed_exp, HasEntropyEP)
            _, observed_values = Flattener.flatten(observed_exp, mapped_to_plane=True)
            observed_phasors = encode_observation_phasors(
                jnp.ones_like(observed_values),
                observed_values,
            )
            reconstruction = phasor_reconstruction_loss_and_score(
                observed_phasors,
                predicted_phasors,
            )

            decoded_values = decode_observation_phasors(stop_gradient(predicted_phasors))
            predicted_exp = self._expectation_flatteners.value[field_name].unflatten(decoded_values)
            assert isinstance(predicted_exp, HasEntropyEP)
            assert isinstance(predicted_exp, type(observed_exp))
            distributional_loss = observed_exp.kl_divergence(
                predicted_exp.to_nat(),
                self_nat=observed_np,
            )

            phasors[field_name] = observed_phasors
            scores.append(reconstruction.score)
            reconstruction_losses[field_name] = reconstruction.loss
            losses[field_name] = copy_cotangent(
                stop_gradient(distributional_loss),
                reconstruction.loss,
            )
            observed_distributions[field_name] = observed_exp
            predicted_distributions[field_name] = predicted_exp

        return PhasorTargetConfiguration(
            values=frozendict(phasors),
            loss=frozendict(losses),
            reconstruction_loss=frozendict(reconstruction_losses),
            observed_distributions=frozendict(observed_distributions),
            predicted_distributions=frozendict(predicted_distributions),
            score=jnp.concatenate(scores, axis=-1),
        )
