from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Self

import equinox as eqx
import jax
import jax.numpy as jnp
from efax import ExpectationParametrization, HasEntropyEP, NaturalParametrization
from tjax import JaxArray, JaxRealArray, RngStream, frozendict

from cem.perceptron.input_node import PerceptronInputConfiguration, PerceptronInputNode
from cem.structure.graph.kernel_node import NodeWithBindings
from cem.structure.graph.model import Model
from cem.structure.graph.node import NodeInferenceResult


class PerceptronTargetConfiguration(PerceptronInputConfiguration):
    """Scored perceptron targets, keyed by field name."""

    observed_distributions: frozendict[str, ExpectationParametrization]
    loss: frozendict[str, JaxArray]
    predicted_distributions: frozendict[str, HasEntropyEP]

    def total_loss(self) -> JaxArray:
        return sum((jnp.sum(loss) for loss in self.loss.values()), start=jnp.asarray(0.0))


class PerceptronTargetNode(PerceptronInputNode, NodeWithBindings):
    """Target node for one or more perceptron-distributed fields.

    Receives a single concatenated flat prediction from one upstream binding and
    splits it into per-field slices using ``field_sizes``.  This mirrors the output
    side, where per-field natural parameters are concatenated into a single flat vector.

    Attributes:
        field_sizes: Per-field flat dimension, used to split the incoming concatenated
            prediction.  Fields are split in insertion order of ``field_defaults`` passed
            to :meth:`create`.
    """

    field_sizes: frozendict[str, int] = eqx.field(static=True)

    @classmethod
    def create(  # ty: ignore
        cls,
        name: str,
        field_defaults: Mapping[str, NaturalParametrization[Any, Any]],
        binding: tuple[str, str],
    ) -> Self:
        """Construct a PerceptronTargetNode.

        Args:
            name: Node name.
            field_defaults: Per-field prior distributions, in the order they will be
                split from the incoming concatenated prediction.
            binding: ``(source_node, source_field)`` of the single upstream node that
                produces the concatenated flat prediction.

        Returns:
            A new :class:`PerceptronTargetNode`.
        """
        assert len(field_defaults) > 0, "PerceptronTargetNode requires at least one field"
        flatteners, flat_defaults = cls._prepare_flat_fields(field_defaults)
        field_sizes = frozendict({field: flat.shape[-1] for field, flat in flat_defaults.items()})
        state_indices = cls._make_state_indices(flat_defaults)
        default_dists = frozendict({field: dist.to_exp() for field, dist in field_defaults.items()})
        zero_config = PerceptronTargetConfiguration(
            values=frozendict(flat_defaults),
            observed_distributions=default_dists,
            loss=frozendict(
                {field: jnp.zeros(flat.shape) for field, flat in flat_defaults.items()}
            ),
            predicted_distributions=default_dists,
        )
        return cls(
            name=name,
            _state_indices=state_indices,
            _output_state_index=eqx.nn.StateIndex(zero_config),
            _bindings=cls.resolve_bindings({"prediction": (binding,)}),
            _flatteners=frozendict(flatteners),
            field_sizes=field_sizes,
        )

    def infer(
        self,
        model: Model,
        streams: Mapping[str, RngStream],
        state: eqx.nn.State,
        *,
        inference: bool,
    ) -> NodeInferenceResult[PerceptronTargetConfiguration]:
        super_result = super().infer(model, streams, state, inference=inference)

        [y_hat_combined] = self.gather_bound_inputs(model, state)["prediction"]
        if not isinstance(y_hat_combined, jax.Array):
            msg = (
                "PerceptronTargetNode expected a JaxRealArray prediction, "
                f"got {type(y_hat_combined).__name__}"
            )
            raise TypeError(msg)

        # Split the combined prediction into per-field flat arrays.
        running, split_points = 0, []
        for s in list(self.field_sizes.values())[:-1]:
            running += s
            split_points.append(running)
        field_values: dict[str, JaxRealArray] = dict(
            zip(
                self.field_sizes,
                jnp.split(y_hat_combined, split_points, axis=-1),
                strict=True,
            )
        )

        losses: dict[str, JaxArray] = {}
        observed_distributions: dict[str, ExpectationParametrization] = {}
        predicted_distributions: dict[str, HasEntropyEP] = {}

        for field_name, y_hat in field_values.items():
            observed_np = self._flatteners[field_name].unflatten(
                super_result.configuration.values[field_name]
            )
            observed_exp = observed_np.to_exp()
            assert isinstance(observed_exp, HasEntropyEP)
            predicted_np = self._flatteners[field_name].unflatten(y_hat)
            predicted_exp = predicted_np.to_exp()
            assert isinstance(predicted_exp, type(observed_exp))
            observed_distributions[field_name] = observed_exp
            losses[field_name] = observed_exp.cross_entropy(predicted_np)
            predicted_distributions[field_name] = predicted_exp

        return NodeInferenceResult(
            PerceptronTargetConfiguration(
                values=super_result.configuration.values,
                observed_distributions=frozendict(observed_distributions),
                loss=frozendict(losses),
                predicted_distributions=frozendict(predicted_distributions),
            ),
            super_result.state,
        )
