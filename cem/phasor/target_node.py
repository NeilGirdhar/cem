from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Self

import equinox as eqx
import jax
import jax.numpy as jnp
from efax import Flattener, HasEntropyEP, NaturalParametrization
from tjax import JaxArray, JaxRealArray, RngStream, frozendict

from cem.phasor.frequency import make_frequency_grid
from cem.phasor.input_node import PhasorInputConfiguration, PhasorInputNode
from cem.phasor.message import PhasorMessage
from cem.structure.graph.kernel_node import NodeWithBindings
from cem.structure.graph.model import Model
from cem.structure.graph.node import NodeInferenceResult


class PhasorTargetConfiguration(PhasorInputConfiguration):
    """Scored phasor targets, keyed by field name."""

    score: PhasorMessage
    loss: frozendict[str, JaxArray]
    predicted_distributions: frozendict[str, HasEntropyEP]

    def total_loss(self) -> JaxArray:
        return sum((jnp.sum(loss) for loss in self.loss.values()), start=jnp.asarray(0.0))


class PhasorTargetNode(PhasorInputNode, NodeWithBindings):
    """Target node for one or more phasor-distributed fields.

    Receives a single concatenated prediction phasor from one upstream binding and
    splits it into per-field slices using ``field_sizes``.  This mirrors the output
    side, where per-field scores are concatenated into a single phasor.

    Attributes:
        frequency_grids: Per-field frequency grid ``t``, shape ``(m * d,)`` each,
            used to recover expectation parameters from predicted phasors.
        field_sizes: Per-field phasor dimension ``m * d``, used to split the incoming
            concatenated prediction.  Fields are split in insertion order of
            ``field_defaults`` passed to :meth:`create`.
    """

    frequency_grids: frozendict[str, NaturalParametrization[Any, Any]] = eqx.field(static=True)
    field_sizes: frozendict[str, int] = eqx.field(static=True)

    @classmethod
    def create(  # type: ignore[override]  # ty: ignore
        cls,
        name: str,
        field_defaults: Mapping[str, NaturalParametrization[Any, Any]],
        binding: tuple[str, str],
        frequencies: JaxRealArray,
    ) -> Self:
        """Construct a PhasorTargetNode.

        Args:
            name: Node name.
            field_defaults: Per-field prior distributions, in the order they will be
                split from the incoming concatenated prediction.
            binding: ``(source_node, source_field)`` of the single upstream node that
                produces the concatenated prediction phasor.
            frequencies: Geometric frequency grid, shape ``(m,)``.

        Returns:
            A new :class:`PhasorTargetNode`.
        """
        assert len(field_defaults) > 0, "PhasorTargetNode requires at least one field"
        assert frequencies.ndim == 1
        flatteners, flat_defaults, phasor_defaults = cls._prepare_phasor_fields(
            field_defaults, frequencies
        )
        frequency_grids: dict[str, NaturalParametrization[Any, Any]] = {}
        for field_name, dist in field_defaults.items():
            nat_flattener, _ = Flattener.flatten(dist, mapped_to_plane=False)
            frequency_grids[field_name] = make_frequency_grid(nat_flattener, frequencies)

        field_sizes = frozendict(
            {field: phasor.data.shape[-1] for field, phasor in phasor_defaults.items()}
        )
        state_indices = cls._make_state_indices(flat_defaults)
        default_dists = frozendict({field: dist.to_exp() for field, dist in field_defaults.items()})
        zero_phasor = PhasorMessage(
            jnp.concatenate([p.data for p in phasor_defaults.values()], axis=-1)
        ).zeros_like()
        zero_config = PhasorTargetConfiguration(
            values=frozendict(phasor_defaults),
            observed_distributions=default_dists,
            score=zero_phasor,
            loss=frozendict(
                {field: jnp.zeros(phasor.shape) for field, phasor in phasor_defaults.items()}
            ),
            predicted_distributions=default_dists,
        )
        return cls(
            name=name,
            _state_indices=state_indices,
            _output_state_index=eqx.nn.StateIndex(zero_config),
            _bindings=cls.resolve_bindings({"prediction": (binding,)}),
            _flatteners=frozendict(flatteners),
            frequencies=frequencies,
            frequency_grids=frozendict(frequency_grids),
            field_sizes=field_sizes,
        )

    def infer(
        self,
        model: Model,
        streams: Mapping[str, RngStream],
        state: eqx.nn.State,
        *,
        inference: bool,
    ) -> NodeInferenceResult[PhasorTargetConfiguration]:
        super_result = super().infer(model, streams, state, inference=inference)

        [z_hat_combined] = self.gather_bound_inputs(model, state)["prediction"]
        if not isinstance(z_hat_combined, PhasorMessage):
            msg = (
                "PhasorTargetNode expected a PhasorMessage prediction, "
                f"got {type(z_hat_combined).__name__}"
            )
            raise TypeError(msg)

        # Split the combined prediction into per-field phasors.
        running, split_points = 0, []
        for s in list(self.field_sizes.values())[:-1]:
            running += s
            split_points.append(running)
        field_phasors = {
            f: PhasorMessage(p)
            for f, p in zip(
                self.field_sizes, jnp.split(z_hat_combined.data, split_points, axis=-1), strict=True
            )
        }

        scores: list[PhasorMessage] = []
        losses: dict[str, JaxArray] = {}
        predicted_distributions: dict[str, HasEntropyEP] = {}

        for field_name, observed_exp in super_result.configuration.observed_distributions.items():
            assert isinstance(observed_exp, HasEntropyEP)
            z_hat = field_phasors[field_name]

            def forward(
                z: PhasorMessage,
                grid: NaturalParametrization[Any, Any] = self.frequency_grids[field_name],
                observed_exp: HasEntropyEP = observed_exp,
            ) -> tuple[JaxArray, tuple[JaxArray, HasEntropyEP]]:
                predicted_exp = z.to_distribution(grid)
                assert isinstance(predicted_exp, type(observed_exp))
                field_losses = observed_exp.cross_entropy(predicted_exp.to_nat())
                return jnp.sum(field_losses), (field_losses, predicted_exp)  # ty: ignore

            (_, (field_losses, predicted_exp)), score = jax.value_and_grad(forward, has_aux=True)(
                z_hat
            )
            scores.append(score)
            losses[field_name] = field_losses
            predicted_distributions[field_name] = predicted_exp

        return NodeInferenceResult(
            PhasorTargetConfiguration(
                values=super_result.configuration.values,
                observed_distributions=super_result.configuration.observed_distributions,
                score=PhasorMessage(jnp.concatenate([s.data for s in scores], axis=-1)),
                loss=frozendict(losses),
                predicted_distributions=frozendict(predicted_distributions),
            ),
            super_result.state,
        )
