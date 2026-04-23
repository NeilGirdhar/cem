from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Self

import jax
import jax.numpy as jnp
from efax import ExpectationParametrization, Flattener, HasEntropyEP, NaturalParametrization
from tjax import JaxArray, JaxRealArray, frozendict

from cem.phasor.frequency import make_frequency_grid
from cem.phasor.input_node import PhasorInputConfiguration
from cem.phasor.message import PhasorMessage
from cem.structure.graph import FixedParameter
from cem.structure.graph.node import TargetConfiguration, TargetNode


class PhasorTargetConfiguration(PhasorInputConfiguration, TargetConfiguration):
    """Scored phasor targets, keyed by field name."""

    score: PhasorMessage


class PhasorTargetNode(TargetNode):
    """Computes phasor reconstruction loss between observed and predicted distributions.

    Attributes:
        frequency_grids: Per-field frequency grid ``t``, shape ``(m * d,)`` each,
            used to recover expectation parameters from predicted phasors.
        field_sizes: Per-field phasor dimension ``m * d``, used to split the incoming
            concatenated prediction.
        frequencies: Geometric frequency grid forwarded to
            :meth:`~cem.phasor.message.PhasorMessage.from_distribution`.
    """

    frequency_grids: FixedParameter[frozendict[str, NaturalParametrization[Any, Any]]]
    frequencies: FixedParameter[JaxRealArray]

    @classmethod
    def create(
        cls,
        field_defaults: Mapping[str, NaturalParametrization[Any, Any]],
        frequencies: JaxRealArray,
    ) -> Self:
        """Construct a PhasorTargetNode from distribution priors and a frequency grid.

        Args:
            field_defaults: Per-field prior distributions, in the order they will be
                split from the incoming concatenated prediction.
            frequencies: Geometric frequency grid, shape ``(m,)``.

        Returns:
            A new :class:`PhasorTargetNode`.
        """
        assert len(field_defaults) > 0, "PhasorTargetNode requires at least one field"
        assert frequencies.ndim == 1
        flatteners_param, _ = cls._build_flatteners(field_defaults)
        phasor_defaults: dict[str, PhasorMessage] = {}
        frequency_grids: dict[str, NaturalParametrization[Any, Any]] = {}
        for field_name, dist in field_defaults.items():
            phasor_defaults[field_name] = PhasorMessage.from_distribution(dist, frequencies)
            nat_flattener, _ = Flattener.flatten(dist, mapped_to_plane=False)
            frequency_grids[field_name] = make_frequency_grid(nat_flattener, frequencies)
        field_sizes = frozendict(
            {field: phasor.data.shape[-1] for field, phasor in phasor_defaults.items()}
        )
        return cls(
            _flatteners=flatteners_param,
            frequency_grids=FixedParameter(frozendict(frequency_grids)),
            field_sizes=field_sizes,
            frequencies=FixedParameter(frequencies),
        )

    def infer(
        self,
        flat_observed: frozendict[str, JaxRealArray],
        z_hat_combined: PhasorMessage,
    ) -> PhasorTargetConfiguration:
        """Compute phasor reconstruction loss between observations and a prediction.

        Args:
            flat_observed: Per-field flat distribution encodings
                (``mapped_to_plane=True`` coordinates).
            z_hat_combined: Concatenated prediction phasor for all fields.

        Returns:
            A :class:`PhasorTargetConfiguration` with per-field losses, scores, and
            distributions.
        """
        field_phasors = {
            f: PhasorMessage(p)
            for f, p in self._split_by_field_sizes(z_hat_combined.data, self.field_sizes).items()
        }

        phasors: dict[str, PhasorMessage] = {}
        scores: list[PhasorMessage] = []
        losses: dict[str, JaxArray] = {}
        observed_distributions: dict[str, ExpectationParametrization] = {}
        predicted_distributions: dict[str, HasEntropyEP] = {}

        for field_name, z_hat in field_phasors.items():
            grid = self.frequency_grids.value[field_name]
            observed_np = self._unflatten_observed(field_name, flat_observed[field_name])
            phasors[field_name] = PhasorMessage.from_distribution(
                observed_np, self.frequencies.value
            )
            observed_exp = observed_np.to_exp()
            assert isinstance(observed_exp, HasEntropyEP)

            def cross_entropy_fn(
                z: PhasorMessage,
                grid: NaturalParametrization[Any, Any] = grid,
                observed_exp: HasEntropyEP = observed_exp,
            ) -> tuple[JaxArray, ExpectationParametrization]:
                pred = z.to_distribution(grid)
                return observed_exp.cross_entropy(pred.to_nat()), pred

            (field_losses, predicted_exp), vjp_fn = jax.vjp(cross_entropy_fn, z_hat)
            assert isinstance(predicted_exp, type(observed_exp))
            zero_pred_ct = jax.tree.map(jnp.zeros_like, predicted_exp)
            (score,) = vjp_fn((jnp.ones_like(field_losses), zero_pred_ct))

            observed_distributions[field_name] = observed_exp
            scores.append(score)
            losses[field_name] = field_losses
            predicted_distributions[field_name] = predicted_exp

        return PhasorTargetConfiguration(
            values=frozendict(phasors),
            loss=frozendict(losses),
            observed_distributions=frozendict(observed_distributions),
            score=PhasorMessage(jnp.concatenate([s.data for s in scores], axis=-1)),
            predicted_distributions=frozendict(predicted_distributions),
        )
