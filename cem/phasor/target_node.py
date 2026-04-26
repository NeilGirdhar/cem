from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Self

import jax.numpy as jnp
from efax import ExpectationParametrization, Flattener, HasEntropyEP, NaturalParametrization
from jax.lax import stop_gradient
from tjax import JaxArray, JaxRealArray, copy_cotangent, frozendict

from cem.phasor.frequency import make_frequency_grid
from cem.phasor.input_node import PhasorInputConfiguration
from cem.phasor.loss import spectral_reconstruction_loss_and_score
from cem.phasor.message import PhasorMessage
from cem.structure.graph import FixedParameter
from cem.structure.graph.node import TargetConfiguration, TargetNode


class PhasorTargetConfiguration(PhasorInputConfiguration, TargetConfiguration):
    """Scored phasor targets, keyed by field name."""

    score: PhasorMessage


class PhasorTargetNode(TargetNode):
    """Target node: distributional reconstruction loss reported, spectral loss for the gradient.

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
        prediction: PhasorMessage,
    ) -> PhasorTargetConfiguration:
        """Compute distributional reconstruction loss between observations and a predicted phasor.

        Args:
            flat_observed: Per-field flat distribution encodings
                (``mapped_to_plane=True`` coordinates).
            prediction: Concatenated prediction phasor for all fields.

        Returns:
            A :class:`PhasorTargetConfiguration` with per-field losses, scores, and
            distributions.
        """
        phasors: dict[str, PhasorMessage] = {}
        scores: list[PhasorMessage] = []
        losses: dict[str, JaxArray] = {}
        observed_distributions: dict[str, ExpectationParametrization] = {}
        predicted_distributions: dict[str, HasEntropyEP] = {}

        field_values = self._split_by_field_sizes(prediction.data, self.field_sizes)
        field_phasors = {f: PhasorMessage(p) for f, p in field_values.items()}

        for field_name, z_hat in field_phasors.items():
            grid = self.frequency_grids.value[field_name]
            observed_np = self._unflatten_observed(field_name, flat_observed[field_name])
            obs_phasor = PhasorMessage.from_distribution(observed_np, self.frequencies.value)
            phasors[field_name] = obs_phasor
            observed_exp = observed_np.to_exp()
            assert isinstance(observed_exp, HasEntropyEP)

            # Use spectral reconstruction loss for the gradient: decoding to distribution space
            # first gives near-zero gradients (~500x smaller) and phase-wrapping errors.
            spectral = spectral_reconstruction_loss_and_score(obs_phasor, z_hat)
            predicted_exp = z_hat.to_distribution(grid)
            assert isinstance(predicted_exp, type(observed_exp))

            observed_distributions[field_name] = observed_exp
            scores.append(spectral.score)
            distributional_loss = stop_gradient(observed_exp.cross_entropy(predicted_exp.to_nat()))
            optimized_loss = jnp.sum(spectral.loss, axis=-1)
            # Report the distributional loss, but optimize phasor reconstruction gradient.
            losses[field_name] = copy_cotangent(distributional_loss, optimized_loss)
            predicted_distributions[field_name] = predicted_exp

        return PhasorTargetConfiguration(
            values=frozendict(phasors),
            loss=frozendict(losses),
            observed_distributions=frozendict(observed_distributions),
            score=PhasorMessage(jnp.concatenate([s.data for s in scores], axis=-1)),
            predicted_distributions=frozendict(predicted_distributions),
        )
