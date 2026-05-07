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
from cem.phasor.message import JaxComplexArray, phasor_from_distribution, phasor_to_distribution
from cem.structure.graph import FixedParameter
from cem.structure.graph.node import TargetConfiguration, TargetNode


class PhasorTargetConfiguration(PhasorInputConfiguration, TargetConfiguration):
    """Scored phasor targets, keyed by field name."""

    score: JaxComplexArray
    spectral_loss: frozendict[str, JaxArray]

    def total_spectral_loss(self) -> JaxArray:
        """Return spectral objective telemetry summed across semantic fields."""
        return sum((jnp.sum(v) for v in self.spectral_loss.values()), start=jnp.asarray(0.0))


class PhasorTargetNode(TargetNode):
    """Target node computing reconstruction loss between observed and predicted distributions.

    The reported distributional loss is KL(observed || predicted), but the spectral reconstruction
    loss drives the gradient, which is better conditioned.

    Attributes:
        frequency_grids: Per-field frequency grid ``t``, shape ``(m * d,)`` each,
            used to recover expectation parameters from predicted phasors.
        field_sizes: Per-field phasor dimension ``m * d``, used to split the incoming
            concatenated prediction.
        frequencies: Geometric frequency grid forwarded to
            :func:`~cem.phasor.message.phasor_from_distribution`.
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
        phasor_defaults: dict[str, JaxComplexArray] = {}
        frequency_grids: dict[str, NaturalParametrization[Any, Any]] = {}
        for field_name, dist in field_defaults.items():
            phasor_defaults[field_name] = phasor_from_distribution(dist, frequencies)
            nat_flattener, _ = Flattener.flatten(dist, mapped_to_plane=False)
            frequency_grids[field_name] = make_frequency_grid(nat_flattener, frequencies)
        field_sizes = frozendict(
            {field: phasor.shape[-1] for field, phasor in phasor_defaults.items()}
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
        prediction: JaxComplexArray,
    ) -> PhasorTargetConfiguration:
        """Compute reconstruction loss between observations and a predicted phasor.

        Args:
            flat_observed: Per-field flat distribution encodings
                (``mapped_to_plane=True`` coordinates).
            prediction: Concatenated prediction phasor for all fields.

        Returns:
            A :class:`PhasorTargetConfiguration` with per-field losses, scores, and
            distributions.
        """
        phasors: dict[str, JaxComplexArray] = {}
        scores: list[JaxArray] = []
        losses: dict[str, JaxArray] = {}
        spectral_losses: dict[str, JaxArray] = {}
        observed_distributions: dict[str, ExpectationParametrization] = {}
        predicted_distributions: dict[str, HasEntropyEP] = {}

        for field_name, z_hat, observed_np, observed_exp in self._iter_target_fields(
            flat_observed, prediction
        ):
            grid = self.frequency_grids.value[field_name]
            obs_phasor = phasor_from_distribution(observed_np, self.frequencies.value)
            phasors[field_name] = obs_phasor
            spectral = spectral_reconstruction_loss_and_score(obs_phasor, z_hat)
            predicted_exp = phasor_to_distribution(z_hat, grid)
            assert isinstance(predicted_exp, HasEntropyEP)
            assert isinstance(predicted_exp, type(observed_exp))
            observed_distributions[field_name] = observed_exp
            scores.append(spectral.score)
            distributional_loss = observed_exp.kl_divergence(
                predicted_exp.to_nat(), self_nat=observed_np
            )
            spectral_losses[field_name] = spectral.loss
            # Report distributional loss but optimize spectral gradient for testing purposes.
            losses[field_name] = copy_cotangent(
                stop_gradient(distributional_loss),
                spectral.loss,
            )
            predicted_distributions[field_name] = predicted_exp

        return PhasorTargetConfiguration(
            values=frozendict(phasors),
            loss=frozendict(losses),
            spectral_loss=frozendict(spectral_losses),
            observed_distributions=frozendict(observed_distributions),
            score=jnp.concatenate(scores, axis=-1),
            predicted_distributions=frozendict(predicted_distributions),
        )
