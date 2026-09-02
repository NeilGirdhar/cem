from collections.abc import Mapping
from typing import Any, Self

import jax.numpy as jnp
from efax import ExpectationParametrization, Flattener, HasEntropyEP, NaturalParametrization
from jax.lax import stop_gradient
from tjax import JaxArray, JaxRealArray, copy_cotangent, frozendict

from cem.phasor.input_node import PhasorInputConfiguration
from cem.phasor.loss import spectral_reconstruction_loss_and_score
from cem.phasor.message import JaxComplexArray, phasor_from_distribution
from cem.phasor.particle import (
    ObservationParticleState,
    ParticleInference,
    initialize_particles,
    refine_best_particle,
    update_particles,
)
from cem.structure.graph import FixedParameter
from cem.structure.graph.node import TargetConfiguration, TargetNode


class PhasorTargetConfiguration(PhasorInputConfiguration, TargetConfiguration):
    """Scored phasor targets, keyed by field name."""

    score: JaxComplexArray
    spectral_loss: frozendict[str, JaxArray]
    particle_state: ObservationParticleState

    def total_spectral_loss(self) -> JaxArray:
        """Return spectral objective telemetry summed across semantic fields."""
        return sum((jnp.sum(v) for v in self.spectral_loss.values()), start=jnp.asarray(0.0))


class PhasorTargetNode(TargetNode):
    """Target node scoring phasors and decoding them with persistent observation particles.

    The reported distributional loss is KL(observed || predicted), but the spectral reconstruction
    loss drives the gradient, which is better conditioned.

    Attributes:
        field_sizes: Per-field phasor dimension ``m * d``, used to split the incoming
            concatenated prediction.
        frequencies: Geometric frequency grid forwarded to
            :func:`~cem.phasor.message.phasor_from_distribution`.
        particle_bounds: Lower and upper flattened observation-parameter bounds per field.
        particle_inference: Static particle update and readout settings.
    """

    frequencies: FixedParameter[JaxRealArray]
    particle_bounds: FixedParameter[frozendict[str, tuple[JaxRealArray, JaxRealArray]]]
    particle_inference: ParticleInference

    @classmethod
    def create(
        cls,
        field_defaults: Mapping[str, NaturalParametrization[Any, Any]],
        frequencies: JaxRealArray,
        *,
        particle_bounds: Mapping[str, tuple[JaxRealArray, JaxRealArray]] | None = None,
        particle_inference: ParticleInference | None = None,
    ) -> Self:
        """Construct a PhasorTargetNode from distribution priors and a frequency grid.

        Args:
            field_defaults: Per-field prior distributions, in the order they will be
                split from the incoming concatenated prediction.
            frequencies: Geometric frequency grid, shape ``(m,)``.
            particle_bounds: Optional lower and upper bounds in each field's flattened,
                mapped-to-plane coordinates. Defaults to one highest-frequency period centred
                on the field default.
            particle_inference: Particle update and temporary-readout settings.

        Returns:
            A new :class:`PhasorTargetNode`.
        """
        assert len(field_defaults) > 0, "PhasorTargetNode requires at least one field"
        assert frequencies.ndim == 1
        flatteners_param, flat_sizes = cls._build_flatteners(field_defaults)
        phasor_defaults: dict[str, JaxComplexArray] = {}
        for field_name, dist in field_defaults.items():
            phasor_defaults[field_name] = phasor_from_distribution(dist, frequencies)
        if particle_bounds is None:
            half_width = jnp.pi / jnp.max(frequencies)
            particle_bounds = {}
            for field_name, dist in field_defaults.items():
                _, centre = Flattener.flatten(dist, mapped_to_plane=True)
                centre = centre.reshape(-1)
                particle_bounds[field_name] = (centre - half_width, centre + half_width)
        if set(particle_bounds) != set(field_defaults):
            msg = "particle bounds must have the same fields as field_defaults"
            raise ValueError(msg)
        checked_bounds: dict[str, tuple[JaxRealArray, JaxRealArray]] = {}
        for field_name, (lower, upper) in particle_bounds.items():
            expected_shape = (flat_sizes[field_name],)
            if lower.shape != expected_shape or upper.shape != expected_shape:
                msg = f"particle bounds for {field_name!r} must have shape {expected_shape}"
                raise ValueError(msg)
            if not bool(jnp.all(lower < upper)):
                msg = f"particle lower bounds for {field_name!r} must be below upper bounds"
                raise ValueError(msg)
            checked_bounds[field_name] = (lower, upper)
        field_sizes = frozendict(
            {field: phasor.shape[-1] for field, phasor in phasor_defaults.items()}
        )
        return cls(
            _flatteners=flatteners_param,
            field_sizes=field_sizes,
            frequencies=FixedParameter(frequencies),
            particle_bounds=FixedParameter(frozendict(checked_bounds)),
            particle_inference=particle_inference or ParticleInference(),
        )

    def initial_particle_state(self) -> ObservationParticleState:
        """Spread each field's persistent particles across its configured domain."""
        return ObservationParticleState(
            frozendict(
                {
                    field_name: initialize_particles(
                        lower,
                        upper,
                        self.particle_inference.n_particles,
                    )
                    for field_name, (lower, upper) in self.particle_bounds.value.items()
                }
            )
        )

    def infer(
        self,
        flat_observed: frozendict[str, JaxRealArray],
        prediction: JaxComplexArray,
        particle_state: ObservationParticleState | None = None,
    ) -> PhasorTargetConfiguration:
        """Compute reconstruction loss between observations and a predicted phasor.

        Args:
            flat_observed: Per-field flat distribution encodings
                (``mapped_to_plane=True`` coordinates).
            prediction: Concatenated prediction phasor for all fields.
            particle_state: Persistent candidate observations from the preceding time step.

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
        particle_positions: dict[str, JaxRealArray] = {}

        if particle_state is None:
            particle_state = self.initial_particle_state()

        for field_name, z_hat, observed_np, observed_exp in self._iter_target_fields(
            flat_observed, prediction
        ):
            obs_phasor = phasor_from_distribution(observed_np, self.frequencies.value)
            phasors[field_name] = obs_phasor
            spectral = spectral_reconstruction_loss_and_score(obs_phasor, z_hat)
            flattener = self._flatteners.value[field_name]
            lower, upper = self.particle_bounds.value[field_name]
            detached_prediction = stop_gradient(z_hat)
            positions = update_particles(
                particle_state.positions[field_name],
                detached_prediction,
                flattener,
                self.frequencies.value,
                lower,
                upper,
                self.particle_inference,
            )
            particle_positions[field_name] = positions
            readout = refine_best_particle(
                positions,
                detached_prediction,
                flattener,
                self.frequencies.value,
                lower,
                upper,
                self.particle_inference,
            )
            predicted_np = flattener.unflatten(readout)
            predicted_exp = predicted_np.to_exp()
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
            particle_state=ObservationParticleState(frozendict(particle_positions)),
        )
