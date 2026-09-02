from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
from efax import ComplexVonMisesNP, Flattener, NaturalParametrization
from jax import lax
from tjax import JaxArray, JaxRealArray, frozendict

from cem.phasor.message import JaxComplexArray, phasor_from_distribution


class ParticleInference(eqx.Module):
    """Settings for persistent observation-particle inference."""

    n_particles: int = eqx.field(static=True, default=32)
    population_steps: int = eqx.field(static=True, default=8)
    readout_steps: int = eqx.field(static=True, default=64)
    population_step_size: float = eqx.field(static=True, default=0.05)
    readout_step_size: float = eqx.field(static=True, default=0.05)
    temperature: float = eqx.field(static=True, default=0.05)


class ObservationParticleState(eqx.Module):
    """Persistent candidate observations, keyed by target field."""

    positions: frozendict[str, JaxRealArray]


def initialize_particles(
    lower: JaxRealArray,
    upper: JaxRealArray,
    n_particles: int,
) -> JaxRealArray:
    """Deterministically spread particles across a rectangular parameter domain."""
    assert lower.shape == upper.shape
    assert lower.ndim == 1
    assert n_particles > 1
    dimensions = lower.shape[0]
    particle_index = jnp.arange(n_particles, dtype=lower.dtype)[:, jnp.newaxis]
    dimension_index = jnp.arange(dimensions, dtype=lower.dtype)[jnp.newaxis, :]
    # Irrational offsets avoid placing every dimension on the same diagonal for d > 1.
    unit_positions = jnp.mod(
        (particle_index + 0.5) / n_particles + dimension_index * ((jnp.sqrt(5.0) - 1.0) / 2.0),
        1.0,
    )
    return lower[jnp.newaxis, :] + unit_positions * (upper - lower)[jnp.newaxis, :]


def observation_cross_entropy[NP: NaturalParametrization[Any, Any]](
    positions: JaxRealArray,
    prediction: JaxComplexArray,
    flattener: Flattener[NP],
    frequencies: JaxRealArray,
) -> JaxRealArray:
    """Return the candidate-dependent part of each observation's cross-entropy.

    The omitted predicted-distribution log normalizer is constant across particles, so this
    quantity has exactly the same observation score and particle ranking as full cross-entropy.
    """
    observations = flattener.unflatten(positions)
    observation_phasors = phasor_from_distribution(observations, frequencies)
    observation_mean = ComplexVonMisesNP(observation_phasors).to_exp().mean
    predicted = jnp.broadcast_to(prediction, observation_phasors.shape)
    return -jnp.mean(jnp.real(observation_mean * jnp.conj(predicted)), axis=-1)


def update_particles[NP: NaturalParametrization[Any, Any]](
    state: JaxRealArray,
    prediction: JaxComplexArray,
    flattener: Flattener[NP],
    frequencies: JaxRealArray,
    lower: JaxRealArray,
    upper: JaxRealArray,
    settings: ParticleInference,
) -> JaxRealArray:
    """Move persistent particles by observation-score attraction and kernel repulsion."""
    scale = jnp.maximum(upper - lower, jnp.finfo(state.dtype).eps)

    def step(_: int, positions: JaxRealArray) -> JaxRealArray:
        def total_cross_entropy(candidate_positions: JaxRealArray) -> JaxArray:
            return jnp.sum(
                observation_cross_entropy(
                    candidate_positions,
                    prediction,
                    flattener,
                    frequencies,
                )
            )

        cross_entropy_gradient = jax.grad(total_cross_entropy)(positions)
        normalized = (positions - lower[jnp.newaxis, :]) / scale[jnp.newaxis, :]
        differences = normalized[:, jnp.newaxis, :] - normalized[jnp.newaxis, :, :]
        squared_distances = jnp.sum(jnp.square(differences), axis=-1)
        bandwidth = jnp.maximum(
            jnp.median(squared_distances) / jnp.log(settings.n_particles + 1.0),
            jnp.asarray(1e-4, dtype=state.dtype),
        )
        kernel = jnp.exp(-squared_distances / bandwidth)

        attraction = -(kernel.T @ cross_entropy_gradient)
        kernel_gradient = (
            -2.0
            * kernel[..., jnp.newaxis]
            * differences
            / (bandwidth * scale[jnp.newaxis, jnp.newaxis, :])
        )
        repulsion = jnp.sum(kernel_gradient, axis=0)
        velocity = (attraction + settings.temperature * repulsion) / settings.n_particles
        return jnp.clip(
            positions + settings.population_step_size * velocity,
            lower[jnp.newaxis, :],
            upper[jnp.newaxis, :],
        )

    return lax.fori_loop(0, settings.population_steps, step, state)


def refine_best_particle[NP: NaturalParametrization[Any, Any]](
    particles: JaxRealArray,
    prediction: JaxComplexArray,
    flattener: Flattener[NP],
    frequencies: JaxRealArray,
    lower: JaxRealArray,
    upper: JaxRealArray,
    settings: ParticleInference,
) -> JaxRealArray:
    """Refine a temporary copy of the best particle without reducing persistent diversity."""
    losses = observation_cross_entropy(particles, prediction, flattener, frequencies)
    position = particles[jnp.argmin(losses)]
    first_moment = jnp.zeros_like(position)
    second_moment = jnp.zeros_like(position)

    def step(
        iteration: int,
        carry: tuple[JaxRealArray, JaxRealArray, JaxRealArray],
    ) -> tuple[JaxRealArray, JaxRealArray, JaxRealArray]:
        current, first, second = carry

        def loss(candidate: JaxRealArray) -> JaxArray:
            return observation_cross_entropy(
                candidate[jnp.newaxis, :],
                prediction,
                flattener,
                frequencies,
            )[0]

        gradient = jax.grad(loss)(current)
        first = 0.9 * first + 0.1 * gradient
        second = 0.999 * second + 0.001 * jnp.square(gradient)
        step_number = jnp.asarray(iteration + 1, dtype=current.dtype)
        corrected_first = first / (1.0 - jnp.power(0.9, step_number))
        corrected_second = second / (1.0 - jnp.power(0.999, step_number))
        current -= (
            settings.readout_step_size * corrected_first / (jnp.sqrt(corrected_second) + 1e-8)
        )
        return jnp.clip(current, lower, upper), first, second

    refined, _, _ = lax.fori_loop(
        0,
        settings.readout_steps,
        step,
        (position, first_moment, second_moment),
    )
    return refined
