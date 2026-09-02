import jax.numpy as jnp
from efax import Flattener, UnitVarianceNormalNP

from cem.phasor.frequency import geometric_frequencies
from cem.phasor.message import phasor_from_distribution
from cem.phasor.particle import (
    ParticleInference,
    initialize_particles,
    refine_best_particle,
    update_particles,
)

_DOMAIN_LOWER = -5.0
_DOMAIN_UPPER = 5.0
_MINIMUM_INITIAL_SPAN = 8.0
_NEAR_TARGET = 0.5


def _flattener() -> Flattener[UnitVarianceNormalNP]:
    flattener, _ = Flattener.flatten(UnitVarianceNormalNP(jnp.asarray(0.0)), mapped_to_plane=True)
    return flattener


def test_initialize_particles_spans_domain() -> None:
    particles = initialize_particles(jnp.array([_DOMAIN_LOWER]), jnp.array([_DOMAIN_UPPER]), 10)
    assert particles.shape == (10, 1)
    assert jnp.all(particles > _DOMAIN_LOWER)
    assert jnp.all(particles < _DOMAIN_UPPER)
    assert jnp.ptp(particles[:, 0]) > _MINIMUM_INITIAL_SPAN


def test_update_particles_retains_population_spread() -> None:
    frequencies = geometric_frequencies(6)
    prediction = phasor_from_distribution(UnitVarianceNormalNP(jnp.asarray(3.0)), frequencies)
    lower = jnp.array([_DOMAIN_LOWER])
    upper = jnp.array([_DOMAIN_UPPER])
    initial = initialize_particles(lower, upper, 16)
    settings = ParticleInference(n_particles=16, population_steps=8, temperature=0.1)

    updated = update_particles(
        initial,
        prediction,
        _flattener(),
        frequencies,
        lower,
        upper,
        settings,
    )

    assert jnp.std(updated[:, 0]) > 1.0
    assert jnp.any(jnp.abs(updated[:, 0] - 3.0) < _NEAR_TARGET)


def test_temporary_readout_recovers_wrapped_observation_without_collapsing_particles() -> None:
    target = jnp.asarray(-4.56)
    frequencies = geometric_frequencies(4)
    prediction = phasor_from_distribution(UnitVarianceNormalNP(target), frequencies)
    lower = jnp.array([-5.0])
    upper = jnp.array([1.0])
    particles = initialize_particles(lower, upper, 32)
    settings = ParticleInference(n_particles=32, population_steps=8, readout_steps=64)
    updated = update_particles(
        particles,
        prediction,
        _flattener(),
        frequencies,
        lower,
        upper,
        settings,
    )
    persistent_copy = updated.copy()

    readout = refine_best_particle(
        updated,
        prediction,
        _flattener(),
        frequencies,
        lower,
        upper,
        settings,
    )

    assert jnp.allclose(readout[0], target, atol=2e-3)
    assert jnp.array_equal(updated, persistent_copy)
    assert jnp.std(updated[:, 0]) > 1.0
