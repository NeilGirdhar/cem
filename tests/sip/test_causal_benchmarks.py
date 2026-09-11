import jax.numpy as jnp

from cem.sip import run_action_sensation_benchmark, run_confounded_action_benchmark

_ACTION_EFFECT_TOLERANCE = 0.3
_CONFOUNDED_EFFECT_TOLERANCE = 0.6


def test_action_sensation_benchmark_recovers_effect() -> None:
    result = run_action_sensation_benchmark(count=64, steps=120)

    assert abs(result.estimated_effect - result.true_effect) < _ACTION_EFFECT_TOLERANCE
    assert jnp.isfinite(result.residual_instrument_covariance)
    assert jnp.isfinite(result.reconstruction_loss)


def test_confounded_action_benchmark_recovers_effect() -> None:
    result = run_confounded_action_benchmark(count=64, steps=180)

    assert abs(result.estimated_effect - result.true_effect) < _CONFOUNDED_EFFECT_TOLERANCE
    assert jnp.isfinite(result.residual_instrument_covariance)
    assert jnp.isfinite(result.reconstruction_loss)
