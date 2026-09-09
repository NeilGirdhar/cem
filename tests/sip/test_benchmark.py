import jax.numpy as jnp

from cem.sip import run_synthetic_sip_benchmark


def test_synthetic_benchmark_returns_all_conditions() -> None:
    results = run_synthetic_sip_benchmark(count=8, steps=3)

    assert set(results) == {"ordinary", "intervention", "purified"}
    for result in results.values():
        assert result.noise_magnitudes
        assert all(jnp.isfinite(value) for value in result.noise_magnitudes)
        assert jnp.isfinite(result.training_loss)
        assert jnp.isfinite(result.inference_loss)
        assert jnp.isfinite(result.residual_instrument_correlation)
        assert jnp.isfinite(result.witness_loss)
