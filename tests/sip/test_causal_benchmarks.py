import jax.numpy as jnp

from cem.sip import run_action_sensation_benchmark, run_confounded_action_benchmark

_ACTION_EFFECT_TOLERANCE = 0.3
_CONFOUNDED_EFFECT_TOLERANCE = 0.6


def test_action_sensation_benchmark_recovers_effect() -> None:
    """Recover an action-to-sensation effect from an instrumented action.

    Action combines an endogenous base value with an exogenous instrument, and
    sensation has a true action coefficient of 1.7. Increasing action by one must
    change the learned prediction by that amount within tolerance. The outcome has no
    action confounder, so ordinary regression could also recover this effect.
    """
    result = run_action_sensation_benchmark(count=64, steps=120)

    assert abs(result.estimated_effect - result.true_effect) < _ACTION_EFFECT_TOLERANCE
    assert jnp.isfinite(result.residual_instrument_covariance)
    assert jnp.isfinite(result.reconstruction_loss)


def test_confounded_action_benchmark_recovers_effect() -> None:
    """Recover the direct action effect when past sensation is a common cause.

    Past sensation affects both action and target sensation, while an exogenous
    instrument also affects action. The learned response to a unit action increase
    must recover the true coefficient of 1.7 within tolerance. Past sensation is
    observed by the predictor, so the benchmark does not test an unknown confounder.
    """
    result = run_confounded_action_benchmark(count=64, steps=180)

    assert abs(result.estimated_effect - result.true_effect) < _CONFOUNDED_EFFECT_TOLERANCE
    assert jnp.isfinite(result.residual_instrument_covariance)
    assert jnp.isfinite(result.reconstruction_loss)
