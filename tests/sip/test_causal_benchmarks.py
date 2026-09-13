import jax.numpy as jnp

from cem.sip import run_action_sensation_benchmark, run_injected_action_noise_benchmark

_ACTION_EFFECT_TOLERANCE = 0.01
_INJECTED_NOISE_EFFECT_TOLERANCE = 0.05
_RECONSTRUCTION_LOSS_TOLERANCE = 0.01
_ZERO_NOISE_EFFECT_ERROR_MINIMUM = 0.2


def test_action_sensation_benchmark_recovers_effect() -> None:
    """Recover an action-to-sensation effect from an instrumented action.

    Action combines a base value with an exogenous instrument, and
    sensation has a true action coefficient of 1.7. Increasing action by one must
    change the learned prediction by that amount within tolerance. The outcome has no
    action confounder, so ordinary regression could also recover this effect.
    """
    result = run_action_sensation_benchmark(count=64, steps=480)

    assert abs(result.estimated_effect - result.true_effect) < _ACTION_EFFECT_TOLERANCE
    assert jnp.isfinite(result.residual_instrument_covariance)
    assert jnp.isfinite(result.reconstruction_loss)


def test_injected_action_noise_identifies_effect() -> None:
    """Injected action noise identifies an effect that prediction alone cannot.

    Both conditions generate action X = 0.9 U + Z and future sensation
    Y = 1.7 X + 2.2 U + noise. The score receives past sensation U and action X as
    predictor observations, and the same injected noise Z as the instrument. The
    conditions share their samples and initial parameters; only the magnitude of Z
    changes.

    The zero-noise condition sets Z = 0, making X collinear with U. Reconstruction can
    then determine their combined contribution to Y but cannot determine how much
    either variable caused it. The random-noise condition varies X independently of U
    and carries that variation through the instrument channel, making the action
    coefficient identifiable.

    Both conditions must reconstruct accurately. The zero-noise estimate must remain
    wrong, while the random-noise estimate must recover the true action coefficient of
    1.7. Because the comparison changes both action variation and the instrument
    input, it tests the complete noise-injection condition rather than the instrument
    channel in isolation.
    """
    results = run_injected_action_noise_benchmark(count=64, steps=960)
    zero = results["zero"]
    random = results["random"]
    zero_error = abs(zero.estimated_effect - zero.true_effect)
    random_error = abs(random.estimated_effect - random.true_effect)

    assert zero_error > _ZERO_NOISE_EFFECT_ERROR_MINIMUM
    assert random_error < _INJECTED_NOISE_EFFECT_TOLERANCE
    assert random_error < zero_error
    for result in results.values():
        assert result.reconstruction_loss < _RECONSTRUCTION_LOSS_TOLERANCE
        assert jnp.isfinite(result.residual_instrument_covariance)
