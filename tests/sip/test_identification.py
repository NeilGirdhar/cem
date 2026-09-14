import jax.numpy as jnp
import pytest

from cem.sip import run_direct_injection_benchmark, run_inherited_instrument_benchmark


def test_direct_injection_identifies_intention_effect() -> None:
    """Direct injection identifies intention A's confounded effect on Y.

    Past sensation X affects both A and future sensation Y. The score for Y receives
    predictor observations X and A together with instrument(A). The target coefficient
    of A in Y is 1.7.

    Both conditions share their samples and initial parameters. The zero-noise
    condition makes instrument(A) zero and leaves A collinear with X. The random-noise
    condition supplies variation in A independent of X and records it in instrument(A).

    Both conditions must reconstruct Y accurately. Without exogenous variation, the
    estimated effect must remain wrong. With random injection, it must recover the true
    effect and improve on the zero-noise estimate.
    """
    maximum_reconstruction_loss = 0.01
    minimum_unidentified_error = 0.2
    identified_effect_tolerance = 0.05
    results = run_direct_injection_benchmark(count=64, steps=960)
    zero = results["zero"]
    random = results["random"]
    zero_error = abs(zero.estimated_effect - zero.true_effect)
    random_error = abs(random.estimated_effect - random.true_effect)

    # Reconstruction succeeds in both conditions.
    for result in results.values():
        assert result.reconstruction_loss < maximum_reconstruction_loss
        assert jnp.isfinite(result.residual_instrument_covariance)
        assert result.trajectory is not None
        assert result.trajectory.training_examples[0] == 0
        assert result.trajectory.training_examples[-1] == 64 * 960
        assert result.trajectory.estimated_effects[-1] == pytest.approx(result.estimated_effect)
        assert result.trajectory.reconstruction_losses[-1] == pytest.approx(
            result.reconstruction_loss
        )

    # Without exogenous variation, the effect remains unidentified.
    assert zero_error > minimum_unidentified_error

    # Direct injection identifies the effect.
    assert random.estimated_effect == pytest.approx(
        random.true_effect,
        abs=identified_effect_tolerance,
    )
    assert random_error < zero_error


def test_inherited_instrument_identifies_sensation_effect() -> None:
    """Instrument inheritance identifies sensation Y's confounded effect on Z.

    Past sensation X affects future sensation Y and subsequent sensation Z, while
    intention A also affects Y. The score for Z receives predictor observations X and
    Y together with instrument(Y). Although Y is observed, its emitter must learn
    instrument(Y) from instrument(A). The target coefficient of Y in Z is 1.7.

    The conditions share their samples and initial parameters. The policy condition sets
    A = f(X) while keeping instrument(A) zero. The injected condition adds noise to A and
    records it in instrument(A), supplying variation that reaches Z through Y.

    Every condition must reconstruct Z accurately. Policy must fail to separate the effects
    of X and Y, while injection must learn instrument(Y) and recover Y's causal effect on Z.
    """
    maximum_reconstruction_loss = 0.01
    minimum_unidentified_error = 0.1
    identified_effect_tolerance = 0.05
    first_stage_effect_tolerance = 0.15
    first_stage_covariance_tolerance = 0.001
    results = run_inherited_instrument_benchmark(count=64, steps=960)
    policy = results["policy"]
    injected = results["injected"]
    policy_error = abs(policy.estimated_effect - policy.true_effect)
    injected_error = abs(injected.estimated_effect - injected.true_effect)

    # Reconstruction succeeds in every condition.
    for result in results.values():
        assert result.reconstruction_loss < maximum_reconstruction_loss
        assert jnp.isfinite(result.residual_instrument_covariance)
        assert result.trajectory is not None
        assert result.trajectory.training_examples[0] == 0
        assert result.trajectory.training_examples[-1] == 64 * 960
        assert result.trajectory.instrument_magnitudes
        assert result.trajectory.instrument_magnitudes

    # Accurate reconstruction alone cannot separate the effects of X and Y.
    assert policy_error > minimum_unidentified_error

    # instrument(Y) identifies the effect of Y on Z.
    assert injected.estimated_effect == pytest.approx(
        injected.true_effect,
        abs=identified_effect_tolerance,
    )
    assert injected_error < policy_error

    # The first stage learns instrument(Y) from instrument(A).
    assert injected.true_first_stage_effect is not None
    assert injected.estimated_first_stage_effect is not None
    assert injected.estimated_first_stage_effect == pytest.approx(
        injected.true_first_stage_effect,
        abs=first_stage_effect_tolerance,
    )
    assert injected.first_stage_residual_covariance is not None
    assert abs(injected.first_stage_residual_covariance) < first_stage_covariance_tolerance
