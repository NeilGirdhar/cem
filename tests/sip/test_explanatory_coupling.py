import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
from tjax import create_streams

from cem.sip import ExplanatoryCoupling, train_explanatory_coupling_adversarial


def _coupling() -> ExplanatoryCoupling:
    return ExplanatoryCoupling.create(
        innovation_features=2,
        goal_features=1,
        parent_instrument_features=2,
        source_features=3,
        target_features=2,
        hidden_features=5,
        initial_noise=0.1,
        streams=create_streams({"parameters": jr.key(20), "inference": jr.key(21)}),
    )


def _inputs() -> tuple[jnp.ndarray, ...]:
    return (
        jnp.array([0.2, -0.4]),
        jnp.array([0.3]),
        jnp.array([0.8]),
        jnp.array([0.1, -0.2]),
        jnp.array([0.5, -0.1]),
        jnp.array([0.7]),
    )


def test_injected_noise_reaches_downstream_prediction() -> None:
    """Injected source noise reaches the target score's prediction.

    With fixed inputs and parameters, different runtime keys must produce different
    injected noise, source observations, and downstream predictions. The test does
    not inspect propagation through the instrument or witness path.
    """
    coupling = _coupling()
    inputs = _inputs()
    first = coupling.infer(
        *inputs,
        streams=create_streams({"inference": jr.key(24)}),
        inference=False,
    )
    second = coupling.infer(
        *inputs,
        streams=create_streams({"inference": jr.key(25)}),
        inference=False,
    )

    assert not jnp.allclose(first.source.injected_noise, second.source.injected_noise)
    assert not jnp.allclose(first.source.observation, second.source.observation)
    assert not jnp.allclose(first.target.prediction, second.target.prediction)


def test_coupling_is_deterministic_without_injected_noise() -> None:
    """Inference is deterministic when the source injects no local noise.

    Different runtime keys must produce identical source observations, source
    instruments, and target predictions. Together with the preceding test, this
    localizes the coupling's stochasticity to training-time noise injection.
    """
    coupling = _coupling()
    inputs = _inputs()
    first = coupling.infer(
        *inputs,
        streams=create_streams({"inference": jr.key(26)}),
        inference=True,
    )
    second = coupling.infer(
        *inputs,
        streams=create_streams({"inference": jr.key(27)}),
        inference=True,
    )

    assert jnp.allclose(first.source.observation, second.source.observation)
    assert jnp.allclose(first.source.instrument, second.source.instrument)
    assert jnp.allclose(first.target.prediction, second.target.prediction)


def test_fixed_injected_noise_improves_coupling_shifted_error() -> None:
    """Fixed injected noise can improve robustness to a predictor shift.

    Training provides a stable noisy feature and an accurate nuisance feature that
    becomes independent noise at evaluation. A coupling trained with fixed injected
    noise must then outperform one trained with nearly zero noise. Because training
    uses reconstruction loss alone, this tests noise regularization, not witness-based
    purification.
    """
    key = jr.key(37)
    signal_key, stable_key, nuisance_key, shift_key = jr.split(key, 4)
    count = 32
    signal = jr.normal(signal_key, (count,))
    stable = signal + 0.5 * jr.normal(stable_key, (count,))
    nuisance = signal + 0.05 * jr.normal(nuisance_key, (count,))
    training_innovation = jnp.stack((stable, nuisance), axis=-1)
    shifted_innovation = jnp.stack((stable, jr.normal(shift_key, (count,))), axis=-1)
    goal = jnp.zeros((count, 1))
    parent_instruments = jnp.zeros((count, 1))
    gain = jnp.ones((count, 1))
    target = signal[:, jnp.newaxis]
    noise_key = jr.key(38)

    def train(initial_noise: float) -> ExplanatoryCoupling:
        coupling = ExplanatoryCoupling.create(
            innovation_features=2,
            goal_features=1,
            parent_instrument_features=1,
            source_features=2,
            target_features=1,
            hidden_features=(),
            initial_noise=initial_noise,
            learn_noise=False,
            streams=create_streams({"parameters": jr.key(39), "inference": jr.key(40)}),
        )
        for step in range(80):

            def loss(current: ExplanatoryCoupling, update_step: int) -> jnp.ndarray:
                result = current.infer(
                    training_innovation,
                    goal,
                    gain,
                    parent_instruments,
                    target,
                    gain,
                    streams=create_streams({"inference": jr.fold_in(noise_key, update_step)}),
                    inference=False,
                )
                return jnp.mean(result.target.reconstruction_loss)

            _, gradients = eqx.filter_value_and_grad(loss)(coupling, step)
            coupling = eqx.apply_updates(
                coupling,
                jax.tree.map(lambda value: -0.02 * value, gradients),
            )
        return coupling

    def shifted_loss(coupling: ExplanatoryCoupling) -> jnp.ndarray:
        result = coupling.infer(
            shifted_innovation,
            goal,
            gain,
            parent_instruments,
            target,
            gain,
            streams=create_streams({"inference": jr.key(41)}),
            inference=True,
        )
        return jnp.mean(result.target.reconstruction_loss)

    clean_error = shifted_loss(train(1e-6))
    noisy_error = shifted_loss(train(0.6))
    assert noisy_error < clean_error


def test_purification_improves_coupling_under_shifted_confounding() -> None:
    """Purification improves an explanatory coupling under shifted confounding.

    The source contains a stable noisy feature and an instrument-contaminated nuisance
    feature that becomes independent noise at evaluation. A coupling trained with the
    witness and confounding objective must have lower shifted reconstruction loss than
    one trained for reconstruction alone.
    """
    count = 32
    signal = jr.normal(jr.key(1), (count,))
    clean_noise = jr.normal(jr.key(2), (count,))
    instrument = jr.normal(jr.key(3), (count,))
    training_innovation = jnp.stack(
        (signal + 0.5 * clean_noise, signal + 0.1 * instrument),
        axis=-1,
    )
    shifted_innovation = jnp.stack(
        (signal + 0.5 * clean_noise, jr.normal(jr.key(8), (count,))),
        axis=-1,
    )
    goal = jnp.zeros((count, 1))
    parent_instruments = instrument[:, jnp.newaxis]
    gain = jnp.ones((count, 1))
    target = signal[:, jnp.newaxis]

    def train(*, adversarial: bool) -> ExplanatoryCoupling:
        coupling = ExplanatoryCoupling.create(
            innovation_features=2,
            goal_features=1,
            parent_instrument_features=1,
            source_features=2,
            target_features=1,
            hidden_features=(),
            initial_noise=0.05,
            learn_noise=False,
            streams=create_streams({"parameters": jr.key(4), "inference": jr.key(5)}),
        )
        trained, _ = train_explanatory_coupling_adversarial(
            coupling,
            training_innovation,
            goal,
            gain,
            parent_instruments,
            target,
            gain,
            steps=100,
            predictor_learning_rate=0.005,
            witness_learning_rate=0.005 if adversarial else 0.0,
            confounding_weight=2.0 if adversarial else 0.0,
            streams=create_streams({"inference": jr.key(6)}),
        )
        return trained

    def shifted_loss(coupling: ExplanatoryCoupling) -> jnp.ndarray:
        output = coupling.infer(
            shifted_innovation,
            goal,
            gain,
            parent_instruments,
            target,
            gain,
            streams=create_streams({"inference": jr.key(7)}),
            inference=True,
        )
        return jnp.mean(output.target.reconstruction_loss)

    ordinary_error = shifted_loss(train(adversarial=False))
    adversarial_error = shifted_loss(train(adversarial=True))
    assert adversarial_error < ordinary_error


def test_witness_only_update_leaves_predictor_path_unchanged() -> None:
    """Witness-only learning leaves the emitter and prediction map unchanged.

    The witness learning rate is positive and the predictor learning rate is zero.
    This checks gradient isolation between witness maximization and prediction
    purification. It does not verify that the witness parameters actually change.
    """
    count = 8
    innovation = jr.normal(jr.key(76), (count, 2))
    goal = jnp.zeros((count, 1))
    gain = jnp.ones((count, 1))
    parent_instruments = jr.normal(jr.key(77), (count, 1))
    target = innovation[:, :1]
    initial = ExplanatoryCoupling.create(
        innovation_features=2,
        goal_features=1,
        parent_instrument_features=1,
        source_features=2,
        target_features=1,
        hidden_features=(),
        streams=create_streams({"parameters": jr.key(78), "inference": jr.key(79)}),
    )
    trained, _ = train_explanatory_coupling_adversarial(
        initial,
        innovation,
        goal,
        gain,
        parent_instruments,
        target,
        gain,
        steps=2,
        predictor_learning_rate=0.0,
        witness_learning_rate=0.01,
        streams=create_streams({"inference": jr.key(80)}),
    )

    assert all(
        jnp.allclose(before, after)
        for before, after in zip(
            jax.tree.leaves(initial.emitter),
            jax.tree.leaves(trained.emitter),
            strict=True,
        )
        if eqx.is_array(before)
    )
    assert all(
        jnp.allclose(before, after)
        for before, after in zip(
            jax.tree.leaves(initial.score.predictor.prediction_map),
            jax.tree.leaves(trained.score.predictor.prediction_map),
            strict=True,
        )
        if eqx.is_array(before)
    )
