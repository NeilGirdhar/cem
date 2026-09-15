import jax
import jax.numpy as jnp
import jax.random as jr
from tjax import create_streams

from cem.sip import SIPTDError, rollout_td_error, run_td_error_benchmark, simulate_remaining_food


def test_remaining_food_trajectories_start_at_one_and_evolve() -> None:
    """The remaining food reward starts at one and follows unrestricted noisy bites."""
    trajectories = simulate_remaining_food(count=8, n=10, seed=31)
    assert trajectories.shape == (8, 11)
    assert jnp.allclose(trajectories[:, 0], jnp.ones((8,)))
    assert jnp.any(trajectories < 0.0)
    assert jnp.any(trajectories[:, 1:] > trajectories[:, :-1])


def _td_error(
    *,
    predictor_observation_features: int = 3,
    predictor_instrument_features: int = 2,
    observation_features: int = 4,
    discount: float = 1.0,
) -> SIPTDError:
    return SIPTDError.create(
        predictor_observation_features=predictor_observation_features,
        predictor_instrument_features=predictor_instrument_features,
        observation_features=observation_features,
        discount=discount,
        hidden_features=6,
        streams=create_streams({"parameters": jr.key(20), "inference": jr.key(21)}),
    )


def _inputs() -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    return (
        jnp.array([0.2, -0.3, 0.5, 0.1]),
        jnp.array([0.4, -0.2, 0.3]),
        jnp.array([0.1, 0.6]),
        jnp.array([0.75]),
    )


def test_td_error_outputs_reconstruction_and_confounding_terms() -> None:
    """The TD-error circuit derives its two errors from its score and delayed witness.

    Its score, reconstruction loss, and confounding error follow the same formulas as
    SIPScore's, but from the delayed prediction and witness rather than the current
    ones. This does not independently verify the bootstrap target itself.
    """
    td_error = _td_error()
    delayed_prediction = jnp.array([0.05, -0.1, 0.2, 0.0])
    delayed_witness = jnp.array([0.3, -0.4, 0.1, 0.2])
    output = td_error.infer(
        *_inputs(),
        delayed_prediction,
        delayed_witness,
        streams=create_streams({"inference": jr.key(22)}),
        inference=True,
    )
    assert output.observation_score.shape == (4,)
    assert output.witness.shape == (4,)
    assert output.delayed_prediction.shape == (4,)
    assert output.delayed_witness.shape == (4,)
    assert jnp.allclose(output.witness, delayed_witness)
    assert jnp.allclose(output.reconstruction_loss, 0.5 * jnp.sum(output.observation_score**2))
    assert jnp.allclose(
        output.confounding_error,
        jnp.sum(output.observation_score * delayed_witness),
    )


def test_bootstrap_target_stops_the_current_prediction() -> None:
    """The bootstrap target must not carry gradient into the current prediction.

    The reconstruction loss must have zero gradient with respect to the predictor
    observations feeding the current prediction, since that prediction enters the
    bootstrap target only through a stopped copy. Only the delayed prediction and
    witness, threaded in from a previous step, may train this step's loss.
    """
    td_error = _td_error()
    observation, predictor_observations, predictor_instruments, gain = _inputs()
    delayed_prediction = jnp.array([0.05, -0.1, 0.2, 0.0])
    delayed_witness = jnp.array([0.3, -0.4, 0.1, 0.2])
    streams = create_streams({"inference": jr.key(23)})

    def reconstruction_loss(current_predictor_observations: jnp.ndarray) -> jnp.ndarray:
        output = td_error.infer(
            observation,
            current_predictor_observations,
            predictor_instruments,
            gain,
            delayed_prediction,
            delayed_witness,
            streams=streams,
            inference=True,
        )
        return output.reconstruction_loss

    gradient = jax.grad(reconstruction_loss)(predictor_observations)
    assert jnp.allclose(gradient, jnp.zeros_like(gradient))


def test_zero_gain_forces_a_terminal_boundary() -> None:
    """Zero gain must zero the current prediction, giving a terminal p = 0.

    A terminal step's bootstrap target then reduces to the observation alone, and
    the delayed state carried past it must vanish.
    """
    td_error = _td_error()
    observation, predictor_observations, predictor_instruments, _ = _inputs()
    delayed_prediction = jnp.array([0.05, -0.1, 0.2, 0.0])
    delayed_witness = jnp.array([0.3, -0.4, 0.1, 0.2])
    output = td_error.infer(
        observation,
        predictor_observations,
        predictor_instruments,
        jnp.array([0.0]),
        delayed_prediction,
        delayed_witness,
        streams=create_streams({"inference": jr.key(24)}),
        inference=True,
    )
    assert jnp.allclose(output.observation_score, delayed_prediction - observation)
    assert jnp.allclose(output.delayed_prediction, jnp.zeros_like(delayed_prediction))
    assert jnp.allclose(output.delayed_witness, jnp.zeros_like(delayed_witness))


def test_rollout_telescopes_regardless_of_intermediate_predictions() -> None:
    """The summed score across an untrained rollout must telescope exactly.

    With discount 1 and a zero-gain terminal step, the sum of every step's score
    must equal the initial delayed prediction minus the summed observations, no
    matter how the untrained intermediate predictions and witnesses come out. This
    is the algebraic guarantee behind @temporal-baseline, independent of training.
    """
    horizon, count, features = 4, 5, 2
    td_error = _td_error(
        predictor_observation_features=3,
        predictor_instrument_features=2,
        observation_features=features,
    )
    key = jr.key(30)
    observations = jr.normal(jr.fold_in(key, 0), (horizon, count, features))
    predictor_observations = jr.normal(jr.fold_in(key, 1), (horizon, count, 3))
    predictor_instruments = jr.normal(jr.fold_in(key, 2), (horizon, count, 2))
    gains = jnp.concatenate(
        [jnp.ones((horizon - 1, count, 1)), jnp.zeros((1, count, 1))],
        axis=0,
    )
    initial_state = jr.normal(jr.fold_in(key, 3), (count, 3))
    initial_instrument = jr.normal(jr.fold_in(key, 4), (count, 2))
    streams = create_streams({"inference": jr.fold_in(key, 5)})

    outputs = rollout_td_error(
        td_error,
        observations,
        predictor_observations,
        predictor_instruments,
        gains,
        initial_state,
        initial_instrument,
        streams=streams,
        inference=True,
    )
    initial_prediction = td_error.predictor.prediction(
        initial_state,
        jnp.ones((count, 1)),
        streams=streams,
        inference=True,
    )
    total_score = sum(output.observation_score for output in outputs)
    total_observation = jnp.sum(observations, axis=0)
    assert jnp.allclose(total_score, initial_prediction - total_observation, atol=1e-4)


def test_td_error_records_link_strength_and_expected_error() -> None:
    """The remaining-food benchmark records both requested training diagnostics."""
    result = run_td_error_benchmark(count=32, steps=8)
    trajectory = result["td"].trajectory
    assert trajectory is not None
    assert len(trajectory.training_examples) == len(trajectory.link_strengths)
    assert len(trajectory.training_examples) == len(trajectory.expected_td_errors)
    assert trajectory.training_examples[0] == 0
