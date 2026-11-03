"""Reproducible synthetic benchmarks for real-valued SIP."""

from collections.abc import Callable
from dataclasses import dataclass, field, replace

import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
from tjax import create_streams

from cem.sip.emitter import SIPEmitter
from cem.sip.explanatory_coupling import ExplanatoryCoupling
from cem.sip.score import SIPScore
from cem.sip.td_error import SIPTDError
from cem.sip.training import (
    SIPTrainingHistory,
    rollout_td_error,
    train_explanatory_coupling_adversarial,
    train_instrument_map,
    train_score_adversarial,
    train_td_error_adversarial,
)


@dataclass(frozen=True)
class SIPBenchmarkResult:
    """Metrics from one synthetic SIP training condition."""

    training_loss: float
    inference_loss: float
    residual_instrument_correlation: float
    witness_loss: float
    noise_magnitudes: tuple[float, ...]


@dataclass(frozen=True)
class CausalBenchmarkTrajectory:
    """Causal-effect and reconstruction metrics recorded during training."""

    training_examples: tuple[int, ...]
    estimated_effects: tuple[float, ...]
    reconstruction_losses: tuple[float, ...]
    instrument_magnitudes: tuple[float, ...] = ()
    link_strengths: tuple[float, ...] = ()
    expected_td_errors: tuple[float, ...] = ()
    td_error_by_step: tuple[tuple[float, ...], ...] = ()
    td_error_mean_by_step: tuple[tuple[float, ...], ...] = ()


@dataclass(frozen=True)
class CreditBenchmarkResult:
    """Score-based credit a training signal would send to a persistent cause."""

    true_effect: float
    estimated_effect: float
    reconstruction_loss: float
    trajectory: CausalBenchmarkTrajectory | None = None


@dataclass(frozen=True)
class CausalBenchmarkResult:
    """Estimated causal effect and residual diagnostics for a SIP task."""

    true_effect: float
    estimated_effect: float
    residual_instrument_covariance: float
    reconstruction_loss: float
    true_first_stage_effect: float | None = None
    estimated_first_stage_effect: float | None = None
    first_stage_residual_covariance: float | None = None
    trajectory: CausalBenchmarkTrajectory | None = None


def simulate_remaining_food(
    *,
    count: int = 8,
    n: int = 10,
    variance: float = 0.01,
    seed: int = 400,
) -> jnp.ndarray:
    """Simulate zero-mean OU paths for the remaining food reward.

    The process starts at one unit and mean-reverts toward zero. Each Euler step
    combines the restoring drift with Gaussian innovation of the requested
    variance. The balance is unrestricted, so negative values represent reward
    debt. The returned array has shape ``(count, n + 1)`` and includes the initial
    balance in its first column.
    """
    if count < 1 or n < 1:
        msg = "count and n must be positive"
        raise ValueError(msg)
    if variance < 0.0:
        msg = "variance must be nonnegative"
        raise ValueError(msg)
    noises = jr.normal(jr.key(seed), (n, count))
    balance = jnp.ones((count,))
    trajectories = [balance]
    for step in range(n):
        sigma = jnp.sqrt(variance) * jnp.exp(-step / 4.0)
        balance += -0.2 * balance + sigma * noises[step]
        trajectories.append(balance)
    return jnp.stack(trajectories, axis=1)


def _data(count: int, *, key: jnp.ndarray) -> tuple[jnp.ndarray, ...]:
    signal_key, clean_key, instrument_key, shifted_key = jr.split(key, 4)
    signal = jr.normal(signal_key, (count,))
    clean_noise = jr.normal(clean_key, (count,))
    instrument = jr.normal(instrument_key, (count,))
    training_innovation = jnp.stack(
        (signal + 0.5 * clean_noise, signal + 0.1 * instrument),
        axis=-1,
    )
    shifted_innovation = jnp.stack(
        (signal + 0.5 * clean_noise, jr.normal(shifted_key, (count,))),
        axis=-1,
    )
    goal = jnp.zeros((count, 1))
    parent_instruments = instrument[:, jnp.newaxis]
    gain = jnp.ones((count, 1))
    target = signal[:, jnp.newaxis]
    return (
        training_innovation,
        shifted_innovation,
        goal,
        parent_instruments,
        gain,
        target,
    )


def _train_condition(
    *,
    initial_noise: float,
    confounding_weight: float,
    witness_learning_rate: float,
    data: tuple[jnp.ndarray, ...],
    steps: int,
    key: jnp.ndarray,
) -> tuple[ExplanatoryCoupling, SIPTrainingHistory]:
    training_innovation, _, goal, parent_instruments, gain, target = data
    coupling = ExplanatoryCoupling.create(
        innovation_features=2,
        goal_features=1,
        parent_instrument_features=1,
        source_features=2,
        target_features=1,
        hidden_features=(),
        initial_noise=initial_noise,
        learn_noise=False,
        streams=create_streams({"parameters": jr.fold_in(key, 0), "inference": jr.fold_in(key, 1)}),
    )
    return train_explanatory_coupling_adversarial(
        coupling,
        training_innovation,
        goal,
        gain,
        parent_instruments,
        target,
        gain,
        steps=steps,
        predictor_learning_rate=0.005,
        witness_learning_rate=witness_learning_rate,
        confounding_weight=confounding_weight,
        streams=create_streams({"inference": jr.fold_in(key, 2)}),
    )


def run_synthetic_sip_benchmark(
    *,
    count: int = 32,
    steps: int = 100,
    seed: int = 0,
) -> dict[str, SIPBenchmarkResult]:
    """Compare ordinary, noisy, and purified couplings under shifted confounding."""
    if count < 1 or steps < 1:
        msg = "count and steps must be positive"
        raise ValueError(msg)
    data = _data(count, key=jr.key(seed))
    _, shifted_innovation, goal, parent_instruments, gain, target = data
    conditions = {
        "ordinary": (1e-6, 0.0, 0.0),
        "intervention": (0.05, 0.0, 0.0),
        "purified": (0.05, 2.0, 0.005),
    }
    results: dict[str, SIPBenchmarkResult] = {}
    parameter_key = jr.key(seed + 1)
    for name, (noise, confounding, witness_rate) in conditions.items():
        coupling, history = _train_condition(
            initial_noise=noise,
            confounding_weight=confounding,
            witness_learning_rate=witness_rate,
            data=data,
            steps=steps,
            key=parameter_key,
        )
        output = coupling.infer(
            shifted_innovation,
            goal,
            gain,
            parent_instruments,
            target,
            gain,
            streams=create_streams({"inference": jr.key(seed + 2)}),
            inference=True,
        )
        results[name] = SIPBenchmarkResult(
            training_loss=float(history.purification_losses[-1]),
            inference_loss=float(jnp.mean(output.target.reconstruction_loss)),
            residual_instrument_correlation=float(
                jnp.square(
                    jnp.mean(output.target.observation_score[:, 0] * output.source.instrument[:, 0])
                )
            ),
            witness_loss=float(history.witness_losses[-1]),
            noise_magnitudes=tuple(float(value) for value in coupling.emitter.noise_magnitudes),
        )
    return results


def _fit_causal_score(
    predictor_observations: jnp.ndarray,
    instruments: jnp.ndarray,
    target: jnp.ndarray,
    *,
    key: jnp.ndarray,
    steps: int,
    checkpoint: Callable[[int, SIPScore], None] | None = None,
    checkpoint_interval: int = 1,
) -> SIPScore:
    score = SIPScore.create(
        predictor_observation_features=predictor_observations.shape[-1],
        predictor_instrument_features=instruments.shape[-1],
        observation_features=1,
        hidden_features=(),
        streams=create_streams({"parameters": jr.fold_in(key, 0), "inference": jr.fold_in(key, 1)}),
    )
    gain = jnp.ones((target.shape[0], 1))
    streams = create_streams({"inference": jr.fold_in(key, 2)})
    if checkpoint is None:
        score, _ = train_score_adversarial(
            score,
            target,
            predictor_observations,
            instruments,
            gain,
            steps=steps,
            predictor_learning_rate=0.005,
            witness_learning_rate=0.005,
            confounding_weight=1.0,
            streams=streams,
        )
        return score

    if checkpoint_interval < 1:
        msg = "checkpoint_interval must be positive"
        raise ValueError(msg)
    checkpoint(0, score)
    completed_steps = 0
    while completed_steps < steps:
        chunk_steps = min(checkpoint_interval, steps - completed_steps)
        score, _ = train_score_adversarial(
            score,
            target,
            predictor_observations,
            instruments,
            gain,
            steps=chunk_steps,
            predictor_learning_rate=0.005,
            witness_learning_rate=0.005,
            confounding_weight=1.0,
            streams=streams,
        )
        completed_steps += chunk_steps
        checkpoint(completed_steps, score)
    return score


def _causal_result(
    score: SIPScore,
    predictor_observations: jnp.ndarray,
    instruments: jnp.ndarray,
    target: jnp.ndarray,
    *,
    source_index: int,
    true_effect: float,
    key: jnp.ndarray,
) -> CausalBenchmarkResult:
    gain = jnp.ones((target.shape[0], 1))
    output = score.infer(
        target,
        predictor_observations,
        instruments,
        gain,
        streams=create_streams({"inference": key}),
        inference=True,
    )
    shifted = predictor_observations.at[:, source_index].add(1.0)
    shifted_output = score.infer(
        target,
        shifted,
        instruments,
        gain,
        streams=create_streams({"inference": key}),
        inference=True,
    )
    estimated_effect = jnp.mean(shifted_output.prediction - output.prediction)
    covariance = jnp.mean(output.observation_score[:, 0] * instruments[:, 0])
    return CausalBenchmarkResult(
        true_effect=true_effect,
        estimated_effect=float(estimated_effect),
        residual_instrument_covariance=float(covariance),
        reconstruction_loss=float(jnp.mean(output.reconstruction_loss)),
    )


@dataclass
class _CausalTrajectoryRecorder:
    predictor_observations: jnp.ndarray
    predictor_instruments: jnp.ndarray
    observation: jnp.ndarray
    source_index: int
    true_effect: float
    count: int
    key: jnp.ndarray
    training_examples: list[int] = field(default_factory=list)
    estimated_effects: list[float] = field(default_factory=list)
    reconstruction_losses: list[float] = field(default_factory=list)

    def __call__(self, completed_steps: int, score: SIPScore) -> None:
        result = _causal_result(
            score,
            self.predictor_observations,
            self.predictor_instruments,
            self.observation,
            source_index=self.source_index,
            true_effect=self.true_effect,
            key=self.key,
        )
        self.training_examples.append(completed_steps * self.count)
        self.estimated_effects.append(result.estimated_effect)
        self.reconstruction_losses.append(result.reconstruction_loss)

    def trajectory(self) -> CausalBenchmarkTrajectory:
        return CausalBenchmarkTrajectory(
            training_examples=tuple(self.training_examples),
            estimated_effects=tuple(self.estimated_effects),
            reconstruction_losses=tuple(self.reconstruction_losses),
        )


@dataclass
class _InstrumentTrajectoryRecorder:
    predictor_instruments: jnp.ndarray
    count: int
    training_examples: list[int] = field(default_factory=list)
    instrument_magnitudes: list[float] = field(default_factory=list)

    def __call__(self, completed_steps: int, emitter: SIPEmitter, _loss: jnp.ndarray) -> None:
        instrument = emitter.infer_inherited_instrument(self.predictor_instruments)
        self.training_examples.append(completed_steps * self.count)
        self.instrument_magnitudes.append(float(jnp.sqrt(jnp.mean(jnp.square(instrument)))))

    def values(self) -> tuple[tuple[int, ...], tuple[float, ...]]:
        return tuple(self.training_examples), tuple(self.instrument_magnitudes)


def run_intention_sensation_benchmark(
    *,
    count: int = 128,
    steps: int = 200,
    seed: int = 100,
) -> CausalBenchmarkResult:
    """Identify an unconfounded intention-to-sensation effect with instrument(A)."""
    beta = 1.7
    base = jr.normal(jr.key(seed), (count,))
    instrument_a = jr.normal(jr.key(seed + 1), (count,))
    intention_a = base + instrument_a
    observation_y = (beta * intention_a + 0.1 * jr.normal(jr.key(seed + 2), (count,)))[
        :, jnp.newaxis
    ]
    predictor_observations = intention_a[:, jnp.newaxis]
    predictor_instruments = instrument_a[:, jnp.newaxis]
    score = _fit_causal_score(
        predictor_observations,
        predictor_instruments,
        observation_y,
        key=jr.key(seed + 3),
        steps=steps,
    )
    return _causal_result(
        score,
        predictor_observations,
        predictor_instruments,
        observation_y,
        source_index=0,
        true_effect=beta,
        key=jr.key(seed + 4),
    )


def run_direct_injection_benchmark(
    *,
    count: int = 128,
    steps: int = 300,
    seed: int = 200,
) -> dict[str, CausalBenchmarkResult]:
    """Compare intention-effect recovery with zero and random injected noise."""
    intention_to_future_effect = 1.7
    past_to_future_effect = 2.2
    observation_x = jr.normal(jr.key(seed), (count,))
    injected_noise = jr.normal(jr.key(seed + 1), (count,))
    observation_noise_y = 0.1 * jr.normal(jr.key(seed + 2), (count,))
    results: dict[str, CausalBenchmarkResult] = {}
    for name, noise_magnitude in {"zero": 0.0, "random": 1.0}.items():
        instrument_a = noise_magnitude * injected_noise
        intention_a = 0.9 * observation_x + instrument_a
        observation_y = (
            intention_to_future_effect * intention_a
            + past_to_future_effect * observation_x
            + observation_noise_y
        )[:, jnp.newaxis]
        predictor_observations = jnp.stack((observation_x, intention_a), axis=-1)
        predictor_instruments = instrument_a[:, jnp.newaxis]
        recorder = _CausalTrajectoryRecorder(
            predictor_observations=predictor_observations,
            predictor_instruments=predictor_instruments,
            observation=observation_y,
            source_index=1,
            true_effect=intention_to_future_effect,
            count=count,
            key=jr.key(seed + 4),
        )

        score = _fit_causal_score(
            predictor_observations,
            predictor_instruments,
            observation_y,
            key=jr.key(seed + 3),
            steps=steps,
            checkpoint=recorder,
            checkpoint_interval=max(1, steps // 96),
        )
        final_result = _causal_result(
            score,
            predictor_observations,
            predictor_instruments,
            observation_y,
            source_index=1,
            true_effect=intention_to_future_effect,
            key=jr.key(seed + 4),
        )
        results[name] = replace(
            final_result,
            trajectory=recorder.trajectory(),
        )
    return results


def run_inherited_instrument_benchmark(  # ruff: ignore[too-many-locals]
    *,
    count: int = 128,
    steps: int = 300,
    seed: int = 300,
) -> dict[str, CausalBenchmarkResult]:
    """Learn instrument(Y) and use it to identify Y's effect on Z."""
    intention_policy_gain = 0.8
    intention_to_future_effect = 1.3
    past_to_future_effect = 0.5
    future_to_subsequent_effect = 1.7
    past_to_subsequent_effect = 2.2
    observation_x = jr.normal(jr.key(seed), (count,))
    injected_noise = jr.normal(jr.key(seed + 1), (count,))
    subsequent_noise = 0.1 * jr.normal(jr.key(seed + 2), (count,))
    conditions = {
        "policy": (True, False),
        "injected": (True, True),
    }
    results: dict[str, CausalBenchmarkResult] = {}
    for name, (use_policy, use_noise) in conditions.items():
        instrument_a = injected_noise if use_noise else jnp.zeros_like(injected_noise)
        policy_intention = (
            intention_policy_gain * observation_x if use_policy else jnp.zeros_like(observation_x)
        )
        intention_a = policy_intention + instrument_a
        observation_y = (
            intention_to_future_effect * intention_a + past_to_future_effect * observation_x
        )[:, jnp.newaxis]
        instrument_a = instrument_a[:, jnp.newaxis]
        future_emitter = SIPEmitter.create(
            innovation_features=1,
            goal_features=1,
            predictor_instrument_features=1,
            observation_features=1,
            hidden_features=(),
            initial_noise=1e-6,
            learn_noise=False,
            streams=create_streams(
                {
                    "parameters": jr.fold_in(jr.key(seed + 5), 0),
                    "inference": jr.fold_in(jr.key(seed + 5), 1),
                }
            ),
        )
        instrument_recorder = _InstrumentTrajectoryRecorder(
            predictor_instruments=instrument_a,
            count=count,
        )
        future_emitter, _ = train_instrument_map(
            future_emitter,
            observation_y,
            instrument_a,
            steps=steps,
            learning_rate=0.01,
            checkpoint=instrument_recorder,
            checkpoint_interval=max(1, steps // 96),
        )
        instrument_y = future_emitter.infer_inherited_instrument(instrument_a)
        observation_z = (
            future_to_subsequent_effect * observation_y[:, 0]
            + past_to_subsequent_effect * observation_x
            + subsequent_noise
        )[:, jnp.newaxis]
        predictor_observations = jnp.stack((observation_x, observation_y[:, 0]), axis=-1)
        score_recorder = _CausalTrajectoryRecorder(
            predictor_observations=predictor_observations,
            predictor_instruments=instrument_y,
            observation=observation_z,
            source_index=1,
            true_effect=future_to_subsequent_effect,
            count=count,
            key=jr.key(seed + 4),
        )
        score = _fit_causal_score(
            predictor_observations,
            instrument_y,
            observation_z,
            key=jr.key(seed + 3),
            steps=steps,
            checkpoint=score_recorder,
            checkpoint_interval=max(1, steps // 96),
        )
        causal_result = _causal_result(
            score,
            predictor_observations,
            instrument_y,
            observation_z,
            source_index=1,
            true_effect=future_to_subsequent_effect,
            key=jr.key(seed + 4),
        )
        training_examples, instrument_magnitudes = instrument_recorder.values()
        score_trajectory = score_recorder.trajectory()
        if training_examples != score_trajectory.training_examples:
            msg = "inherited benchmark stages recorded different training checkpoints"
            raise ValueError(msg)
        first_stage_score = instrument_y - observation_y
        results[name] = replace(
            causal_result,
            true_first_stage_effect=intention_to_future_effect if use_noise else None,
            estimated_first_stage_effect=(
                float(future_emitter.instrument_map.weight.value[0, 0]) if use_noise else None
            ),
            first_stage_residual_covariance=(
                float(jnp.mean(first_stage_score[:, 0] * instrument_a[:, 0])) if use_noise else None
            ),
            trajectory=replace(
                score_trajectory,
                instrument_magnitudes=instrument_magnitudes,
            ),
        )
    return results


def _credit_trajectory(
    training_examples: list[int],
    estimated_effects: list[float],
    reconstruction_losses: list[float],
    *,
    true_effect: float,
) -> CreditBenchmarkResult:
    return CreditBenchmarkResult(
        true_effect=true_effect,
        estimated_effect=estimated_effects[-1],
        reconstruction_loss=reconstruction_losses[-1],
        trajectory=CausalBenchmarkTrajectory(
            training_examples=tuple(training_examples),
            estimated_effects=tuple(estimated_effects),
            reconstruction_losses=tuple(reconstruction_losses),
        ),
    )


def _train_ordinary_credit(
    prospect_observation: jnp.ndarray,
    prospect_instrument: jnp.ndarray,
    target: jnp.ndarray,
    *,
    true_effect: float,
    count: int,
    steps: int,
    key: jnp.ndarray,
) -> CreditBenchmarkResult:
    """Fit a one-shot predictor of the whole reward sequence from the prospect alone.

    Because the predictor sees the prospect's observation directly, its prediction
    can fit that observation's own effect on every reward. Subtracting that
    prediction as an ordinary score then explains this effect away rather than
    crediting it.
    """
    horizon = target.shape[-1]
    predictor_observations = prospect_observation[:, jnp.newaxis]
    predictor_instruments = prospect_instrument[:, jnp.newaxis]
    gain = jnp.ones((count, 1))
    score = SIPScore.create(
        predictor_observation_features=1,
        predictor_instrument_features=1,
        observation_features=horizon,
        hidden_features=(),
        streams=create_streams({"parameters": jr.fold_in(key, 0), "inference": jr.fold_in(key, 1)}),
    )
    streams = create_streams({"inference": jr.fold_in(key, 2)})

    def effect_and_loss(current: SIPScore) -> tuple[float, float]:
        output = current.infer(
            target,
            predictor_observations,
            predictor_instruments,
            gain,
            streams=create_streams({"inference": jr.fold_in(key, 3)}),
            inference=True,
        )
        total_score = jnp.sum(output.observation_score, axis=-1)
        effect = float(-jnp.mean(total_score * prospect_observation))
        loss = float(jnp.mean(output.reconstruction_loss))
        return effect, loss

    checkpoint_interval = max(1, steps // 96)
    training_examples: list[int] = [0]
    effect, loss = effect_and_loss(score)
    estimated_effects: list[float] = [effect]
    reconstruction_losses: list[float] = [loss]

    completed_steps = 0
    while completed_steps < steps:
        chunk_steps = min(checkpoint_interval, steps - completed_steps)
        score, _ = train_score_adversarial(
            score,
            target,
            predictor_observations,
            predictor_instruments,
            gain,
            steps=chunk_steps,
            predictor_learning_rate=0.005,
            witness_learning_rate=0.0,
            confounding_weight=0.0,
            streams=streams,
        )
        completed_steps += chunk_steps
        effect, loss = effect_and_loss(score)
        training_examples.append(completed_steps * count)
        estimated_effects.append(effect)
        reconstruction_losses.append(loss)

    return _credit_trajectory(
        training_examples,
        estimated_effects,
        reconstruction_losses,
        true_effect=true_effect,
    )


def _train_td_credit(  # ruff: ignore[too-many-locals,too-many-statements]
    prospect_observation: jnp.ndarray,
    prospect_instrument: jnp.ndarray,
    rewards: jnp.ndarray,
    *,
    true_effect: float,
    count: int,
    steps: int,
    key: jnp.ndarray,
) -> CreditBenchmarkResult:
    """Fit a TD-error circuit whose baseline predates the prospect it credits.

    The delayed prediction at step 0 is drawn from a fixed state independent of the
    prospect's observation, so it cannot explain away that observation's effect.
    Its telescoped total credit across the episode therefore stays at the true
    effect regardless of how well the intermediate predictions, which do see the
    observation, come to fit.
    """
    horizon = rewards.shape[0]
    balance_before = 1.0 - jnp.cumsum(
        jnp.concatenate((jnp.zeros((1, count)), rewards[:-1]), axis=0), axis=0
    )
    live_state = balance_before[..., jnp.newaxis]
    live_instrument = prospect_instrument[:, jnp.newaxis]
    zero_instrument = jnp.zeros((count, 1))
    predictor_instruments = jnp.stack(
        [live_instrument if step < horizon - 1 else zero_instrument for step in range(horizon)],
        axis=0,
    )
    initial_state = live_state[0]
    initial_instrument = zero_instrument
    observations = rewards[:, :, jnp.newaxis]
    predictor_observations = live_state
    horizon = observations.shape[0]
    gains = jnp.ones((horizon, count, 1))
    initial_gain = jnp.zeros((count, 1))

    td_error = SIPTDError.create(
        predictor_observation_features=1,
        predictor_instrument_features=1,
        observation_features=1,
        discount=1.0,
        hidden_features=(),
        streams=create_streams({"parameters": jr.fold_in(key, 0), "inference": jr.fold_in(key, 1)}),
    )
    # Start with no P-to-R prediction so the plotted link strength has a clear
    # zero baseline; the hidden representation remains randomly initialized.
    td_error = eqx.tree_at(
        lambda model: model.predictor.prediction_map.layers[-1].weight.value,
        td_error,
        jnp.zeros_like(td_error.predictor.prediction_map.layers[-1].weight.value),
    )
    streams = create_streams({"inference": jr.fold_in(key, 2)})

    def effect_and_loss(current: SIPTDError) -> tuple[float, float]:
        outputs = rollout_td_error(
            current,
            observations,
            predictor_observations,
            predictor_instruments,
            gains,
            initial_state,
            initial_instrument,
            initial_gain,
            streams=create_streams({"inference": jr.fold_in(key, 3)}),
            inference=True,
        )
        total_score = sum(output.observation_score[:, 0] for output in outputs)
        effect = float(-jnp.mean(total_score * prospect_observation))
        loss = float(jnp.mean(sum(output.reconstruction_loss for output in outputs)) / horizon)
        return effect, loss

    def link_and_error(
        current: SIPTDError,
    ) -> tuple[float, float, tuple[float, ...], tuple[float, ...]]:
        outputs = rollout_td_error(
            current,
            observations,
            predictor_observations,
            predictor_instruments,
            gains,
            initial_state,
            initial_instrument,
            initial_gain,
            streams=create_streams({"inference": jr.fold_in(key, 3)}),
            inference=True,
        )
        # The first score consumes the externally supplied pseudo-reward and its
        # zero delayed baseline, so it is not a learned TD-error prediction.
        interior_outputs = outputs
        step_errors = tuple(
            float(jnp.mean(jnp.abs(output.observation_score[:, 0]))) for output in interior_outputs
        )
        step_means = tuple(
            float(-jnp.mean(output.observation_score[:, 0])) for output in interior_outputs
        )
        td_values = jnp.asarray(step_errors)
        td_magnitude = float(jnp.mean(jnp.abs(td_values)))
        link_input = predictor_observations[..., 0].reshape(-1)
        prediction = current.predictor.prediction(
            predictor_observations.reshape(-1, 1),
            jnp.ones((link_input.shape[0], 1)),
            streams=create_streams({"inference": jr.fold_in(key, 4)}),
            inference=True,
        )[:, 0]
        centred = link_input - jnp.mean(link_input)
        link_strength = float(
            jnp.mean(centred * (prediction - jnp.mean(prediction)))
            / (jnp.mean(jnp.square(centred)) + 1e-8)
        )
        return link_strength, td_magnitude, step_errors, step_means

    checkpoint_interval = max(1, steps // 4)
    training_examples: list[int] = [0]
    effect, loss = effect_and_loss(td_error)
    link_strength, td_magnitude, step_errors, step_means = link_and_error(td_error)
    estimated_effects: list[float] = [effect]
    reconstruction_losses: list[float] = [loss]
    link_strengths: list[float] = [link_strength]
    expected_td_errors: list[float] = [td_magnitude]
    td_error_by_step: list[tuple[float, ...]] = [step_errors]
    td_error_mean_by_step: list[tuple[float, ...]] = [step_means]

    completed_steps = 0
    while completed_steps < steps:
        chunk_steps = min(checkpoint_interval, steps - completed_steps)
        td_error, _ = train_td_error_adversarial(
            td_error,
            observations,
            predictor_observations,
            predictor_instruments,
            gains,
            initial_state,
            initial_instrument,
            initial_gain,
            steps=chunk_steps,
            predictor_learning_rate=0.4,
            witness_learning_rate=0.0,
            confounding_weight=0.0,
            streams=streams,
        )
        completed_steps += chunk_steps
        effect, loss = effect_and_loss(td_error)
        link_strength, td_magnitude, step_errors, step_means = link_and_error(td_error)
        training_examples.append(completed_steps)
        estimated_effects.append(effect)
        reconstruction_losses.append(loss)
        link_strengths.append(link_strength)
        expected_td_errors.append(td_magnitude)
        td_error_by_step.append(step_errors)
        td_error_mean_by_step.append(step_means)

    result = _credit_trajectory(
        training_examples,
        estimated_effects,
        reconstruction_losses,
        true_effect=true_effect,
    )
    assert result.trajectory is not None
    return replace(
        result,
        trajectory=replace(
            result.trajectory,
            link_strengths=tuple(link_strengths),
            expected_td_errors=tuple(expected_td_errors),
            td_error_by_step=tuple(td_error_by_step),
            td_error_mean_by_step=tuple(td_error_mean_by_step),
        ),
    )


def run_td_error_benchmark(
    *,
    count: int = 128,
    steps: int = 300,
    seed: int = 400,
) -> dict[str, CreditBenchmarkResult]:
    """Train ordinary and TD baselines on a sequential remaining-food process."""
    if count < 1 or steps < 1:
        msg = "count and steps must be positive"
        raise ValueError(msg)
    balance = simulate_remaining_food(count=count, n=16, variance=0.01, seed=seed)
    prospect_observation = balance[:, 0]
    prospect_instrument = jnp.zeros((count,))
    rewards = jnp.moveaxis(balance[:, :-1] - balance[:, 1:], 0, 1)
    true_effect = 0.0
    target = jnp.moveaxis(rewards, 0, -1)

    ordinary = _train_ordinary_credit(
        prospect_observation,
        prospect_instrument,
        target,
        true_effect=true_effect,
        count=count,
        steps=steps,
        key=jr.key(seed + 10),
    )
    td = _train_td_credit(
        prospect_observation,
        prospect_instrument,
        rewards,
        true_effect=true_effect,
        count=count,
        steps=steps,
        key=jr.key(seed + 20),
    )
    return {"ordinary": ordinary, "td": td}
