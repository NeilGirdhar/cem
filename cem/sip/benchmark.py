"""Reproducible synthetic benchmarks for real-valued SIP."""

from dataclasses import dataclass

import jax.numpy as jnp
import jax.random as jr
from tjax import create_streams

from cem.sip.emitter import SIPEmitter
from cem.sip.explanatory_coupling import ExplanatoryCoupling
from cem.sip.score import SIPScore
from cem.sip.training import (
    SIPTrainingHistory,
    train_explanatory_coupling_adversarial,
    train_instrument_map,
    train_score_adversarial,
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
class CausalBenchmarkResult:
    """Estimated causal effect and residual diagnostics for a SIP task."""

    true_effect: float
    estimated_effect: float
    residual_instrument_covariance: float
    reconstruction_loss: float
    true_first_stage_effect: float | None = None
    estimated_first_stage_effect: float | None = None
    first_stage_residual_covariance: float | None = None


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
) -> tuple[SIPScore, jnp.ndarray]:
    score = SIPScore.create(
        predictor_observation_features=predictor_observations.shape[-1],
        predictor_instrument_features=instruments.shape[-1],
        observation_features=1,
        hidden_features=(),
        streams=create_streams({"parameters": jr.fold_in(key, 0), "inference": jr.fold_in(key, 1)}),
    )
    score, _ = train_score_adversarial(
        score,
        target,
        predictor_observations,
        instruments,
        jnp.ones((target.shape[0], 1)),
        steps=steps,
        predictor_learning_rate=0.005,
        witness_learning_rate=0.005,
        confounding_weight=1.0,
        streams=create_streams({"inference": jr.fold_in(key, 2)}),
    )
    return score, jnp.ones((target.shape[0], 1))


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
    score, _ = _fit_causal_score(
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
        score, _ = _fit_causal_score(
            predictor_observations,
            predictor_instruments,
            observation_y,
            key=jr.key(seed + 3),
            steps=steps,
        )
        results[name] = _causal_result(
            score,
            predictor_observations,
            predictor_instruments,
            observation_y,
            source_index=1,
            true_effect=intention_to_future_effect,
            key=jr.key(seed + 4),
        )
    return results


def run_inherited_instrument_benchmark(
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
        "inactive": (False, False),
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
        future_emitter, _ = train_instrument_map(
            future_emitter,
            observation_y,
            instrument_a,
            steps=steps,
            learning_rate=0.01,
        )
        instrument_y = future_emitter.infer_inherited_instrument(instrument_a)
        observation_z = (
            future_to_subsequent_effect * observation_y[:, 0]
            + past_to_subsequent_effect * observation_x
            + subsequent_noise
        )[:, jnp.newaxis]
        predictor_observations = jnp.stack((observation_x, observation_y[:, 0]), axis=-1)
        score, _ = _fit_causal_score(
            predictor_observations,
            instrument_y,
            observation_z,
            key=jr.key(seed + 3),
            steps=steps,
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
        first_stage_score = instrument_y - observation_y
        results[name] = CausalBenchmarkResult(
            true_effect=causal_result.true_effect,
            estimated_effect=causal_result.estimated_effect,
            residual_instrument_covariance=causal_result.residual_instrument_covariance,
            reconstruction_loss=causal_result.reconstruction_loss,
            true_first_stage_effect=intention_to_future_effect if use_noise else None,
            estimated_first_stage_effect=(
                float(future_emitter.instrument_map.weight.value[0, 0]) if use_noise else None
            ),
            first_stage_residual_covariance=(
                float(jnp.mean(first_stage_score[:, 0] * instrument_a[:, 0])) if use_noise else None
            ),
        )
    return results
