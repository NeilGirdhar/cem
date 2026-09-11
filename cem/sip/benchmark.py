"""Reproducible synthetic benchmarks for the real-valued SIP chain."""

from dataclasses import dataclass

import jax.numpy as jnp
import jax.random as jr
from tjax import create_streams

from cem.sip.model import SIPChain
from cem.sip.score import SIPScore
from cem.sip.training import (
    SIPTrainingHistory,
    train_chain_adversarial,
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
) -> tuple[SIPChain, SIPTrainingHistory]:
    training_innovation, _, goal, parent_instruments, gain, target = data
    chain = SIPChain.create(
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
    return train_chain_adversarial(
        chain,
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
    """Compare ordinary, noisy, and purified chains on shifted confounding."""
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
        chain, history = _train_condition(
            initial_noise=noise,
            confounding_weight=confounding,
            witness_learning_rate=witness_rate,
            data=data,
            steps=steps,
            key=parameter_key,
        )
        output = chain.infer(
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
            noise_magnitudes=tuple(float(value) for value in chain.emitter.noise_magnitudes),
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
    action_index: int,
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
    shifted = predictor_observations.at[:, action_index].add(1.0)
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


def run_action_sensation_benchmark(
    *,
    count: int = 128,
    steps: int = 200,
    seed: int = 100,
) -> CausalBenchmarkResult:
    """Identify an action-to-sensation effect with an exogenous action instrument."""
    beta = 1.7
    base = jr.normal(jr.key(seed), (count,))
    instrument = jr.normal(jr.key(seed + 1), (count,))
    action = base + instrument
    target = (beta * action + 0.1 * jr.normal(jr.key(seed + 2), (count,)))[:, jnp.newaxis]
    observations = action[:, jnp.newaxis]
    instruments = instrument[:, jnp.newaxis]
    score, _ = _fit_causal_score(
        observations,
        instruments,
        target,
        key=jr.key(seed + 3),
        steps=steps,
    )
    return _causal_result(
        score,
        observations,
        instruments,
        target,
        action_index=0,
        true_effect=beta,
        key=jr.key(seed + 4),
    )


def run_confounded_action_benchmark(
    *,
    count: int = 128,
    steps: int = 300,
    seed: int = 200,
) -> CausalBenchmarkResult:
    """Identify an action effect when past sensation confounds action and target."""
    beta = 1.7
    gamma = 2.2
    past_sensation = jr.normal(jr.key(seed), (count,))
    instrument = jr.normal(jr.key(seed + 1), (count,))
    action = 0.9 * past_sensation + instrument + 0.1 * jr.normal(jr.key(seed + 2), (count,))
    target = (beta * action + gamma * past_sensation + 0.1 * jr.normal(jr.key(seed + 3), (count,)))[
        :, jnp.newaxis
    ]
    observations = jnp.stack((past_sensation, action), axis=-1)
    instruments = instrument[:, jnp.newaxis]
    score, _ = _fit_causal_score(
        observations,
        instruments,
        target,
        key=jr.key(seed + 4),
        steps=steps,
    )
    return _causal_result(
        score,
        observations,
        instruments,
        target,
        action_index=1,
        true_effect=beta,
        key=jr.key(seed + 5),
    )
