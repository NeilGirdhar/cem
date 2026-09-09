"""Reproducible synthetic benchmarks for the real-valued SIP chain."""

from dataclasses import dataclass

import jax.numpy as jnp
import jax.random as jr
from tjax import create_streams

from cem.sip.model import SIPChain
from cem.sip.training import SIPTrainingHistory, train_chain_adversarial


@dataclass(frozen=True)
class SIPBenchmarkResult:
    """Metrics from one synthetic SIP training condition."""

    training_loss: float
    inference_loss: float
    residual_instrument_correlation: float
    witness_loss: float
    noise_magnitudes: tuple[float, ...]


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
    for index, (name, (noise, confounding, witness_rate)) in enumerate(conditions.items()):
        chain, history = _train_condition(
            initial_noise=noise,
            confounding_weight=confounding,
            witness_learning_rate=witness_rate,
            data=data,
            steps=steps,
            key=jr.fold_in(jr.key(seed + 1), index),
        )
        output = chain.infer(
            shifted_innovation,
            goal,
            gain,
            parent_instruments,
            target,
            gain,
            streams=create_streams({"inference": jr.key(seed + 2 + index)}),
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
